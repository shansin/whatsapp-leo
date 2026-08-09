"""Core message processing for WhatsApp Leo agent."""

import asyncio
import base64
import contextlib
import sqlite3
from dataclasses import asdict
import re
from datetime import datetime, timedelta
from io import BytesIO

import orjson
from agents import RunConfig, Runner, trace
from PIL import Image
# config must come first: it puts whatsapp-mcp-server on sys.path.
from config import (
    IS_DEDICATED_NUMBER,
    ALLOWED_SENDERS,
    HOOKS,
    IS_HOOK_ENABLED,
    LEO_MENTION_ID,
    MAX_IMAGE_DIMENSION,
    MESSAGES_DB_PATH,
    PRESENCE_ENABLED,
    PRESENCE_REFRESH_SECONDS,
    TRACING_ENABLED,
    WHISPER_BEAM_SIZE,
    WHISPER_VAD_FILTER,
    _cached_model,
    _cached_vision_model,
    get_whisper_model,
)
from whatsapp import send_message as whatsapp_send_message
from whatsapp import download_media, send_presence
from whatsapp import send_audio_message as whatsapp_send_audio
from mcp_pool import mcp_pool
import write_guard
import user_prefs
from instructions import INSTRUCTIONS_PRIVILEGED_TEMPLATE, INSTRUCTIONS_BASIC_TEMPLATE
from memory import load_memory, make_memory_tools
from history_tools import make_history_tools
from send_tools import make_send_tools
from reply import reply_to as _reply
from reply import send_reply, send_voice_reply
from debounce import debouncer, merge_content
from models import ReceivedMessage
from agent_factory import agent_factory, parse_remindme_with_agent
from session_store import trim_session
from command_handlers import (
    handle_briefing_command,
    handle_help_command,
    handle_model_command,
    handle_remindme_cancel,
    handle_remindme_list,
    handle_reminder_command,
    handle_status_command,
    handle_tz_command,
)
from reminder import validate_reminder_time, store_reminder
from hooks import match_hook, match_hook_session_command, write_to_hook, start_hook_session, stop_hook_session, get_hook_session
from logging_setup import logger


def _lookup_quoted_message(message_id: str, chat_jid: str) -> tuple[str, str, str]:
    """Look up a quoted message's content, sender, and media_type from the messages DB.

    Returns (content, sender, media_type) or ("", "", "") if not found.
    """
    try:
        conn = sqlite3.connect(MESSAGES_DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT content, sender, media_type FROM messages WHERE id = ? AND chat_jid = ?",
            (message_id, chat_jid),
        )
        row = cursor.fetchone()
        conn.close()
        if row:
            return row[0] or "", row[1] or "", row[2] or ""
    except Exception:
        pass
    return "", "", ""


def format_leo_response(text: str) -> str:
    return f"_*(Leo)*_ {text}" if not IS_DEDICATED_NUMBER else text


def is_own_output(message: ReceivedMessage) -> bool:
    """True if Leo sent this message itself, rather than the user typing it.

    ``is_from_me`` alone cannot answer this, because the answer depends on who
    owns the account:

    * Dedicated number — Leo owns it and the user is on another number, so
      anything from us is our own output. The guard is needed: in this mode any
      DM triggers a response, so Leo would answer its own replies.
    * Shared number — the user types from this same linked device, so their
      messages are ``is_from_me`` too. What separates the two is the #leo /
      @leo mention, which only the user's messages carry; ``_should_respond``
      already enforces it. Guarding on ``is_from_me`` here would drop every
      command the user sends.
    """
    return message.is_from_me and IS_DEDICATED_NUMBER


async def _keep_typing(chat_jid: str) -> None:
    """Refresh the WhatsApp typing indicator until cancelled."""
    try:
        while True:
            ok, detail = await asyncio.to_thread(send_presence, chat_jid, "composing")
            if not ok:
                logger.debug(f"Typing indicator failed for {chat_jid}: {detail}")
                return  # bridge doesn't support it / not connected — stop trying
            await asyncio.sleep(PRESENCE_REFRESH_SECONDS)
    except asyncio.CancelledError:
        raise
    except Exception as e:  # never let presence break a run
        logger.debug(f"Typing indicator stopped for {chat_jid}: {e}")


@contextlib.asynccontextmanager
async def typing(chat_jid: str):
    """Show "typing…" in the chat for the duration of the block.

    Local models can take 30+ seconds; without this the user cannot tell
    "thinking" from "dead".
    """
    if not PRESENCE_ENABLED:
        yield
        return

    task = asyncio.create_task(_keep_typing(chat_jid), name=f"typing-{chat_jid}")
    try:
        yield
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        with contextlib.suppress(Exception):
            await asyncio.to_thread(send_presence, chat_jid, "paused")


def _load_and_resize_image(file_path: str) -> Image.Image:
    """Open and downscale the image if it exceeds MAX_IMAGE_DIMENSION."""
    img: Image.Image = Image.open(file_path).convert("RGB")
    w, h = img.size
    if max(w, h) > MAX_IMAGE_DIMENSION:
        scale = MAX_IMAGE_DIMENSION / max(w, h)
        img = img.resize(
            (int(w * scale), int(h * scale)), Image.Resampling.LANCZOS
        )
    return img


def _encode_image_b64(img: Image.Image) -> str:
    """Encode PIL image to base64 JPEG string."""
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("ascii")


async def _build_vision_input(
    image_message_id: str, chat_jid: str, text_payload: str
) -> list | None:
    """Download and encode the image, returning a multimodal input list.

    Returns None if the download fails, signaling fallback to text-only.
    """
    file_path = await asyncio.to_thread(download_media, image_message_id, chat_jid)
    if not file_path:
        logger.warning(f"Could not download image for message {image_message_id}")
        return None

    try:
        img = await asyncio.to_thread(_load_and_resize_image, file_path)
        image_b64 = _encode_image_b64(img)
    except Exception as e:
        logger.error(f"Failed to process image file {file_path}: {e}", exc_info=True)
        return None

    # _encode_image_b64 always re-encodes to JPEG, whatever the source format,
    # so labelling the data URI with the original extension was a lie.
    data_uri = f"data:image/jpeg;base64,{image_b64}"

    return [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": text_payload},
                {"type": "input_image", "image_url": data_uri},
            ],
        }
    ]


def _transcribe_audio(file_path: str) -> str:
    """Transcribe an audio file using faster-whisper. Returns transcript text."""
    model = get_whisper_model()
    segments, info = model.transcribe(
        file_path, beam_size=WHISPER_BEAM_SIZE, vad_filter=WHISPER_VAD_FILTER
    )
    transcript = " ".join(segment.text.strip() for segment in segments)
    logger.info(f"Transcribed audio ({info.language}, {info.duration:.1f}s): {transcript[:100]}...")
    return transcript


async def _transcribe_audio_message(
    audio_message_id: str, chat_jid: str
) -> str | None:
    """Download and transcribe an audio message. Returns transcript or None."""
    file_path = await asyncio.to_thread(download_media, audio_message_id, chat_jid)
    if not file_path:
        logger.warning(f"Could not download audio for message {audio_message_id}")
        return None

    try:
        return await asyncio.to_thread(_transcribe_audio, file_path)
    except Exception as e:
        logger.error(f"Failed to transcribe audio {file_path}: {e}", exc_info=True)
        return None


# Commands Leo answers. Matched as a prefix on the whole message, so a
# sentence that merely mentions "#briefing" is treated as normal chat.
COMMAND_PREFIXES = (
    "#remindme", "#reminder", "#briefing", "#tz", "#help", "#status", "#model",
)

# Reply "snooze 15m" to a fired reminder to push it back.
SNOOZE_RE = re.compile(
    r"^\s*snooze\s+(\d+)\s*(m|min|mins|minute|minutes|h|hr|hrs|hour|hours|d|day|days)?\s*$",
    re.IGNORECASE,
)
REMINDER_HEADER = "⏰ *Reminder*"


def match_command(content: str) -> str | None:
    """Return the command a message invokes, or None.

    Substring matching used to hijack any message containing "#reminder"
    anywhere — and silently drop it for non-privileged senders.
    """
    stripped = content.strip().lower()
    for prefix in COMMAND_PREFIXES:
        if stripped == prefix or stripped.startswith(prefix + " "):
            return prefix
    return None


def parse_snooze(content: str) -> timedelta | None:
    """Parse "snooze 10m" into a duration, or None if it isn't a snooze."""
    match = SNOOZE_RE.match(content or "")
    if not match:
        return None
    amount = int(match.group(1))
    unit = (match.group(2) or "m").lower()
    if unit.startswith("h"):
        return timedelta(hours=amount)
    if unit.startswith("d"):
        return timedelta(days=amount)
    return timedelta(minutes=amount)


async def _handle_snooze(message: ReceivedMessage, delay: timedelta) -> bool:
    """Re-arm the reminder the user replied to. True if it was handled."""
    quoted = (message.quoted_message_content or "").strip()
    if not quoted.startswith(REMINDER_HEADER):
        return False

    text = quoted[len(REMINDER_HEADER):].strip() or "Reminder"
    user_tz = user_prefs.get_tz(message.phone_number)
    remind_at = datetime.now(user_tz) + delay

    store_reminder(
        message.chat_jid,
        text,
        remind_at,
        message_id=message.id,
        sender_jid=message.sender_jid,
    )
    when = remind_at.strftime("%b %d, %I:%M %p %Z")
    await _reply(message, f"😴 Snoozed — I'll remind you again at *{when}*.")
    logger.info(f"Snoozed reminder in {message.chat_jid} until {when}")
    return True


async def _handle_remindme(message: ReceivedMessage) -> None:
    """Parse and store a one-shot #remindme reminder."""
    stripped = message.content.strip()
    parts = stripped.split(maxsplit=2)
    subcommand = parts[1].lower() if len(parts) > 1 else ""
    if subcommand == "list":
        await handle_remindme_list(message)
        return
    if subcommand == "cancel":
        await handle_remindme_cancel(message, parts)
        return

    user_tz = user_prefs.get_tz(message.phone_number)
    try:
        remind_at, original_msg = await parse_remindme_with_agent(
            message.content, tz=user_tz
        )
        validate_reminder_time(remind_at)
        store_reminder(
            message.chat_jid,
            original_msg,
            remind_at,
            message_id=message.id,
            sender_jid=message.sender_jid,
        )
        confirm_time = remind_at.astimezone(user_tz).strftime("%b %d, %I:%M %p %Z")
        await _reply(message, f"⏰ Reminder set for *{confirm_time}*!")
        logger.info(f"Reminder set for {confirm_time} in {message.chat_jid}")
    except ValueError as e:
        await _reply(message, f"❌ {e}")
    except Exception as e:
        logger.error(f"Error handling #remindme: {e}", exc_info=True)
        await _reply(message, "❌ Something went wrong setting the reminder.")



# ── Pipeline stages ─────────────────────────────────────────────────────────
# process_message used to be one ~230-line function. It is now a gate → enrich
# → media → run → reply pipeline; each stage returns True if it fully handled
# the message.


async def _handle_hook_commands(message: ReceivedMessage) -> bool:
    """Hook session start/stop and single-message hook intercepts."""
    session_cmd = match_hook_session_command(message.content)
    if session_cmd and message.phone_number in ALLOWED_SENDERS:
        hook_name, action = session_cmd
        if action == "start":
            start_hook_session(message.chat_jid, hook_name)
            await _reply(
                message,
                f"🔗 Hook session *{hook_name}* started. All messages will be "
                f"forwarded. Send *#{hook_name} #stop* to end.",
            )
        else:
            stopped = stop_hook_session(message.chat_jid)
            if stopped:
                await _reply(
                    message,
                    f"Hook session *{stopped}* ended. Regular processing resumed.",
                )
            else:
                await _reply(message, "No active hook session to stop.")
        return True

    hook_match = match_hook(message.content)
    if hook_match and message.phone_number in ALLOWED_SENDERS:
        hook_name, body = hook_match
        await write_to_hook(hook_name, body)
        return True

    return False


async def _enrich_quoted(message: ReceivedMessage) -> str:
    """Fill in a quoted message from the DB. Returns its media type."""
    if not message.quoted_message_id:
        return ""

    db_content, db_sender, quoted_media_type = await asyncio.to_thread(
        _lookup_quoted_message, message.quoted_message_id, message.chat_jid
    )
    if db_content and not message.quoted_message_content:
        message.quoted_message_content = db_content
    if db_sender and not message.quoted_message_sender:
        message.quoted_message_sender = db_sender
    return quoted_media_type


async def _handle_commands(message: ReceivedMessage) -> bool:
    """Dispatch a #command. Returns True if the message was a command."""
    command = match_command(message.content)
    if not command:
        return False

    if message.phone_number not in ALLOWED_SENDERS:
        # Previously these messages were dropped without a word.
        await _reply(message, f"🔒 *{command}* is only available to Leo's owner.")
        return True

    if command == "#remindme":
        await _handle_remindme(message)
    elif command == "#reminder":
        await handle_reminder_command(message)
    elif command == "#briefing":
        await handle_briefing_command(message)
    elif command == "#tz":
        await handle_tz_command(message)
    elif command == "#status":
        await handle_status_command(message)
    elif command == "#model":
        await handle_model_command(message)
    elif command == "#help":
        await handle_help_command(
            message, hook_names=HOOKS if IS_HOOK_ENABLED else ()
        )
    return True


def _should_respond(message: ReceivedMessage) -> bool:
    """Whether this message is addressed to Leo at all."""
    if IS_DEDICATED_NUMBER:
        # A DM (ends with @lid), or a group message that @-mentions Leo.
        is_dm = message.chat_jid.endswith("@lid")
        is_group_mention = (
            "@g" in message.chat_jid and LEO_MENTION_ID in message.content
        )
        return is_dm or is_group_mention

    lowered = message.content.lower()
    return "#leo" in lowered or "@leo" in lowered


def _detect_media(message: ReceivedMessage, quoted_media_type: str) -> dict:
    """Locate an image or voice note, whether sent directly or replied to."""
    is_direct_image = message.media_type == "image"
    is_quoted_image = (
        not is_direct_image
        and message.quoted_message_id != ""
        and quoted_media_type == "image"
    )
    is_direct_audio = message.media_type == "audio"
    is_quoted_audio = (
        not is_direct_audio
        and message.quoted_message_id != ""
        and quoted_media_type == "audio"
    )
    return {
        "has_image": is_direct_image or is_quoted_image,
        "image_id": message.id if is_direct_image else message.quoted_message_id,
        "has_audio": is_direct_audio or is_quoted_audio,
        "audio_id": message.id if is_direct_audio else message.quoted_message_id,
    }


def _build_instructions(message: ReceivedMessage, is_allowed: bool) -> str:
    """System prompt for this sender, with their clock and memory."""
    # The clock Leo reports must be the sender's, not the server's.
    user_tz = user_prefs.get_tz(message.phone_number)
    current_time = datetime.now(user_tz).strftime("%I:%M %p %Z, %A %B %d, %Y")

    template = (
        INSTRUCTIONS_PRIVILEGED_TEMPLATE if is_allowed else INSTRUCTIONS_BASIC_TEMPLATE
    )
    instructions = template.format(current_time=current_time)

    memory_content = load_memory(message.phone_number)
    if memory_content:
        instructions += (
            "\n\n[USER MEMORY]\nThese are things this user has asked you to "
            f"remember:\n{memory_content}"
        )
    return instructions


def _build_tools(message: ReceivedMessage, is_allowed: bool) -> list:
    """Local (non-MCP) tools available to this sender."""
    tools = make_memory_tools(message.phone_number)
    if is_allowed:
        # The full message archive, and the ability to send files back —
        # privileged senders only.
        tools += make_history_tools(message.chat_jid, message.phone_number)
        tools += make_send_tools(message.chat_jid)
    return tools


async def _build_input(message: ReceivedMessage, media: dict, notices: list[str]):
    """Assemble the model input, transcribing audio and encoding images.

    Returns (runner_input, model).
    """
    if media["has_audio"]:
        transcript = await _transcribe_audio_message(
            media["audio_id"], message.chat_jid
        )
        if transcript:
            message.content = (
                f"{message.content}\n\n[Audio transcript]: {transcript}"
                if message.content
                else f"[Audio transcript]: {transcript}"
            )
        else:
            notices.append(
                "I couldn't transcribe that voice note, so I answered from the "
                "text only."
            )

    text_payload = orjson.dumps(asdict(message)).decode()

    vision_input = None
    if media["has_image"]:
        vision_input = await _build_vision_input(
            media["image_id"], message.chat_jid, text_payload
        )
        if vision_input is None:
            notices.append(
                "I couldn't read that image, so I answered from the text only."
            )

    if vision_input:
        return vision_input, _cached_vision_model
    return text_payload, _cached_model


async def _run_agent(message: ReceivedMessage, runner_input, model, instructions, tools):
    """Run the agent for one turn and return its result."""
    await mcp_pool.ensure_started()
    is_allowed = message.phone_number in ALLOWED_SENDERS

    # Serialize runs within a chat: the cached agent and the session are
    # shared, so concurrent runs would swap each other's tools and interleave
    # conversation history.
    async with agent_factory.lock_for(message.chat_jid):
        agent, session = await agent_factory.get_agent(
            chat_jid=message.chat_jid,
            mcp_servers=mcp_pool.servers(is_privileged=is_allowed),
            model=model,
            instructions=instructions,
            tools=tools,
        )

        # Multimodal list input needs a callback to merge with history.
        run_config = None
        if not isinstance(runner_input, str):
            run_config = RunConfig(
                session_input_callback=lambda history, new: history + new,
            )

        # Gate write-capable tools on an explicit confirmation turn (a no-op
        # unless REQUIRE_WRITE_CONFIRMATION is set).
        write_guard.current_chat.set(message.chat_jid)
        write_guard.begin_turn(message.chat_jid, message.content)
        try:
            with trace("LeoWhatsappAssistant", disabled=not TRACING_ENABLED):
                result = await Runner.run(
                    agent, runner_input, session=session, run_config=run_config
                )
        finally:
            write_guard.end_turn(message.chat_jid)

        await trim_session(session, message.chat_jid)

    return result


async def _deliver(
    message: ReceivedMessage, result, notices: list[str], spoken: bool
) -> None:
    """Send the model's answer back, as voice or split text."""
    logger.info(f"Agent execution completed. Result: {result.final_output}")

    reply_text = (result.final_output or "").strip()
    if not reply_text:
        logger.warning(f"Empty model output for {message.chat_jid}")
        reply_text = "🤔 I came up empty on that one — mind rephrasing?"
    if notices:
        reply_text += "\n\n" + "\n".join(f"_⚠️ {n}_" for n in notices)

    # Answer a voice note with a voice note when TTS is configured; always fall
    # back to text.
    if spoken and await send_voice_reply(
        whatsapp_send_audio, message.chat_jid, reply_text
    ):
        return

    if await send_reply(
        whatsapp_send_message, message.chat_jid, format_leo_response(reply_text)
    ):
        logger.info(f"Message sent successfully to {message.chat_jid}")
    else:
        logger.error(f"Failed to send message to {message.chat_jid}")


async def _respond(message: ReceivedMessage, quoted_media_type: str) -> None:
    """The full AI turn: media → input → run → reply."""
    logger.info(f"Leo mentioned by {message.sender}! Processing...")
    is_allowed = message.phone_number in ALLOWED_SENDERS
    media = _detect_media(message, quoted_media_type)

    # Problems the user should hear about, appended to the reply.
    notices: list[str] = []

    try:
        instructions = _build_instructions(message, is_allowed)
        tools = _build_tools(message, is_allowed)

        # Merge rapid consecutive messages into one turn. Skipped for media:
        # an image or voice note is a turn on its own.
        if not (media["has_image"] or media["has_audio"]):
            burst = await debouncer.collect(message.chat_jid, message)
            if burst is None:
                logger.debug(f"Superseded by a later message in {message.chat_jid}")
                return
            if len(burst) > 1:
                message.content = merge_content(burst)

        async with typing(message.chat_jid):
            runner_input, model = await _build_input(message, media, notices)
            result = await _run_agent(
                message, runner_input, model, instructions, tools
            )
            await _deliver(message, result, notices, spoken=media["has_audio"])

    except Exception as e:
        logger.error(f"Error processing message: {e}", exc_info=True)
        try:
            await _reply(
                message,
                format_leo_response(
                    "❌ Sorry, something went wrong handling that. "
                    "The error has been logged — try again in a moment."
                ),
            )
        except Exception as reply_error:
            logger.error(f"Could not deliver error reply: {reply_error}")


async def process_message(data: dict):
    """Route one inbound WhatsApp message."""
    if logger.isEnabledFor(10):  # logging.DEBUG
        logger.debug("Full message payload: %s", orjson.dumps(data).decode())
    message = ReceivedMessage.from_dict(data)

    # Leo's own replies come back through the bridge; answering them would
    # loop. Messages the user types on a shared linked device are also
    # is_from_me and must still be processed — see is_own_output.
    if is_own_output(message):
        logger.debug(f"Ignoring own output {message.id} in {message.chat_jid}")
        return

    if await _handle_hook_commands(message):
        return

    quoted_media_type = await _enrich_quoted(message)

    snooze_delay = parse_snooze(message.content)
    if snooze_delay and message.phone_number in ALLOWED_SENDERS:
        if await _handle_snooze(message, snooze_delay):
            return

    if await _handle_commands(message):
        return

    # An active hook session swallows everything else in the chat.
    active_hook = get_hook_session(message.chat_jid)
    if active_hook:
        await write_to_hook(active_hook, message.content)
        return

    if not _should_respond(message):
        return

    await _respond(message, quoted_media_type)
