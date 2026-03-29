"""Core message processing for WhatsApp Leo agent."""

import asyncio
import base64
import os
import sqlite3
from dataclasses import asdict
from datetime import datetime
from io import BytesIO

import orjson
from agents import RunConfig, Runner, trace
from PIL import Image
from whatsapp import send_message as whatsapp_send_message
from whatsapp import download_media

from config import (
    IS_DEDICATED_NUMBER,
    ALLOWED_SENDERS,
    LEO_MENTION_ID,
    MAX_IMAGE_DIMENSION,
    _cached_model,
    _cached_vision_model,
    mcp_stack,
)
from instructions import INSTRUCTIONS_PRIVILEGED_TEMPLATE, INSTRUCTIONS_BASIC_TEMPLATE
from models import ReceivedMessage
from agent_factory import agent_factory, parse_remindme_with_agent, TZ
from command_handlers import handle_briefing_command, handle_reminder_command
from reminder import validate_reminder_time, store_reminder
from hooks import match_hook, match_hook_session_command, write_to_hook, start_hook_session, stop_hook_session, get_hook_session
from logging_setup import logger


_MESSAGES_DB = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "store", "messages.db"
)


def _lookup_quoted_message(message_id: str, chat_jid: str) -> tuple[str, str, str]:
    """Look up a quoted message's content, sender, and media_type from the messages DB.

    Returns (content, sender, media_type) or ("", "", "") if not found.
    """
    try:
        conn = sqlite3.connect(_MESSAGES_DB)
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


async def _reply(message: ReceivedMessage, text: str) -> None:
    """Send a WhatsApp reply to the originating message (non-blocking)."""
    await asyncio.to_thread(
        whatsapp_send_message,
        message.chat_jid,
        text,
        reply_to=message.id,
        reply_to_sender=message.sender_jid,
    )


def _load_and_resize_image(file_path: str) -> Image.Image:
    """Open and downscale the image if it exceeds MAX_IMAGE_DIMENSION."""
    img = Image.open(file_path)
    img = img.convert("RGB")
    w, h = img.size
    if max(w, h) > MAX_IMAGE_DIMENSION:
        scale = MAX_IMAGE_DIMENSION / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return img


def _encode_image_b64(img: Image.Image) -> str:
    """Encode PIL image to base64 JPEG string."""
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _infer_mime(file_path: str) -> str:
    """Return MIME type based on file extension, defaulting to image/jpeg."""
    ext = file_path.rsplit(".", 1)[-1].lower() if "." in file_path else ""
    return {"png": "image/png", "gif": "image/gif", "webp": "image/webp"}.get(
        ext, "image/jpeg"
    )


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

    mime_type = _infer_mime(file_path)
    data_uri = f"data:{mime_type};base64,{image_b64}"

    return [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": text_payload},
                {"type": "input_image", "image_url": data_uri},
            ],
        }
    ]


async def process_message(data: dict):
    """Process a single message asynchronously."""
    if logger.isEnabledFor(10):  # logging.DEBUG
        logger.debug("Full message payload: %s", orjson.dumps(data).decode())
    message = ReceivedMessage.from_dict(data)

    # ── Hook session start/stop ──────────────────
    session_cmd = match_hook_session_command(message.content)
    if session_cmd and message.phone_number in ALLOWED_SENDERS:
        hook_name, action = session_cmd
        if action == "start":
            start_hook_session(message.chat_jid, hook_name)
            await _reply(message, f"🔗 Hook session *{hook_name}* started. All messages will be forwarded. Send *#{hook_name} #stop* to end.")
        else:
            stopped = stop_hook_session(message.chat_jid)
            if stopped:
                await _reply(message, f"Hook session *{stopped}* ended. Regular processing resumed.")
            else:
                await _reply(message, "No active hook session to stop.")
        return

    # ── Hook intercept (single-message) ──────────
    hook_match = match_hook(message.content)
    if hook_match and message.phone_number in ALLOWED_SENDERS:
        hook_name, body = hook_match
        await write_to_hook(hook_name, body)
        return

    is_leo_mentioned = (
        "#leo" in message.content.lower() or "@leo" in message.content.lower()
    )

    # ── Enrich quoted message from DB if proto didn't embed content ──
    quoted_media_type = ""
    if message.quoted_message_id:
        db_content, db_sender, quoted_media_type = _lookup_quoted_message(
            message.quoted_message_id, message.chat_jid
        )
        if db_content and not message.quoted_message_content:
            message.quoted_message_content = db_content
        if db_sender and not message.quoted_message_sender:
            message.quoted_message_sender = db_sender

    # ── Handle #remindme ─────────────────────────
    if (
        "#remindme" in message.content.lower()
        and message.phone_number in ALLOWED_SENDERS
    ):
        try:
            remind_at, original_msg = await parse_remindme_with_agent(message.content)
            validate_reminder_time(remind_at)
            store_reminder(
                message.chat_jid,
                original_msg,
                remind_at,
                message_id=message.id,
                sender_jid=message.sender_jid,
            )
            confirm_time = remind_at.strftime("%b %d, %I:%M %p %Z")
            await _reply(message, f"⏰ Reminder set for *{confirm_time}*!")
            logger.info(f"Reminder set for {confirm_time} in {message.chat_jid}")
            return
        except ValueError as e:
            await _reply(message, f"❌ {e}")
            return
        except Exception as e:
            logger.error(f"Error handling #remindme: {e}", exc_info=True)
            await _reply(message, "❌ Something went wrong setting the reminder.")
            return

    if "#remindme" in message.content.lower():
        return

    # ── Handle #reminder commands ─────────────────────────
    if (
        "#reminder" in message.content.lower()
        and message.phone_number in ALLOWED_SENDERS
    ):
        await handle_reminder_command(message)
        return

    if "#reminder" in message.content.lower():
        return

    # ── Handle #briefing commands ─────────────────────────
    if (
        "#briefing" in message.content.lower()
        and message.phone_number in ALLOWED_SENDERS
    ):
        await handle_briefing_command(message)
        return

    if "#briefing" in message.content.lower():
        return

    # ── Active hook session forwarding ───────────
    active_hook = get_hook_session(message.chat_jid)
    if active_hook:
        await write_to_hook(active_hook, message.content)
        return

    should_leo_respond = False

    if IS_DEDICATED_NUMBER:
        # Respond if: DM (ends with @lid) OR group mention (@g in jid AND @23833461416078 in content)
        is_dm = message.chat_jid.endswith("@lid")
        is_group_mention = (
            "@g" in message.chat_jid and LEO_MENTION_ID in message.content
        )
        should_leo_respond = is_dm or is_group_mention
    else:
        should_leo_respond = is_leo_mentioned

    if should_leo_respond:
        logger.info(f"Leo mentioned by {message.sender}! Processing...")

        # Check if sender is allowed to use privileged feature
        is_allowed = message.phone_number in ALLOWED_SENDERS

        # ── Detect image: direct send or reply-to-image ──
        is_direct_image = message.media_type == "image"
        is_quoted_image = (
            not is_direct_image
            and message.quoted_message_id != ""
            and quoted_media_type == "image"
        )
        has_image = is_direct_image or is_quoted_image
        image_message_id = (
            message.id if is_direct_image else message.quoted_message_id
        )

        try:
            now = datetime.now(TZ)
            current_time = now.strftime("%I:%M %p PST, %B %d, %Y")

            # Use pre-built template — only interpolate the timestamp
            template = (
                INSTRUCTIONS_PRIVILEGED_TEMPLATE
                if is_allowed
                else INSTRUCTIONS_BASIC_TEMPLATE
            )
            instructions = template.format(current_time=current_time)

            text_payload = orjson.dumps(asdict(message)).decode()

            # Build multimodal input if image is present
            vision_input = None
            if has_image:
                vision_input = await _build_vision_input(
                    image_message_id, message.chat_jid, text_payload
                )

            # Use vision model if image was successfully processed, else text model
            model = _cached_vision_model if vision_input else _cached_model
            runner_input = vision_input if vision_input else text_payload

            async with mcp_stack(is_privileged=is_allowed) as mcp_servers:
                agent, session = await agent_factory.get_agent(
                    chat_jid=message.chat_jid,
                    mcp_servers=mcp_servers,
                    model=model,
                    instructions=instructions,
                )

                # When passing multimodal list input with session, we need a
                # callback to merge it with conversation history.
                run_config = None
                if vision_input:
                    run_config = RunConfig(
                        session_input_callback=lambda history, new: history + new,
                    )

                with trace("LeoWhatsappAssistant"):
                    result = await Runner.run(
                        agent, runner_input, session=session,
                        run_config=run_config,
                    )

                logger.info(f"Agent execution completed. Result: {result.final_output}")

                if result.final_output:
                    success, send_result = await asyncio.to_thread(
                        whatsapp_send_message,
                        message.chat_jid,
                        format_leo_response(result.final_output),
                    )
                    if success:
                        logger.info(f"Message sent successfully to {message.chat_jid}")
                    else:
                        logger.error(
                            f"Failed to send message to {message.chat_jid}: {send_result}"
                        )

        except Exception as e:
            logger.error(f"Error processing message: {e}", exc_info=True)
