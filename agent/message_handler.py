"""Core message processing for WhatsApp Leo agent."""

import os
import asyncio
from contextlib import AsyncExitStack
from dataclasses import asdict
from datetime import datetime

import orjson
from agents import Runner, trace
from agents.mcp import MCPServerStdio
from whatsapp import send_message as whatsapp_send_message

from config import (
    IS_DEDICATED_NUMBER,
    ALLOWED_SENDERS,
    LEO_MENTION_ID,
    WORKSPACE_MCP_PATH,
    _cached_model,
    _brave_mcp_params,
    _workspace_mcp_params,
    _garmin_mcp_params,
)
from instructions import INSTRUCTIONS_PRIVILEGED_TEMPLATE, INSTRUCTIONS_BASIC_TEMPLATE
from models import ReceivedMessage
from agent_factory import agent_factory, parse_remindme_with_agent, TZ
from command_handlers import handle_briefing_command, handle_reminder_command
from reminder import validate_reminder_time, store_reminder
from logging_setup import logger


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


async def process_message(data: dict):
    """Process a single message asynchronously."""
    if logger.isEnabledFor(10):  # logging.DEBUG
        logger.debug("Full message payload: %s", orjson.dumps(data).decode())
    message = ReceivedMessage.from_dict(data)

    is_leo_mentioned = (
        "#leo" in message.content.lower() or "@leo" in message.content.lower()
    )

    # ── Handle #remindme ─────────────────────────
    if (
        IS_DEDICATED_NUMBER
        and ("#remindme" in message.content.lower())
        and (message.phone_number in ALLOWED_SENDERS)
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
        IS_DEDICATED_NUMBER
        and "#reminder" in message.content.lower()
        and message.phone_number in ALLOWED_SENDERS
    ):
        await handle_reminder_command(message)
        return

    if "#reminder" in message.content.lower():
        return

    # ── Handle #briefing commands ─────────────────────────
    if (
        IS_DEDICATED_NUMBER
        and "#briefing" in message.content.lower()
        and message.phone_number in ALLOWED_SENDERS
    ):
        await handle_briefing_command(message)
        return

    if "#briefing" in message.content.lower():
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

            async with AsyncExitStack() as stack:
                # Start Brave MCP server (WhatsApp is handled via direct function calls)
                brave_mcp_server = await stack.enter_async_context(
                    MCPServerStdio(
                        params=_brave_mcp_params, client_session_timeout_seconds=30
                    )
                )

                mcp_servers = [brave_mcp_server]

                # Conditionally start privileged MCPs
                if is_allowed:
                    if os.path.exists(WORKSPACE_MCP_PATH):
                        workspace_mcp_server = await stack.enter_async_context(
                            MCPServerStdio(
                                params=_workspace_mcp_params,
                                client_session_timeout_seconds=300,
                            )
                        )
                        mcp_servers.append(workspace_mcp_server)
                    else:
                        logger.warning(f"Workspace MCP not found at {WORKSPACE_MCP_PATH}")

                    garmin_mcp_server = await stack.enter_async_context(
                        MCPServerStdio(
                            params=_garmin_mcp_params,
                            client_session_timeout_seconds=120,
                        )
                    )
                    mcp_servers.append(garmin_mcp_server)

                agent, session = await agent_factory.get_agent(
                    chat_jid=message.chat_jid,
                    mcp_servers=mcp_servers,
                    model=_cached_model,
                    instructions=instructions,
                )

                with trace("LeoWhatsappAssistant"):
                    result = await Runner.run(
                        agent, orjson.dumps(asdict(message)).decode(), session=session
                    )

                logger.info(f"Agent execution completed. Result: {result.final_output}")

                # Send the agent's response directly via WhatsApp
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
