"""Command handlers for #briefing and #reminder WhatsApp commands."""

import asyncio
import re
from datetime import datetime

from whatsapp import send_message as whatsapp_send_message
from models import ReceivedMessage
from logging_setup import logger
from reminder import (
    store_recurring_reminder,
    get_all_recurring_reminders,
    delete_recurring_reminder,
    delete_all_recurring_reminders,
)
from briefing import (
    add_briefing,
    list_briefings,
    remove_briefing,
    remove_all_briefings,
    parse_schedule_to_cron,
    get_next_run_from_cron,
)


async def _reply(message: ReceivedMessage, text: str) -> None:
    """Send a WhatsApp reply to the originating message (non-blocking)."""
    await asyncio.to_thread(
        whatsapp_send_message,
        message.chat_jid,
        text,
        reply_to=message.id,
        reply_to_sender=message.sender_jid,
    )


# ── #briefing command handlers ──────────────────────────────────────────────


async def handle_briefing_command(message: ReceivedMessage):
    """Handle #briefing commands for managing briefings."""
    content = message.content.strip()
    parts = content.split(maxsplit=2)

    if len(parts) < 2:
        await send_briefing_help(message)
        return

    subcommand = parts[1].lower()

    try:
        if subcommand == "add":
            await handle_briefing_add(message, parts)
        elif subcommand == "list":
            await handle_briefing_list(message)
        elif subcommand == "remove":
            await handle_briefing_remove(message, parts)
        elif subcommand == "remove-all":
            await handle_briefing_remove_all(message)
        elif subcommand == "help":
            await send_briefing_help(message)
        else:
            await _reply(message, f"❌ Unknown briefing command: {subcommand}\n\nUse #briefing help for usage.")
    except Exception as e:
        logger.error(f"Error handling briefing command: {e}", exc_info=True)
        await _reply(message, f"❌ Error: {str(e)}")


async def handle_briefing_add(message: ReceivedMessage, parts: list):
    """Handle #briefing add command."""
    content = message.content.strip()

    # Extract quoted strings
    quoted = re.findall(r'"([^"]*)"', content)

    if len(quoted) < 2:
        await _reply(
            message,
            '❌ Usage: #briefing add "Name" "Schedule" Prompt text...\n\nExample:\n#briefing add "Morning Brief" "9am everyday" Get my sleep data from Garmin and calendar events for today',
        )
        return

    name = quoted[0]
    schedule = quoted[1]

    # Get prompt (everything after the second quoted string)
    prompt_start = content.find(f'"{schedule}"') + len(f'"{schedule}"')
    prompt = content[prompt_start:].strip()

    if not prompt:
        await _reply(message, "❌ Please provide a prompt for the briefing.")
        return

    try:
        briefing_id, cron_expr = add_briefing(name, prompt, schedule, message.chat_jid)
        next_run = get_next_run_from_cron(cron_expr)
        next_run_str = next_run.strftime("%b %d, %I:%M %p %Z")

        await _reply(
            message,
            f"📋 Briefing created!\n\n*Name:* {name}\n*Schedule:* {schedule}\n*Cron:* {cron_expr}\n*Next run:* {next_run_str}\n*ID:* {briefing_id}",
        )
        logger.info(f"Briefing '{name}' created with ID {briefing_id}")
    except ValueError as e:
        await _reply(message, f"❌ {e}")


async def handle_briefing_list(message: ReceivedMessage):
    """Handle #briefing list command."""
    briefings = list_briefings()

    if not briefings:
        await _reply(message, "📋 No briefings configured.\n\nUse #briefing add to create one.")
        return

    lines = ["📋 *Configured Briefings:*\n"]
    for b in briefings:
        status = "✅" if b["enabled"] else "⏸️"
        next_run = b["next_run_at"]
        if next_run:
            try:
                next_dt = datetime.fromisoformat(next_run)
                next_str = next_dt.strftime("%b %d, %I:%M %p")
            except (ValueError, TypeError):
                next_str = next_run
        else:
            next_str = "Not scheduled"
        lines.append(f"{status} *ID {b['id']}:* {b['name']}")
        lines.append(f"   Schedule: {b['schedule_cron']}")
        lines.append(f"   Next: {next_str}\n")

    await _reply(message, "\n".join(lines))


async def handle_briefing_remove(message: ReceivedMessage, parts: list):
    """Handle #briefing remove command."""
    if len(parts) < 3:
        await _reply(message, "❌ Usage: #briefing remove <id>")
        return

    try:
        briefing_id = int(parts[2])
        if remove_briefing(briefing_id):
            await _reply(message, f"✅ Briefing {briefing_id} removed.")
            logger.info(f"Briefing {briefing_id} removed")
        else:
            await _reply(message, f"❌ Briefing {briefing_id} not found.")
    except ValueError:
        await _reply(message, "❌ Please provide a valid briefing ID number.")


async def handle_briefing_remove_all(message: ReceivedMessage):
    """Handle #briefing remove-all command."""
    count = remove_all_briefings()
    await _reply(message, f"✅ Removed all briefings ({count} deleted).")
    logger.info(f"All briefings removed ({count} deleted)")


async def send_briefing_help(message: ReceivedMessage):
    """Send briefing help message."""
    help_text = """📋 *Briefing Commands*

Create automated AI briefings that run on a schedule.

*Commands:*
• #briefing add "Name" "Schedule" Prompt
  _Create a new briefing_
  Example: #briefing add "Morning Brief" "9am everyday" Get my sleep and calendar

• #briefing list
  _Show all briefings_

• #briefing remove <id>
  _Remove a briefing by ID_

• #briefing remove-all
  _Remove all briefings_

• #briefing help
  _Show this help_

*Schedule formats:*
• "9am everyday" - Daily at 9 AM
• "8am monday" - Mondays at 8 AM
• "5pm friday" - Fridays at 5 PM
• "every morning" - Daily at 9 AM
"""
    await _reply(message, help_text)


# ── #reminder command handlers ──────────────────────────────────────────────


async def handle_reminder_command(message: ReceivedMessage):
    """Handle #reminder commands for managing recurring reminders."""
    content = message.content.strip()
    parts = content.split(maxsplit=2)

    if len(parts) < 2:
        await send_reminder_help(message)
        return

    subcommand = parts[1].lower()

    try:
        if subcommand == "add":
            await handle_reminder_add(message)
        elif subcommand == "list":
            await handle_reminder_list(message)
        elif subcommand == "remove":
            await handle_reminder_remove(message, parts)
        elif subcommand == "remove-all":
            await handle_reminder_remove_all(message)
        elif subcommand == "help":
            await send_reminder_help(message)
        else:
            await _reply(
                message,
                f"❌ Unknown reminder command: {subcommand}\n\nUse #reminder help for usage.",
            )
    except Exception as e:
        logger.error(f"Error handling reminder command: {e}", exc_info=True)
        await _reply(message, f"❌ Error: {str(e)}")


async def handle_reminder_add(message: ReceivedMessage):
    """Handle #reminder add command.

    Format: #reminder add "schedule" reminder message text
    Example: #reminder add "9pm everyday" brush teeth
    """
    content = message.content.strip()

    # Extract the quoted schedule string
    quoted = re.findall(r'"([^"]*)"', content)

    if not quoted:
        await _reply(
            message,
            '❌ Usage: #reminder add "Schedule" Reminder text...\n\nExample:\n#reminder add "9pm everyday" brush teeth',
        )
        return

    schedule = quoted[0]

    # Get reminder message (everything after the quoted schedule)
    msg_start = content.find(f'"{schedule}"') + len(f'"{schedule}"')
    reminder_message = content[msg_start:].strip()

    if not reminder_message:
        await _reply(message, "❌ Please provide a reminder message.")
        return

    try:
        cron_expr = parse_schedule_to_cron(schedule)
        next_run = get_next_run_from_cron(cron_expr)
        reminder_id = store_recurring_reminder(
            reminder_message, cron_expr, message.chat_jid, next_run
        )
        next_run_str = next_run.strftime("%b %d, %I:%M %p %Z")

        await _reply(
            message,
            f"⏰ Recurring reminder created!\n\n*Reminder:* {reminder_message}\n*Schedule:* {schedule}\n*Cron:* {cron_expr}\n*Next run:* {next_run_str}\n*ID:* {reminder_id}",
        )
        logger.info(f"Recurring reminder '{reminder_message}' created with ID {reminder_id}")
    except ValueError as e:
        await _reply(message, f"❌ {e}")


async def handle_reminder_list(message: ReceivedMessage):
    """Handle #reminder list command."""
    rows = get_all_recurring_reminders()

    if not rows:
        await _reply(
            message,
            "⏰ No recurring reminders configured.\n\nUse #reminder add to create one.",
        )
        return

    lines = ["⏰ *Recurring Reminders:*\n"]
    for row in rows:
        rid, msg, cron, chat_jid, enabled, created_at, last_run_at, next_run_at = row
        status = "✅" if enabled else "⏸️"
        if next_run_at:
            try:
                next_dt = datetime.fromisoformat(next_run_at)
                next_str = next_dt.strftime("%b %d, %I:%M %p")
            except (ValueError, TypeError):
                next_str = next_run_at
        else:
            next_str = "Not scheduled"
        lines.append(f"{status} *ID {rid}:* {msg}")
        lines.append(f"   Schedule: {cron}")
        lines.append(f"   Next: {next_str}\n")

    await _reply(message, "\n".join(lines))


async def handle_reminder_remove(message: ReceivedMessage, parts: list):
    """Handle #reminder remove command."""
    if len(parts) < 3:
        await _reply(message, "❌ Usage: #reminder remove <id>")
        return

    try:
        reminder_id = int(parts[2])
        if delete_recurring_reminder(reminder_id):
            await _reply(message, f"✅ Recurring reminder {reminder_id} removed.")
            logger.info(f"Recurring reminder {reminder_id} removed")
        else:
            await _reply(message, f"❌ Recurring reminder {reminder_id} not found.")
    except ValueError:
        await _reply(message, "❌ Please provide a valid reminder ID number.")


async def handle_reminder_remove_all(message: ReceivedMessage):
    """Handle #reminder remove-all command."""
    count = delete_all_recurring_reminders()
    await _reply(message, f"✅ Removed all recurring reminders ({count} deleted).")
    logger.info(f"All recurring reminders removed ({count} deleted)")


async def send_reminder_help(message: ReceivedMessage):
    """Send recurring reminder help message."""
    help_text = """⏰ *Reminder Commands*

Create recurring reminders that fire on a schedule.

*Commands:*
• #reminder add "Schedule" Reminder text
  _Create a new recurring reminder_
  Example: #reminder add "9pm everyday" brush teeth

• #reminder list
  _Show all recurring reminders_

• #reminder remove <id>
  _Remove a reminder by ID_

• #reminder remove-all
  _Remove all recurring reminders_

• #reminder help
  _Show this help_

*Schedule formats:*
• "9pm everyday" - Daily at 9 PM
• "8am monday" - Mondays at 8 AM
• "5pm friday" - Fridays at 5 PM
• "every morning" - Daily at 9 AM

_For one-time reminders, use #remindme instead._
"""
    await _reply(message, help_text)
