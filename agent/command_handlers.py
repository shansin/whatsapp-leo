"""Command handlers for #briefing and #reminder WhatsApp commands."""

import re
from datetime import datetime

from models import ReceivedMessage
from logging_setup import logger
# config must come first: it puts whatsapp-mcp-server on sys.path.
import config  # noqa: F401
from reply import reply_to as _reply
from timeutil import from_db
import user_prefs
from reminder import (
    store_recurring_reminder,
    get_all_recurring_reminders,
    delete_recurring_reminder,
    delete_all_recurring_reminders,
    get_pending_reminders,
    cancel_reminder,
)
from briefing import (
    add_briefing,
    list_briefings,
    remove_briefing,
    remove_all_briefings,
    parse_schedule_to_cron,
    get_next_run_from_cron,
    toggle_briefing,
)


def _user_tz(message: ReceivedMessage):
    """The requesting user's timezone — what times should be shown in."""
    return user_prefs.get_tz(message.phone_number)


def _fmt(dt_or_iso, tz) -> str:
    """Format a stored (UTC) timestamp or datetime in the user's timezone."""
    try:
        dt = from_db(dt_or_iso) if isinstance(dt_or_iso, str) else dt_or_iso
        return dt.astimezone(tz).strftime("%b %d, %I:%M %p %Z")
    except (ValueError, TypeError, OverflowError, AttributeError):
        return str(dt_or_iso)


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
        elif subcommand == "run":
            await handle_briefing_run(message, parts)
        elif subcommand in ("pause", "resume"):
            await handle_briefing_toggle(message, parts, enable=subcommand == "resume")
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
        tz = _user_tz(message)
        briefing_id, cron_expr = add_briefing(
            name, prompt, schedule, message.chat_jid, tz=tz
        )
        next_run = get_next_run_from_cron(cron_expr, datetime.now(tz))
        next_run_str = _fmt(next_run, tz)

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

    tz = _user_tz(message)
    lines = ["📋 *Configured Briefings:*\n"]
    for b in briefings:
        status = "✅" if b["enabled"] else "⏸️"
        next_run = b["next_run_at"]
        # Stored as UTC; show it in the user's local time.
        next_str = _fmt(next_run, tz) if next_run else "Not scheduled"
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
        tz = _user_tz(message)
        cron_expr = parse_schedule_to_cron(schedule)
        next_run = get_next_run_from_cron(cron_expr, datetime.now(tz))
        reminder_id = store_recurring_reminder(
            reminder_message, cron_expr, message.chat_jid, next_run, tz_name=str(tz)
        )
        next_run_str = _fmt(next_run, tz)

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

    tz = _user_tz(message)
    lines = ["⏰ *Recurring Reminders:*\n"]
    for row in rows:
        rid, msg, cron, chat_jid, enabled, created_at, last_run_at, next_run_at = row[:8]
        status = "✅" if enabled else "⏸️"
        # Stored as UTC; show it in the user's local time.
        next_str = _fmt(next_run_at, tz) if next_run_at else "Not scheduled"
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


# ── #briefing run / pause / resume (item 14) ────────────────────────────────


async def handle_briefing_run(message: ReceivedMessage, parts: list):
    """Handle #briefing run <id> — execute a briefing immediately.

    Lets you test a briefing without waiting for its schedule.
    """
    if len(parts) < 3:
        await _reply(message, "❌ Usage: #briefing run <id>")
        return

    try:
        briefing_id = int(parts[2].split()[0])
    except (ValueError, IndexError):
        await _reply(message, "❌ Please provide a valid briefing ID number.")
        return

    briefing = next((b for b in list_briefings() if b["id"] == briefing_id), None)
    if briefing is None:
        await _reply(message, f"❌ Briefing {briefing_id} not found.")
        return

    await _reply(message, f"⏳ Running *{briefing['name']}* now…")

    # Imported here: briefing_executor pulls in the whole agent pipeline.
    from briefing_executor import execute_briefing_prompt

    try:
        result = await execute_briefing_prompt(
            briefing["prompt"], message.chat_jid, briefing["name"]
        )
    except Exception as e:
        logger.error(f"Manual briefing run failed: {e}", exc_info=True)
        await _reply(message, f"❌ Briefing failed: {e}")
        return

    await _reply(message, f"📋 *{briefing['name']}*\n\n{result}")


async def handle_briefing_toggle(message: ReceivedMessage, parts: list, enable: bool):
    """Handle #briefing pause|resume <id>."""
    verb = "resume" if enable else "pause"
    if len(parts) < 3:
        await _reply(message, f"❌ Usage: #briefing {verb} <id>")
        return

    try:
        briefing_id = int(parts[2].split()[0])
    except (ValueError, IndexError):
        await _reply(message, "❌ Please provide a valid briefing ID number.")
        return

    if toggle_briefing(briefing_id, enable):
        state = "resumed ✅" if enable else "paused ⏸️"
        await _reply(message, f"Briefing {briefing_id} {state}")
        logger.info(f"Briefing {briefing_id} {verb}d")
    else:
        await _reply(message, f"❌ Briefing {briefing_id} not found.")


# ── #remindme list / cancel (item 14) ───────────────────────────────────────


async def handle_remindme_list(message: ReceivedMessage):
    """Handle #remindme list — show pending one-shot reminders for this chat."""
    rows = get_pending_reminders(message.chat_jid)
    if not rows:
        await _reply(
            message,
            "⏰ No pending reminders.\n\nSet one with: #remindme in 30 minutes call the dentist",
        )
        return

    tz = _user_tz(message)
    lines = ["⏰ *Pending Reminders:*\n"]
    for rid, text, remind_at in rows:
        lines.append(f"*ID {rid}:* {text}")
        lines.append(f"   {_fmt(remind_at, tz)}\n")
    lines.append("_Cancel one with: #remindme cancel <id>_")
    await _reply(message, "\n".join(lines))


async def handle_remindme_cancel(message: ReceivedMessage, parts: list):
    """Handle #remindme cancel <id>."""
    if len(parts) < 3:
        await _reply(message, "❌ Usage: #remindme cancel <id>")
        return

    try:
        reminder_id = int(parts[2].split()[0])
    except (ValueError, IndexError):
        await _reply(message, "❌ Please provide a valid reminder ID number.")
        return

    if cancel_reminder(reminder_id, message.chat_jid):
        await _reply(message, f"✅ Reminder {reminder_id} cancelled.")
        logger.info(f"Reminder {reminder_id} cancelled in {message.chat_jid}")
    else:
        await _reply(message, f"❌ No pending reminder {reminder_id} in this chat.")


# ── #tz (item 15) ───────────────────────────────────────────────────────────


async def handle_tz_command(message: ReceivedMessage):
    """Handle #tz [Area/City] — show or set the sender's timezone."""
    parts = message.content.strip().split(maxsplit=1)

    if len(parts) < 2:
        tz = _user_tz(message)
        now = datetime.now(tz).strftime("%I:%M %p on %A, %b %d")
        await _reply(
            message,
            f"🌍 Your timezone is *{tz}* — it's {now} for you.\n\n"
            "_Change it with: #tz Europe/London_",
        )
        return

    try:
        tz = user_prefs.set_tz(message.phone_number, parts[1].strip())
    except ValueError as e:
        await _reply(message, f"❌ {e}")
        return

    now = datetime.now(tz).strftime("%I:%M %p on %A, %b %d")
    await _reply(message, f"🌍 Timezone set to *{tz}* — it's {now} for you.")
    logger.info(f"Timezone for {message.phone_number} set to {tz}")


# ── #help (item 14) ─────────────────────────────────────────────────────────


HELP_TEXT = """🦁 *Leo — Commands*

*⏰ One-off reminders*
• `#remindme in 30 minutes call the dentist`
• `#remindme list` — pending reminders
• `#remindme cancel <id>`
• Reply *snooze 10m* to a fired reminder to push it back

*🔁 Recurring reminders*
• `#reminder add "9pm everyday" brush teeth`
• `#reminder list` / `#reminder remove <id>` / `#reminder remove-all`
• `#reminder help`

*📋 Briefings* — scheduled AI reports
• `#briefing add "Morning Brief" "9am everyday" My sleep and calendar`
• `#briefing list` / `#briefing remove <id>` / `#briefing remove-all`
• `#briefing run <id>` — run it now
• `#briefing pause <id>` / `#briefing resume <id>`
• `#briefing help`

*🌍 Timezone*
• `#tz` — show yours · `#tz Europe/London` — set it

*🩺 Diagnostics*
• `#status` — model in use, MCP servers, uptime, recent errors

*🧠 Memory*
• Just ask: _"remember that I'm allergic to peanuts"_, _"what do you remember?"_, _"forget everything"_

*💬 Everything else*
Just talk to me — I can search the web, read your calendar and mail,
pull Garmin data, look at images, and listen to voice notes.
"""


async def handle_help_command(message: ReceivedMessage, hook_names=()):
    """Handle #help — list every command."""
    text = HELP_TEXT
    if hook_names:
        hooks = "\n".join(f"• `#{name} <message>`" for name in hook_names)
        text += f"\n*🪝 Hooks*\n{hooks}\n• `#<hook> #start` / `#<hook> #stop` for a session\n"
    await _reply(message, text)


# ── #status (item 19) ───────────────────────────────────────────────────────


def _format_uptime(seconds: float) -> str:
    seconds = int(seconds)
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes = seconds // 60
    if days:
        return f"{days}d {hours}h {minutes}m"
    if hours:
        return f"{hours}h {minutes}m"
    return f"{minutes}m"


async def handle_status_command(message: ReceivedMessage):
    """Handle #status — model, MCP servers, caches, uptime, recent errors."""
    import time

    from agent_factory import agent_factory
    from config import (
        MODEL_NAME,
        OLLAMA_BACKUP_MODEL_NAME,
        VISION_MODEL_NAME,
        _fallback_router,
    )
    from logging_setup import STARTED_AT, error_deque
    from mcp_pool import mcp_pool
    from session_store import _sessions

    tz = _user_tz(message)
    lines = ["🦁 *Leo Status*\n"]

    # Which model is actually answering right now.
    on_backup = await _fallback_router.should_use_backup()
    if on_backup:
        lines.append(f"*Model:* ⚠️ backup — {OLLAMA_BACKUP_MODEL_NAME}")
    else:
        lines.append(f"*Model:* {MODEL_NAME}")
    lines.append(f"*Vision:* {VISION_MODEL_NAME}")

    servers = mcp_pool._servers
    if servers:
        up = [name for name, srv in servers.items() if srv.healthy]
        down = [name for name in servers if name not in up]
        lines.append(f"*MCP:* {len(up)}/{len(servers)} up — {', '.join(up) or 'none'}")
        if down:
            lines.append(f"   ⚠️ down: {', '.join(down)}")
    else:
        lines.append("*MCP:* not started yet")

    lines.append(f"*Agents cached:* {len(agent_factory._agents)}")
    lines.append(f"*Chats with history:* {len(_sessions)}")
    lines.append(f"*Uptime:* {_format_uptime(time.time() - STARTED_AT)}")
    lines.append(f"*Your timezone:* {tz}")

    if error_deque:
        lines.append("\n*Recent problems:*")
        for created, level, name, text in list(error_deque)[-5:]:
            when = datetime.fromtimestamp(created, tz).strftime("%H:%M")
            lines.append(f"• `{when}` [{level}] {name}: {text[:120]}")
    else:
        lines.append("\n_No warnings or errors logged._")

    await _reply(message, "\n".join(lines))
