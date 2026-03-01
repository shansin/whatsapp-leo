# 🦁 WhatsApp Leo

A personal AI assistant on WhatsApp, powered by local LLMs via [Ollama](https://ollama.com). Leo lives on a dedicated WhatsApp number and can answer questions, search the web, manage your Google Workspace, pull Garmin fitness data, set reminders, run scheduled briefings, and bridge messages to external programs through named-pipe hooks.

## Architecture

```
┌──────────────┐  Unix Socket   ┌──────────────────┐   MCP (stdio)   ┌───────────────────┐
│  Go WhatsApp │◄──────────────►│  Python Agent    │◄───────────────►│  MCP Servers       │
│  Bridge      │                │  Server          │                 │  • Brave Search    │
│  (whatsmeow) │                │  (OpenAI Agents) │                 │  • Google Workspace│
└──────────────┘                └──────────────────┘                 │  • Garmin Connect  │
       │                               │                            └───────────────────┘
       │                               │
  WhatsApp Web API              ┌──────┴──────┐
                                │  SQLite DBs │
                                │  • messages │
                                │  • reminders│
                                │  • briefings│
                                └─────────────┘
```

**Go WhatsApp Bridge** (`whatsapp-mcp/whatsapp-bridge/`) — Connects to WhatsApp's web multidevice API via [whatsmeow](https://github.com/tulir/whatsmeow), authenticates with a QR code, and persists message history in SQLite.

**Python Agent Server** (`agent/`) — Receives messages over a Unix domain socket, runs them through an OpenAI-Agents-SDK pipeline backed by a local Ollama model, and replies via the bridge. Agents are cached per chat with LRU eviction and a configurable TTL.

Communication between the two processes uses **Unix domain sockets** (paths configurable via `INSTANCE_GUID`), allowing multiple instances on the same machine.

## Capabilities

### 💬 Conversational AI
- General knowledge Q&A powered by your chosen Ollama model
- WhatsApp-native formatting (bold, italic, lists, quotes)
- Per-chat conversation memory via `SQLiteSession`

### 🔍 Web Search
- Real-time web search via **Brave Search** MCP server

### 📎 Google Workspace *(privileged users)*
- **Google Docs** — create, read, update, move
- **Google Drive** — find/create folders, search files
- **Google Calendar** — list events, create/update/delete, find free time
- **Google Sheets** — read content, get ranges, metadata
- **Google Slides** — read text, find presentations
- **Gmail** — search threads, draft/send emails, manage labels

### 🏃 Garmin Connect *(privileged users)*
- Access fitness and health data from Garmin devices (sleep, activities, etc.)

### ⏰ Reminders
- **One-shot** — `#remindme in 30 minutes call dentist` — parsed by a dedicated AI agent into a precise datetime, stored in SQLite, and fired by a background scheduler
- **Recurring** — `#reminder add "9pm everyday" brush teeth` — cron-based schedule with add / list / remove / remove-all

### 📋 Scheduled Briefings
- Automated AI-driven briefings that execute prompts on a cron schedule
- Example: `#briefing add "Morning Brief" "9am everyday" Get my sleep data from Garmin and today's calendar events`
- Subcommands: `add`, `list`, `remove`, `remove-all`, `help`
- Briefings run through the full AI pipeline with access to all MCP tools
- Retry logic (up to 3 attempts) for transient LLM errors

### 🪝 Hooks (bidirectional named pipes)
- Route WhatsApp messages to external programs and receive responses back
- Each hook creates two FIFOs: `{name}-in.fifo` (WhatsApp → program) and `{name}-out.fifo` (program → WhatsApp)
- Trigger with `#hook-name message` or `@hook-name message`
- Example hooks: `claude`, `codex`

### 🧪 Test Mode
- Local **Gradio** chat UI at `http://127.0.0.1:7860` that bypasses the WhatsApp bridge
- Model selector to hot-swap Ollama models at runtime
- Live system log panel
- All background schedulers (reminders, briefings) still run

### 🔒 Access Control
- `ALLOWED_SENDERS` whitelist — only listed phone numbers get privileged features (Google Workspace, Garmin, reminders, briefings)
- Non-privileged users can still chat and use web search
- Dedicated-number mode (`IS_DEDICATED_NUMBER=true`) responds to all DMs; in shared-number mode Leo only responds when mentioned (`@leo` / `#leo`)

## Prerequisites

| Dependency | Purpose |
|---|---|
| **Python ≥ 3.13** | Agent server |
| **[uv](https://docs.astral.sh/uv/)** | Python package manager |
| **Go** | WhatsApp bridge |
| **[Ollama](https://ollama.com)** | Local LLM inference |
| **Node.js / npm** | Brave Search MCP, Workspace MCP |

## Setup

### 1. Clone & install Python dependencies

```bash
git clone <repo-url>
cd whatsapp-leo-dedicated-number
uv sync
```

### 2. Configure environment

Copy the example and fill in your values:

```bash
cp .env_example .env
```

Key variables:

| Variable | Description | Default |
|---|---|---|
| `INSTANCE_GUID` | Unique ID for this instance (allows multiple instances) | `default` |
| `OLLAMA_BASE_URL` | Ollama API endpoint | `http://localhost:11434/v1` |
| `MODEL_NAME` | Ollama model to use (e.g. `qwen3.5:35b`) | — |
| `MAX_AGENTS` | Max cached agent instances (LRU eviction) | `20` |
| `TTL_SECONDS` | Agent cache TTL | `1800` |
| `ALLOWED_SENDERS` | Comma-separated phone numbers for privileged access | — |
| `LEO_MENTION_ID` | WhatsApp mention ID for group triggers | — |
| `IS_DEDICATED_NUMBER` | `true` if Leo has its own phone number | `false` |
| `IS_TEST_MODE` | `true` to launch Gradio UI instead of WhatsApp bridge | `false` |
| `OPENAI_API_KEY` | Required by the OpenAI Agents SDK (can be a dummy value for Ollama) | — |
| `BRAVE_API_KEY` | API key for Brave Search | — |
| `REMINDER_POLL_INTERVAL` | Seconds between reminder scheduler polls | `60` |
| `IS_HOOK_ENABLED` | Enable the hooks system | `false` |
| `HOOKS` | Comma-separated hook names (e.g. `claude,codex`) | — |
| `WORKSPACE_MCP_PATH` | Path to the Workspace MCP server `index.js` | — |

### 3. Pull an Ollama model

```bash
ollama pull qwen3.5:35b
```

### 4. Start services

```bash
./start_services.sh
```

This script:
1. Sources `.env`
2. Starts the Python agent server (`uv run python agent/agent.py`)
3. Builds and starts the Go WhatsApp bridge (skipped if `IS_TEST_MODE=true`)
4. Prints Unix socket paths and hook FIFO paths (if enabled)
5. Handles graceful shutdown on `Ctrl+C`

On first run the Go bridge will display a **QR code** — scan it with your WhatsApp mobile app to authenticate.

### 5. Test mode (optional)

```bash
IS_TEST_MODE=true ./start_services.sh
```

Open `http://127.0.0.1:7860` for the Gradio chat UI.

## Project Structure

```
.
├── agent/                    # Python agent server
│   ├── agent.py              # Entry point — dispatches to server or test UI
│   ├── server.py             # Unix domain socket server, starts schedulers
│   ├── config.py             # Environment config, model/MCP singletons
│   ├── agent_factory.py      # LRU-cached Agent instances + reminder parser
│   ├── message_handler.py    # Core message routing (hooks, commands, AI)
│   ├── command_handlers.py   # #briefing and #reminder command handlers
│   ├── briefing.py           # Briefing persistence, scheduling, cron parsing
│   ├── briefing_executor.py  # Executes briefing prompts through AI pipeline
│   ├── reminder.py           # One-shot & recurring reminder persistence/scheduler
│   ├── hooks.py              # FIFO-based bidirectional hook system
│   ├── instructions.py       # Loads and templates system prompts
│   ├── instructions.txt      # System prompt sections
│   ├── models.py             # Data models (ReceivedMessage, ReminderParsed)
│   ├── logging_setup.py      # Logging config + deque for test UI log stream
│   └── test_ui.py            # Gradio test mode UI
├── whatsapp-mcp/             # Forked WhatsApp MCP project
│   ├── whatsapp-bridge/      # Go bridge (whatsmeow + SQLite)
│   └── whatsapp-mcp-server/  # Python MCP server for WhatsApp tools
├── store/                    # SQLite databases (gitignored)
│   ├── messages.db
│   ├── reminders.db
│   └── briefings.db
├── start_services.sh         # Service orchestration script
├── pyproject.toml            # Python project config (uv)
├── .env_example              # Environment variable template
└── .python-version           # Python 3.13
```

## WhatsApp Commands

| Command | Description |
|---|---|
| `#remindme <time> <message>` | Set a one-time reminder |
| `#reminder add "schedule" message` | Create a recurring reminder |
| `#reminder list` | List all recurring reminders |
| `#reminder remove <id>` | Remove a recurring reminder |
| `#reminder remove-all` | Remove all recurring reminders |
| `#briefing add "Name" "Schedule" Prompt` | Create a scheduled briefing |
| `#briefing list` | List all briefings |
| `#briefing remove <id>` | Remove a briefing |
| `#briefing remove-all` | Remove all briefings |
| `#hook-name <message>` | Send message to a named hook |
