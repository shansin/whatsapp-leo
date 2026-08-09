# 🦁 WhatsApp Leo

A personal AI assistant on WhatsApp, powered by local LLMs via [Ollama](https://ollama.com). Leo lives on a dedicated WhatsApp number and can answer questions, search the web, browse real pages in Chrome, read X/Twitter feeds, manage your Google Workspace, pull Garmin fitness data, set reminders, run scheduled briefings, understand voice notes and images, and bridge messages to external programs through named-pipe hooks.

## Architecture

```
┌──────────────┐  Unix Socket   ┌──────────────────┐   MCP (stdio)   ┌───────────────────────┐
│  Go WhatsApp │◄──────────────►│  Python Agent    │◄───────────────►│  MCP Servers           │
│  Bridge      │                │  Server          │                 │  • Brave Search        │
│  (whatsmeow) │                │  (OpenAI Agents) │                 │  • Playwright (Chrome) │
└──────────────┘                └──────────────────┘                 │  • X (Twitter)         │
       │                               │                             │  • Google Workspace    │
  WhatsApp Web API              ┌──────┴──────┐                      │  • Garmin Connect      │
                                │  SQLite DBs │                      └───────────────────────┘
                                │  • messages │
                                │  • reminders│
                                │  • briefings│
                                └─────────────┘
```

**Go WhatsApp Bridge** (`whatsapp-mcp/whatsapp-bridge/`) — Connects to WhatsApp's web multidevice API via [whatsmeow](https://github.com/tulir/whatsmeow), authenticates with a QR code, and persists message history in SQLite.

**Python Agent Server** (`agent/`) — Receives messages over a Unix domain socket, runs them through an OpenAI-Agents-SDK pipeline backed by a local Ollama model, and replies via the bridge. Agents are cached per chat with LRU eviction and a configurable TTL. MCP servers are started **once** per process and shared by every message (a crashed server is respawned in the background), so no message pays subprocess-spawn and handshake cost.

Communication between the two processes uses **Unix domain sockets** (paths configurable via `INSTANCE_GUID`), allowing multiple instances on the same machine.

## Capabilities

### 💬 Conversational AI
- General knowledge Q&A powered by your chosen Ollama model
- WhatsApp-native formatting (bold, italic, lists, quotes)
- Per-chat conversation memory, persisted to `store/sessions.db` (survives restarts) and trimmed to a rolling window so long chats don't overflow the model's context
- **Quoted message context** — when replying to a message, Leo retrieves the quoted message (text, image, or audio) from the database and includes it as context for the response
- **Live feedback** — a WhatsApp typing indicator runs for the duration of a request (local models can take 30s+), and failures reply with an error instead of going silent

### 🖼️ Vision (Multimodal)
- Send or reply-to an image and Leo will analyze it using a dedicated vision model (configurable via `VISION_MODEL_NAME`, default: `gemma3:27b`)
- Images are automatically downscaled to `MAX_IMAGE_DIMENSION` (default: 1280px) and JPEG-encoded before being sent to the model
- Vision and text agents are cached separately per chat (keyed by model name) so switching between them doesn't break session history

### 🎤 Voice Notes (Audio Transcription)
- Send or reply-to a voice note and Leo will transcribe it using [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (local Whisper inference, auto-detects CPU or CUDA)
- The transcript is injected into the message content and processed through the normal AI pipeline
- Model configurable via `WHISPER_MODEL_SIZE` (default: `distil-medium.en` — several times faster on CPU, but English-only; use `medium`/`large-v3` for other languages)
- Greedy decoding and VAD silence-skipping by default (`WHISPER_BEAM_SIZE`, `WHISPER_VAD_FILTER`); the model is pre-warmed at startup so the first voice note doesn't pay load time

### ⏰ Reminders
- **One-shot** — `#remindme in 30 minutes call dentist` — parsed by a dedicated AI agent into a precise datetime, stored in SQLite, and fired by a background scheduler. `#remindme list` / `#remindme cancel <id>` to manage them
- **Recurring** — `#reminder add "9pm everyday" brush teeth` — cron-based schedule with add / list / remove / remove-all
- **Snooze** — reply *snooze 10m* (or `2h`, `1d`) to a fired reminder to push it back
- Schedules are stored with the timezone they were created in, so "9pm every day" keeps meaning 9pm where you are

### 📋 Scheduled Briefings
- Automated AI-driven briefings that execute prompts on a cron schedule
- Example: `#briefing add "Morning Brief" "9am everyday" Get my sleep data from Garmin and today's calendar events`
- Subcommands: `add`, `list`, `remove`, `remove-all`, `run` (execute now), `pause`, `resume`, `help`
- Briefings run through the full AI pipeline with access to all MCP tools
- Retry logic (up to 3 attempts) for transient LLM errors

### 📤 Richer replies
- Long answers are split at paragraph / sentence boundaries instead of arriving as one wall of text; code fences are closed and reopened across the split
- Leo can send files back into the chat (`send_file_to_chat`), restricted to its own store and an optional `SHARE_DIR` — resolved through symlinks so it can't be talked into sending `.env`
- Optional voice replies to voice notes: set `VOICE_REPLIES=true` and a `TTS_COMMAND` (piper, kokoro, anything that reads stdin and writes `{out}`)

### ⌨️ Message debouncing
- People send a thought as three or four rapid messages; Leo waits `DEBOUNCE_SECONDS` (default 2.5) and answers the whole thought once
- Images and voice notes are never debounced — each is a turn on its own

### 🔎 Chat history
- Leo can search the message archive the bridge already stores: *"what did Sam say about the trip?"*, *"summarise this group today"*
- Tools: `search_chat_history`, `recent_chat_messages`, `messages_from_person`, `find_chats`
- Privileged senders only — this is the full personal message archive

### 🌍 Per-user timezone
- `#tz Europe/London` sets your timezone; `#tz` shows it
- Reminders, briefings and the clock Leo reports all follow the sender's timezone, falling back to `DEFAULT_TZ`

### 🤖 Live model switching
- `#model` lists what Ollama actually has installed, marking the active text (✅) and vision (👁) models
- `#model set <name>` switches the running instance — chats, briefings and reminder parsing all follow — with no restart and no dropped MCP servers
- The choice is persisted, so it survives a restart; `#model reset` returns to `MODEL_NAME` / `VISION_MODEL_NAME`
- Unknown names are refused against the installed list, so a typo can't leave Leo 404ing on every message

### 🪝 Hooks (bidirectional named pipes)
- Route WhatsApp messages to external programs and receive responses back
- Each hook creates two FIFOs: `{name}-in.fifo` (WhatsApp → program) and `{name}-out.fifo` (program → WhatsApp)
- Trigger with `#hook-name message` or `@hook-name message`
- **Session mode** — `#hook-name #start` enters a persistent session where all subsequent messages from that chat are forwarded to the hook (no prefix needed). `#hook-name #stop` ends the session and resumes normal Leo processing
- Example hooks: `claude`, `claude-session`

### 🧪 Test Mode
- Local **Gradio** chat UI at `http://127.0.0.1:7860` that bypasses the WhatsApp bridge
- Model selector to hot-swap Ollama models at runtime
- Live system log panel
- All background schedulers (reminders, briefings) still run

### 🔒 Access Control
- `ALLOWED_SENDERS` whitelist — only listed phone numbers get privileged features (Google Workspace, Garmin, X, Chrome browsing, reminders, briefings)
- Non-privileged users can still chat and use web search
- Dedicated-number mode (`IS_DEDICATED_NUMBER=true`) responds to all DMs; in shared-number mode Leo only responds when mentioned (`@leo` / `#leo`)
- Unix sockets live in `$XDG_RUNTIME_DIR` (falling back to `/tmp`) with mode `0600` — anything that can write to the agent socket could otherwise spoof an allowed sender
- Commands are matched as a **prefix**, so a sentence merely mentioning `#briefing` is treated as normal chat; non-privileged senders get a refusal rather than silence
- `REQUIRE_WRITE_CONFIRMATION=true` refuses mutating tool calls (calendar writes, drafts, and acting browser tools like click/type/evaluate/upload) at the MCP boundary until the user replies with a confirmation — a code-level guard against prompt injection arriving via quoted messages, transcripts, web results, or the text of any page Leo opens. Strongly recommended whenever `PLAYWRIGHT_ENABLED=true`

## MCP Servers & Tools

Leo connects to MCP servers over stdio. Tool exposure is controlled by an allowlist in `agent/tools_config.py` — only the tools listed below are visible to the model. Brave is listed **last** so local models (which have recency bias) reliably find web tools even when many others are present.

### 🔍 Brave Search — all users

Web search via the [Brave Search API](https://brave.com/search/api/). All tools exposed (allowlist = `None`).

| Tool | Description |
|---|---|
| `brave_web_search` | Real-time web search |
| `brave_local_search` | Location-aware local business/place search |

### 🌐 Chrome browsing — privileged users

Real page browsing via [Playwright MCP](https://github.com/microsoft/playwright-mcp), driving a Chrome instance with its own persistent profile. Search finds a URL; browsing reads what's actually on it — full articles, pages behind a cookie wall, or pages behind a login the profile already holds. **Off by default**; set `PLAYWRIGHT_ENABLED=true`.

| Tool | Description |
|---|---|
| `browser_navigate` / `browser_navigate_back` | Go to a URL, or back in history |
| `browser_snapshot` | Accessibility-tree text of the page — the main way Leo *reads* |
| `browser_take_screenshot` | Pixels, for when you want to *see* the page |
| `browser_click` / `browser_hover` / `browser_drag` | Pointer interaction |
| `browser_type` / `browser_fill_form` / `browser_press_key` / `browser_select_option` | Text and form input |
| `browser_evaluate` / `browser_run_code` | Run JavaScript, or a Playwright snippet, against the page |
| `browser_file_upload` | Attach a local file to a form |
| `browser_handle_dialog` | Accept or dismiss a native dialog |
| `browser_tabs` / `browser_resize` / `browser_wait_for` / `browser_close` | Page and tab management |
| `browser_console_messages` / `browser_network_requests` | Console and network inspection |

**The profile.** `PLAYWRIGHT_USER_DATA_DIR` (default `~/.cache/whatsapp-leo/playwright-profile`) starts logged out, so Leo reaches only public pages. To give it authenticated access, sign that profile in once by hand — stop the agent first, as Chrome locks the directory to a single process:

```bash
systemctl --user stop whatsapp-leo-agent
npx @playwright/mcp@0.0.70 --browser chrome \
  --user-data-dir ~/.cache/whatsapp-leo/playwright-profile
```

That needs a display. On a headless box, `ssh -X` in from a desktop (needs `xauth` server-side), use the physical console, or run it under Xvfb + x11vnc. Keep the profile dedicated to Leo: whatever it is signed into is what a page that successfully injects Leo can act on.

**Headless.** `PLAYWRIGHT_HEADLESS` defaults to `true`, since Leo normally runs as a systemd user unit with no `DISPLAY`. Some sites bot-detect headless Chrome; the fallback is a virtual display (`Xvfb :99`, `Environment=DISPLAY=:99` in the unit) with `PLAYWRIGHT_HEADLESS=false`. `PLAYWRIGHT_NO_SANDBOX` defaults to `true` because recent kernels block Chrome's own sandbox under a user unit — turn it off wherever the sandbox works.

**Safety.** A web page is attacker-controlled text arriving in a privileged agent, so run browsing with `REQUIRE_WRITE_CONFIRMATION=true`. Reading (`browser_navigate`, `browser_snapshot`, …) is never gated; acting (`browser_click`, `browser_type`, `browser_fill_form`, `browser_evaluate`, `browser_run_code`, `browser_file_upload`, …) is refused at the MCP boundary until you reply with a confirmation.

### 📅 Google Workspace — privileged users

Calendar, Gmail, and auth tools via the [Workspace MCP](https://github.com/gemini-cli-extensions/workspace).

**Calendar**

| Tool | Description |
|---|---|
| `calendar.list` | List all calendars |
| `calendar.listEvents` | List events from a calendar (defaults to upcoming) |
| `calendar.getEvent` | Get details of a specific event |
| `calendar.createEvent` | Create a new calendar event |
| `calendar.updateEvent` | Update an existing event |
| `calendar.deleteEvent` | Delete an event |
| `calendar.findFreeTime` | Find a free time slot across multiple people |
| `calendar.respondToEvent` | Accept, decline, or mark tentative for a meeting invite |

**Gmail**

| Tool | Description |
|---|---|
| `gmail.search` | Search emails using Gmail query syntax |
| `gmail.get` | Get the full content of a specific email |
| `gmail.createDraft` | Create a draft email |
| `gmail.downloadAttachment` | Download an attachment to a local file |

**Auth**

| Tool | Description |
|---|---|
| `auth.clear` | Clear credentials, forcing re-login on next request |
| `auth.refreshToken` | Manually trigger token refresh |

### 🏃 Garmin Connect — privileged users

Fitness and health data via [`garmin_mcp`](https://github.com/Taxuspt/garmin_mcp).

**Daily stats & body**

| Tool | Description |
|---|---|
| `get_stats` | Daily activity stats (curated essential metrics) |
| `get_user_summary` | User summary (compatible with garminconnect-ha) |
| `get_stats_and_body` | Stats and body composition combined |
| `get_body_composition` | Body composition for a date or range |
| `get_weigh_ins` | Weight measurements between dates |
| `get_fitnessage_data` | Fitness age |
| `get_user_profile` | User profile information |

**Sleep**

| Tool | Description |
|---|---|
| `get_sleep_data` | Full sleep data with all details |
| `get_sleep_summary` | Sleep summary (lightweight, essential metrics only) |

**Heart rate & HRV**

| Tool | Description |
|---|---|
| `get_heart_rates` | Full heart rate time-series |
| `get_heart_rates_summary` | Heart rate summary (lightweight) |
| `get_hrv_data` | Heart Rate Variability data |
| `get_rhr_day` | Resting heart rate |

**Stress & recovery**

| Tool | Description |
|---|---|
| `get_stress_data` | Full stress time-series |
| `get_stress_summary` | Stress summary (lightweight) |
| `get_body_battery` | Body battery with events |
| `get_body_battery_events` | Body battery events |
| `get_training_readiness` | Training readiness score |
| `get_training_status` | Training status |
| `get_all_day_stress` | All-day stress data |
| `get_weekly_stress` | Weekly stress aggregates |

**Steps & activity**

| Tool | Description |
|---|---|
| `get_steps_data` | Detailed steps with 15-minute intervals |
| `get_daily_steps` | Steps for a date range |
| `get_weekly_steps` | Weekly step aggregates |
| `get_floors` | Floors climbed |
| `get_weekly_intensity_minutes` | Weekly intensity minutes |
| `get_all_day_events` | Daily wellness events |

**Activities**

| Tool | Description |
|---|---|
| `get_activities` | Activities with pagination |
| `get_activity` | Basic activity info |
| `get_activities_by_date` | Activities between dates, optionally filtered by type |
| `count_activities` | Total activity count |
| `get_activity_types` | All available activity types |
| `get_activity_hr_in_timezones` | Heart rate in time zones for an activity |
| `get_activity_split_summaries` | Split summaries for an activity |
| `get_activity_splits` | Splits for an activity |
| `get_activity_typed_splits` | Typed splits for an activity |
| `get_activity_weather` | Weather data for an activity |
| `get_training_effect` | Training effect for an activity |

**Training & performance**

| Tool | Description |
|---|---|
| `get_endurance_score` | Endurance score between dates |
| `get_hill_score` | Hill score between dates |
| `get_lactate_threshold` | Lactate threshold data |
| `get_personal_record` | Personal records |
| `get_race_predictions` | Predicted race times based on current fitness |
| `get_respiration_data` | Full respiration time-series |
| `get_respiration_summary` | Respiration summary (lightweight) |
| `get_scheduled_workouts` | Scheduled workouts between two dates |
| `get_training_plan_workouts` | Training plan workouts for the week |
| `get_workouts` | All workouts (curated summary) |
| `get_workout_by_id` | Detailed info for a specific workout |
| `get_progress_summary_between_dates` | Progress summary for a metric between dates |

### 🐦 X (Twitter) — privileged users

Tweet fetching via [twikit](https://github.com/d60/twikit) (cookie-based, no API key required).

| Tool | Description |
|---|---|
| `get_user_tweets` | Fetch recent tweets from any public account by username |
| `search_tweets` | Search tweets by keyword, hashtag, or `from:username` syntax |

---

## Tool Selection Framework

With many MCP servers active, models can be overwhelmed by hundreds of tools. Leo uses a per-server allowlist in **`agent/tools_config.py`** to expose only what's needed.

`make_tool_filter(server_name)` converts these lists into `MCPServerStdio(tool_filter=...)` calls that the SDK applies before handing tools to the model. `cache_tools_list=True` is set on every server so the filtered list is fetched once per session.

To disable a tool, delete or comment out its line in `tools_config.py`. The inline description tells you what it does.

### Syncing after adding a new MCP

When you add a new MCP server to `MCP_REGISTRY` in `config.py`, run:

```bash
# Dry-run: see what would change
uv run scripts/update_tools_config.py

# Write changes (new server added; descriptions refreshed on all servers)
uv run scripts/update_tools_config.py --write

# Also pull in newly available tools on existing servers
uv run scripts/update_tools_config.py --write --add-new
```

**Merge rules:**

| Situation | Behaviour |
|---|---|
| New server in `MCP_REGISTRY` | Added to `TOOL_CONFIG` with `None` (all tools) |
| Existing entry = `None` | Unchanged |
| Tool removed from MCP server | Pruned from allowlist |
| New tool on existing server | Reported only — not added unless `--add-new` |
| Server missing from registry | Warning printed, config left intact |

### Adding a new MCP server (full checklist)

1. Add the params dict in `config.py` (e.g. `_mytool_mcp_params = {...}`)
2. Register it in `MCP_REGISTRY`: `"mytool": {"params": _mytool_mcp_params, "timeout": 60}`
3. Wire it into `mcp_stack` with `tool_filter=make_tool_filter("mytool")`
4. Run `uv run scripts/update_tools_config.py --write` to populate `TOOL_CONFIG` with all tools and their descriptions
5. Delete lines in `tools_config.py` for tools you don't need — the inline descriptions make it easy to decide

## Prerequisites

| Dependency | Purpose |
|---|---|
| **Python ≥ 3.13** | Agent server |
| **[uv](https://docs.astral.sh/uv/)** | Python package manager |
| **Go** | WhatsApp bridge |
| **[Ollama](https://ollama.com)** | Local LLM inference |
| **Node.js / npm** | Brave Search MCP, Playwright MCP, Workspace MCP |
| **Google Chrome** | Chrome browsing (optional; or `npx playwright install chromium` with `PLAYWRIGHT_BROWSER=chromium`) |
| **[faster-whisper](https://github.com/SYSTRAN/faster-whisper)** | Voice note transcription (installed via `uv sync`) |

## Setup

### Quick start

```bash
git clone https://github.com/shansin/whatsapp-leo.git
cd whatsapp-leo
./setup.sh
```

`setup.sh` is a one-shot interactive installer that:

1. Checks all system dependencies (Python 3.13+, uv, Go, Node.js, Ollama)
2. Installs Python packages (`uv sync`) and builds the Go bridge
3. Downloads and installs the [Google Workspace MCP](#-google-workspace--privileged-users) as a sibling directory (`../linux.google-workspace-extension`)
4. Pulls the default Ollama models (`qwen3.5:35b`, `gemma3:27b`)
5. Walks you through `.env` configuration interactively (API keys, phone numbers, model choices)
6. Offers to authenticate X/Twitter on the spot
7. Explains how each remaining service authenticates on first use

Every value can be changed later by editing `.env` directly.

### Environment variables

| Variable | Description | Default |
|---|---|---|
| `INSTANCE_GUID` | Unique ID for this instance (allows multiple instances) | `default` |
| `OLLAMA_BASE_URL` | Ollama API endpoint | `http://localhost:11434/v1` |
| `MODEL_NAME` | Ollama model to use (e.g. `qwen3.5:35b`) | — |
| `VISION_MODEL_NAME` | Ollama model for image messages | `gemma3:27b` |
| `MAX_IMAGE_DIMENSION` | Max pixel dimension before downscaling images | `1280` |
| `WHISPER_MODEL_SIZE` | faster-whisper model (tiny/base/small/medium/large, or distil-*.en) | `distil-medium.en` |
| `WHISPER_BEAM_SIZE` | Decoding beam width; 1 is greedy and much faster | `1` |
| `WHISPER_VAD_FILTER` | Skip silence before transcribing | `true` |
| `WHISPER_PREWARM` | Load the transcription model at startup | `true` |
| `MAX_AGENTS` | Max cached agent instances (LRU eviction) | `20` |
| `TTL_SECONDS` | Agent cache TTL | `1800` |
| `DEFAULT_TZ` | Instance default timezone (per-user override via `#tz`) | `America/Los_Angeles` |
| `USER_PREFS_PATH` | Where per-user preferences are stored | `store/user_prefs.json` |
| `MAX_SESSION_ITEMS` | Rolling window of conversation items kept per chat (0 = unlimited) | `40` |
| `SESSIONS_DB_PATH` | Where conversation history is stored | `store/sessions.db` |
| `DEBOUNCE_SECONDS` | Window for merging rapid consecutive messages (0 disables) | `2.5` |
| `MAX_BURST_MESSAGES` | Flush a burst once it reaches this many messages | `10` |
| `MAX_REPLY_CHARS` | Split replies longer than this | `3500` |
| `SHARE_DIR` | Extra directory Leo may send files from | — |
| `MAX_SEND_BYTES` | Largest file Leo will send | `64MB` |
| `VOICE_REPLIES` / `TTS_COMMAND` | Reply to voice notes with a voice note | off |
| `LOG_FILE` / `LOG_MAX_BYTES` / `LOG_BACKUP_COUNT` | Rotated file log | off / 10MB / 5 |
| `LOG_LEVEL` | Root log level | `INFO` |
| `PRESENCE_ENABLED` | Show a WhatsApp typing indicator while a run is in flight | `true` |
| `PRESENCE_REFRESH_SECONDS` | How often the typing indicator is refreshed | `8` |
| `AGENTS_TRACING_ENABLED` | Export Agents SDK run traces to OpenAI (off — traces contain message content) | `false` |
| `GARMIN_MCP_COMMAND` | Path to a locally installed `garmin-mcp` binary | auto-detected |
| `GARMIN_MCP_REF` | garmin_mcp commit used when falling back to `uvx` | pinned SHA |
| `ALLOWED_SENDERS` | Comma-separated phone numbers for privileged access | — |
| `LEO_MENTION_ID` | WhatsApp mention ID for group triggers | — |
| `IS_DEDICATED_NUMBER` | `true` if Leo has its own phone number | `false` |
| `IS_TEST_MODE` | `true` to launch Gradio UI instead of WhatsApp bridge | `false` |
| `OPENAI_API_KEY` | Required by the OpenAI Agents SDK (can be a dummy value for Ollama) | — |
| `BRAVE_API_KEY` | API key for Brave Search | — |
| `REMINDER_POLL_INTERVAL` | Seconds between reminder scheduler polls | `60` |
| `BRIEFING_POLL_INTERVAL` | Seconds between briefing scheduler polls | `60` |
| `MESSAGES_DB_PATH` | Shared messages database (bridge writes, agent reads) | `store/messages.db` |
| `REQUIRE_WRITE_CONFIRMATION` | Refuse write-capable tool calls until the user confirms in a reply | `false` |
| `IS_HOOK_ENABLED` | Enable the hooks system | `false` |
| `HOOKS` | Comma-separated hook names (e.g. `claude,claude-session`) | — |
| `WORKSPACE_MCP_PATH` | Path to the Workspace MCP server `dist/index.js` | — |
| `PLAYWRIGHT_ENABLED` | Enable Chrome browsing (privileged users only) | `false` |
| `PLAYWRIGHT_BROWSER` | `chrome` \| `chromium` \| `msedge` \| `firefox` \| `webkit` | `chrome` |
| `PLAYWRIGHT_HEADLESS` | Run without a display; `false` needs a real display or Xvfb | `true` |
| `PLAYWRIGHT_NO_SANDBOX` | Pass `--no-sandbox`; needed where the kernel blocks Chrome's sandbox | `true` |
| `PLAYWRIGHT_USER_DATA_DIR` | Persistent Chrome profile directory | `~/.cache/whatsapp-leo/playwright-profile` |
| `PLAYWRIGHT_VIEWPORT` | Browser viewport size | `1280x800` |
| `X_COOKIE_PATH` | Path to the X (Twitter) session cookie file | `/tmp/x_cookies.json` |

### Service authentication

Each MCP service authenticates independently. `setup.sh` handles or explains all of these, but here's the full reference:

#### WhatsApp (QR code — first run)

The Go bridge uses WhatsApp's multidevice API and authenticates via QR code — no phone number or SIM is required on the server.

1. Run `./start_services.sh`
2. On **first run**, the bridge prints a QR code in the terminal
3. Open WhatsApp on your phone → **Settings → Linked Devices → Link a Device**
4. Scan the QR code

The session is persisted in SQLite (`whatsapp-mcp/whatsapp-bridge/`) so subsequent starts skip the QR step. If the session expires, delete the bridge's `.db` files and re-scan.

#### Google Workspace (OAuth — first use)

The Workspace MCP is downloaded by `setup.sh` from [gemini-cli-extensions/workspace](https://github.com/gemini-cli-extensions/workspace/releases) into `../linux.google-workspace-extension` (sibling to the Leo directory). The path is auto-configured in `.env` as `WORKSPACE_MCP_PATH`.

To set up manually instead:

```bash
# Download the latest release (Linux example)
mkdir -p ../linux.google-workspace-extension
curl -fsSL https://github.com/gemini-cli-extensions/workspace/releases/latest/download/linux.google-workspace-extension.tar.gz \
  | tar xz -C ../linux.google-workspace-extension
cd ../linux.google-workspace-extension && npm install
```

Then set in `.env`:
```
WORKSPACE_MCP_PATH=<absolute-path-to>/linux.google-workspace-extension/dist/index.js
```

On **first use**, the Workspace MCP opens your browser for Google OAuth consent. Tokens are stored locally in `gemini-cli-workspace-token.json` inside the extension directory and refreshed automatically.

#### Garmin Connect (interactive login — first use)

The Garmin MCP server runs via `uvx git+https://github.com/Taxuspt/garmin_mcp` — no env vars or API keys needed. On **first use**, it prompts interactively for your Garmin credentials. OAuth tokens are stored in `~/.garminconnect/` and refreshed automatically on subsequent starts.

To re-authenticate (e.g. if tokens expire), delete `~/.garminconnect/` and restart Leo.

#### X / Twitter (browser cookies — one-time)

The X MCP uses cookie-based auth via [twikit](https://github.com/d60/twikit). No API key or developer account required — just a browser session.

1. Log in to [x.com](https://x.com) in Firefox or Chrome/Chromium
2. Run the setup script (or let `setup.sh` do it for you):

```bash
uv run python x-mcp/x-mcp-server/setup.py           # auto-detects Firefox then Chrome
uv run python x-mcp/x-mcp-server/setup.py --firefox
uv run python x-mcp/x-mcp-server/setup.py --chrome
```

Cookies are saved to `X_COOKIE_PATH` and loaded silently on every start. To re-authenticate, log in to x.com in your browser and re-run the script.

### Start services

```bash
./start_services.sh
```

This script:
1. Sources `.env`
2. Starts the Python agent server (`uv run python agent/agent.py`)
3. Builds and starts the Go WhatsApp bridge (skipped if `IS_TEST_MODE=true`)
4. Prints Unix socket paths and hook FIFO paths (if enabled)
5. Handles graceful shutdown on `Ctrl+C`

It waits for the agent's socket to appear before starting the bridge, rather
than sleeping and hoping.

### Running under systemd (recommended for always-on)

`start_services.sh` backgrounds both processes and dies with your shell. For an
always-on instance use the user units in `deploy/`, which add
`Restart=on-failure`:

```bash
mkdir -p ~/.config/systemd/user
for unit in agent bridge; do
  sed "s|%WORKDIR%|$PWD|g" deploy/whatsapp-leo-$unit.service \
    > ~/.config/systemd/user/whatsapp-leo-$unit.service
done
systemctl --user daemon-reload
systemctl --user enable --now whatsapp-leo-agent whatsapp-leo-bridge
loginctl enable-linger "$USER"      # keep running after logout
journalctl --user -u whatsapp-leo-agent -f
```

Pair the bridge once interactively (QR scan) before enabling the unit.

### Test mode (optional)

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
│   ├── config.py             # Environment config, model/MCP singletons, MCP_REGISTRY
│   ├── mcp_pool.py           # Long-lived MCP servers (one process each) + auto-restart
│   ├── session_store.py      # Durable per-chat history with a rolling window
│   ├── timeutil.py           # UTC storage format for scheduled times
│   ├── write_guard.py        # Optional confirmation gate for write-capable tools
│   ├── user_prefs.py         # Per-user preferences (timezone)
│   ├── history_tools.py      # Read-only chat-history search tools
│   ├── send_tools.py         # Send files back to chat (path-restricted)
│   ├── reply.py              # Reply splitting + optional TTS voice replies
│   ├── debounce.py           # Merges rapid consecutive messages into one turn
│   ├── sqlite_store.py       # Shared SQLite connection + polling-scheduler base
│   ├── tools_config.py       # Per-server tool allowlists (TOOL_CONFIG)
│   ├── agent_factory.py      # LRU-cached Agent instances + reminder parser
│   ├── message_handler.py    # Core message routing (hooks, commands, AI, vision, audio)
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
├── scripts/
│   └── update_tools_config.py  # Sync tools_config.py from live MCP tool lists
├── x-mcp/                    # X (Twitter) MCP
│   └── x-mcp-server/
│       ├── main.py           # FastMCP server (get_user_tweets, search_tweets)
│       └── setup.py          # One-time interactive login → saves cookies
├── whatsapp-mcp/             # Forked WhatsApp MCP project
│   ├── whatsapp-bridge/      # Go bridge (whatsmeow + SQLite)
│   └── whatsapp-mcp-server/  # Python MCP server for WhatsApp tools
├── store/                    # SQLite databases (gitignored)
│   ├── messages.db
│   ├── sessions.db
│   ├── reminders.db
│   └── briefings.db
├── deploy/                   # systemd user units (Restart=on-failure)
├── tests/                    # pytest suite (no WhatsApp/Ollama required)
├── .github/workflows/ci.yml  # ruff + pytest + go vet/build + shell checks
├── start_services.sh         # Service orchestration script
├── pyproject.toml            # Python project config (uv) — direct deps only, uv.lock pins the rest
├── .env_example              # Environment variable template
└── .python-version           # Python 3.13
```

## Development

```bash
uv sync --all-groups     # install runtime + dev dependencies
uv run pytest -q         # 151 tests, no WhatsApp or Ollama needed
uv run ruff check agent/ tests/ scripts/
uv run mypy              # advisory; the tree is only partly annotated
(cd whatsapp-mcp/whatsapp-bridge && go vet ./... && go build ./...)
```

CI (`.github/workflows/ci.yml`) runs the same checks on push and PR.

## WhatsApp Commands

| Command | Description |
|---|---|
| `#help` | List every command |
| `#remindme <time> <message>` | Set a one-time reminder |
| `#remindme list` | List pending one-time reminders |
| `#remindme cancel <id>` | Cancel a pending one-time reminder |
| `snooze 10m` (as a reply) | Push a fired reminder back by 10m / 2h / 1d |
| `#reminder add "schedule" message` | Create a recurring reminder |
| `#reminder list` | List all recurring reminders |
| `#reminder remove <id>` | Remove a recurring reminder |
| `#reminder remove-all` | Remove all recurring reminders |
| `#briefing add "Name" "Schedule" Prompt` | Create a scheduled briefing |
| `#briefing list` | List all briefings |
| `#briefing remove <id>` | Remove a briefing |
| `#briefing remove-all` | Remove all briefings |
| `#briefing run <id>` | Run a briefing immediately (test it) |
| `#briefing pause <id>` / `#briefing resume <id>` | Disable or re-enable a briefing |
| `#tz` / `#tz Europe/London` | Show or set your timezone |
| `#model` | List the models Ollama has installed |
| `#model set <name>` / `#model vision <name>` | Switch the main or vision model live |
| `#model reset` | Go back to the `MODEL_NAME` / `VISION_MODEL_NAME` defaults |
| `#status` | Model in use, MCP health, uptime, recent errors |
| `#hook-name <message>` | Send message to a named hook |
| `#hook-name #start` | Start a hook session — all messages forwarded to hook |
| `#hook-name #stop` | End a hook session — resume normal Leo processing |
