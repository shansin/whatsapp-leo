"""Gradio test mode UI for WhatsApp Leo agent."""

import time
import asyncio
from dataclasses import asdict

import gradio as gr

from config import (
    MODEL_NAME,
    ALLOWED_SENDERS,
    IS_DEDICATED_NUMBER,
)
from models import ReceivedMessage
from agent_factory import agent_factory
from message_handler import process_message
from briefing_executor import execute_briefing_prompt
from logging_setup import logger, log_deque
from whatsapp import send_message as whatsapp_send_message
from reminder import ReminderScheduler, RecurringReminderScheduler
from briefing import BriefingScheduler
import ollama_models


def get_ollama_models():
    """Model names from the local Ollama, or a usable guess if it is unreachable."""
    try:
        return ollama_models.list_models_sync()
    except Exception as e:
        logger.error(f"Failed to fetch Ollama models: {e}")
        return [MODEL_NAME] if MODEL_NAME else ["llama3"]


def get_logs():
    return "\n".join(log_deque)


def start_test_ui():
    """Start the Gradio testing UI instead of the Unix socket server."""
    import config  # mutable access to module-level vars

    logger.info("Starting Gradio Test Mode UI...")

    available_models = get_ollama_models()
    default_model = MODEL_NAME if MODEL_NAME in available_models else (available_models[0] if available_models else "")

    with gr.Blocks(title="WhatsApp Leo - Test Mode") as app:
        gr.Markdown("# 🦁 WhatsApp Leo - Test Mode")

        with gr.Row():
            with gr.Column(scale=2):
                model_dropdown = gr.Dropdown(
                    choices=available_models,
                    value=default_model,
                    label="🧠 Active Model (Update to swap models)",
                    allow_custom_value=True,
                    interactive=True
                )

                chat_interface = gr.Chatbot(height=600, show_label=False)
                msg_input = gr.Textbox(placeholder="Send a test message to Leo...", show_label=False)

                with gr.Row():
                    clear_btn = gr.Button("🗑️ Clear Chat")

            with gr.Column(scale=1):
                gr.Markdown("### 📜 System Logs")
                logs_output = gr.Textbox(
                    label="Agent Logs",
                    lines=35,
                    max_lines=35,
                    interactive=False,
                    autoscroll=True,
                    show_label=False
                )
                timer = gr.Timer(1)
                timer.tick(get_logs, inputs=None, outputs=logs_output)

        def update_model(new_model):
            # Mutates the shared FallbackModel rather than rebinding it, so the
            # swap reaches briefings and the reminder parser too.
            config.set_text_model(new_model)
            agent_factory.clear()
            gr.Info(f"Model updated to {new_model}")

        model_dropdown.change(update_model, inputs=[model_dropdown], outputs=[])

        async def submit_message(user_text, history):
            if not user_text.strip():
                yield history
                return

            history.append({"role": "user", "content": user_text})
            # Add a placeholder for the assistant's reply
            history.append({"role": "assistant", "content": "..."})
            yield history

            # To trigger DM logic when IS_DEDICATED_NUMBER is True
            fake_jid = "test@lid"
            msg_id = f"TEST_{int(time.time()*1000)}"
            sender_phone = ALLOWED_SENDERS[0] if ALLOWED_SENDERS else "1234567890"

            # Prepend @leo to ensure it triggers if not dedicated
            text = user_text
            if not text.lower().startswith("@leo") and not IS_DEDICATED_NUMBER:
                text = f"@leo {text}"

            msg = ReceivedMessage(
                chat_jid=fake_jid,
                chat_name="Test User",
                content=text,
                file_length=0,
                filename="",
                id=msg_id,
                is_from_me=False,
                media_type="",
                phone_number=sender_phone,
                sender="Test User",
                sender_jid=fake_jid,
                timestamp=str(int(time.time())),
                url=""
            )

            loop = asyncio.get_running_loop()
            response_queue = asyncio.Queue()

            def mock_send(to_jid, text, **kwargs):
                if to_jid == fake_jid:
                    loop.call_soon_threadsafe(response_queue.put_nowait, text)
                else:
                    logger.info(f"[TEST MODE SEND] To {to_jid}: {text}")
                return True, "Mock sent"

            # Monkey-patch the whatsapp send function for test mode
            import message_handler
            original_send = message_handler.whatsapp_send_message
            message_handler.whatsapp_send_message = mock_send

            try:
                process_task = asyncio.create_task(process_message(asdict(msg)))

                reply_text = None
                try:
                    reply_text = await asyncio.wait_for(response_queue.get(), timeout=120.0)
                except TimeoutError:
                    reply_text = "❌ Request timed out."

                # Update the placeholder content
                history[-1]["content"] = reply_text
                yield history
                await process_task
            except Exception as e:
                logger.error(f"Test UI execution failed: {e}", exc_info=True)
                history[-1]["content"] = f"❌ Error: {str(e)}"
                yield history
            finally:
                message_handler.whatsapp_send_message = original_send

        msg_input.submit(
            submit_message,
            inputs=[msg_input, chat_interface],
            outputs=[chat_interface]
        ).then(lambda: "", None, msg_input)

        clear_btn.click(lambda: [], None, chat_interface, queue=False)

    # Launch with prevent_thread_lock so we can run background schedulers too
    app.launch(server_name="127.0.0.1", server_port=7860, quiet=True, prevent_thread_lock=True, theme=gr.themes.Soft())

    # Run the background schedulers that main() normally runs
    async def run_schedulers():
        scheduler = ReminderScheduler(send_fn=whatsapp_send_message)
        asyncio.create_task(scheduler.run())

        briefing_scheduler = BriefingScheduler(
            execute_fn=execute_briefing_prompt,
            send_fn=whatsapp_send_message,
        )
        asyncio.create_task(briefing_scheduler.run())

        recurring_scheduler = RecurringReminderScheduler(send_fn=whatsapp_send_message)
        await recurring_scheduler.run()

    asyncio.run(run_schedulers())


if __name__ == "__main__":
    start_test_ui()
