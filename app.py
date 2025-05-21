import os
os.system("pip uninstall -y aioice")
os.system("pip install --no-cache-dir aioice")
os.system("pip uninstall -y asyncio")
os.system("pip install --no-cache-dir asyncio")

import asyncio
import base64
import json
import secrets
import signal
from pathlib import Path

import gradio as gr
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from fastrtc import (
    AdditionalOutputs,
    AsyncStreamHandler,
    Stream,
    get_cloudflare_turn_credentials_async,
    wait_for_item,
)
from gradio.utils import get_space
from websockets.asyncio.client import connect

load_dotenv()

cur_dir = Path(__file__).parent

API_KEY = "Your_API_KEY"
API_URL = "wss://dashscope.aliyuncs.com/api-ws/v1/realtime?model=qwen-omni-turbo-realtime-2025-03-26"
VOICES = ["Chelsie", "Serena", "Ethan", "Cherry"]
headers = {"Authorization": "Bearer " + API_KEY}


class QwenOmniHandler(AsyncStreamHandler):
    def __init__(self) -> None:
        super().__init__(
            expected_layout="mono",
            output_sample_rate=24_000,
            input_sample_rate=16_000,
        )
        self.connection = None
        self.output_queue = asyncio.Queue()

    def copy(self):
        return QwenOmniHandler()

    @staticmethod
    def msg_id() -> str:
        return f"event_{secrets.token_hex(10)}"

    async def start_up(self):
        try:
            await self.wait_for_args()
            voice_id = self.latest_args[1]
            print(voice_id)

            async with connect(API_URL, additional_headers=headers) as conn:
                self.client = conn
                await conn.send(
                    json.dumps(
                        {
                            "event_id": self.msg_id(),
                            "type": "session.update",
                            "session": {
                                "modalities": ["text", "audio"],
                                "voice": voice_id,
                                "input_audio_format": "pcm16",
                            },
                        }
                    )
                )
                self.connection = conn

                async for data in self.connection:
                    event = json.loads(data)
                    if "type" not in event:
                        continue

                    if event["type"] == "input_audio_buffer.speech_started":
                        self.clear_queue()
                    if event["type"] == "conversation.item.input_audio_transcription.completed":
                        await self.output_queue.put(
                            AdditionalOutputs({"role": "user", "content": event["transcript"]})
                        )
                    if event["type"] == "response.audio_transcript.done":
                        await self.output_queue.put(
                            AdditionalOutputs({"role": "assistant", "content": event["transcript"]})
                        )
                    if event["type"] == "response.audio.delta":
                        await self.output_queue.put(
                            (self.output_sample_rate, np.frombuffer(base64.b64decode(event["delta"]), dtype=np.int16).reshape(1, -1))
                        )

        except Exception as e:
            print(f"Connection error: {e}")
            await self.shutdown()

    async def receive(self, frame: tuple[int, np.ndarray]) -> None:
        if not self.connection:
            return
        _, array = frame
        array = array.squeeze()
        audio_message = base64.b64encode(array.tobytes()).decode("utf-8")
        await self.connection.send(
            json.dumps(
                {
                    "event_id": self.msg_id(),
                    "type": "input_audio_buffer.append",
                    "audio": audio_message,
                }
            )
        )

    async def emit(self) -> tuple[int, np.ndarray] | AdditionalOutputs | None:
        return await wait_for_item(self.output_queue)

    async def shutdown(self) -> None:
        if self.connection:
            await self.connection.close()
            self.connection = None

        # 清空队列
        while not self.output_queue.empty():
            self.output_queue.get_nowait()


def update_chatbot(chatbot: list[dict], response: dict):
    chatbot.append(response)
    return chatbot


chatbot = gr.Chatbot(type="messages")
voice = gr.Dropdown(choices=VOICES, value=VOICES[0], type="value", label="Voice")
latest_message = gr.Textbox(type="text", visible=False)

# 可选：暂时禁用 TURN 配置进行测试
rtc_config = get_cloudflare_turn_credentials_async if get_space() else None
# rtc_config = None  # 取消注释可禁用 TURN 测试

stream = Stream(
    QwenOmniHandler(),
    mode="send-receive",
    modality="audio",
    additional_inputs=[voice, chatbot],
    additional_outputs=[chatbot],
    additional_outputs_handler=update_chatbot,
    rtc_configuration=rtc_config,
    concurrency_limit=5 if get_space() else None,
    time_limit=90 if get_space() else None,
)

app = FastAPI()

stream.mount(app)


@app.get("/")
async def _():
    rtc_config = await get_cloudflare_turn_credentials_async() if get_space() else None
    html_content = (cur_dir / "index.html").read_text()
    html_content = html_content.replace("__RTC_CONFIGURATION__", json.dumps(rtc_config))
    return HTMLResponse(content=html_content)


@app.get("/outputs")
def _(webrtc_id: str):
    async def output_stream():
        import json

        async for output in stream.output_stream(webrtc_id):
            s = json.dumps(output.args[0])
            yield f"event: output\ndata: {s}\n\n"

    return StreamingResponse(output_stream(), media_type="text/event-stream")


def handle_exit(sig, frame):
    print("Shutting down gracefully...")
    # 可扩展为执行更多清理逻辑


signal.signal(signal.SIGINT, handle_exit)
signal.signal(signal.SIGTERM, handle_exit)

if __name__ == "__main__":
    import os

    if (mode := os.getenv("MODE")) == "UI":
        stream.ui.launch(server_port=7860)
    elif mode == "PHONE":
        stream.fastphone(host="0.0.0.0", port=7860)
    else:
        import uvicorn

        uvicorn.run(app, host="0.0.0.0", port=7860)
