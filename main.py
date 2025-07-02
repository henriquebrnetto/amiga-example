import asyncio
from fastapi import FastAPI, WebSocket
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path

from camera_stream import run_camera_stream, latest_frame
from utils import encode_image_to_jpeg, image_to_base64

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    print("Starting image stream...")
    asyncio.create_task(run_camera_stream(Path("configs/camera_config.json")))

@app.get("/latest_frame")
async def get_latest_frame():
    if latest_frame is None:
        return JSONResponse(content={"error": "No image yet"}, status_code=503)
    return StreamingResponse(io.BytesIO(encode_image_to_jpeg(latest_frame)), media_type="image/jpeg")

@app.websocket("/stream")
async def stream_frames(websocket: WebSocket):
    await websocket.accept()
    while True:
        if latest_frame is not None:
            await websocket.send_text(image_to_base64(latest_frame))
        await asyncio.sleep(0.1)