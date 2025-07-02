import asyncio
import numpy as np
from pathlib import Path

from farm_ng.core.events_file_reader import proto_from_json_file
from farm_ng.core.event_client import EventClient
from farm_ng.image_decoder import ImageDecoder
from farm_ng.core.event_service_pb2 import EventServiceConfig

from inference import annotate_image

latest_frame: np.ndarray = None

async def run_camera_stream(service_config_path: Path):
    global latest_frame
    config = proto_from_json_file(service_config_path, EventServiceConfig())
    image_decoder = ImageDecoder()

    async for event, message in EventClient(config).subscribe(config.subscriptions[0], decode=True):
        image = np.from_dlpack(image_decoder.decode(message.image_data))
        latest_frame = annotate_image(image)

