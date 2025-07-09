# save_and_view_camera.py
import asyncio
import numpy as np
import cv2
import os

from ultralytics import YOLO
from inference import annotate_image

from google.protobuf.empty_pb2 import Empty

from farm_ng.core.event_client      import EventClient
from farm_ng.core.event_service_pb2 import EventServiceConfig, SubscribeRequest
from farm_ng.core.events_file_reader import proto_from_json_file, payload_to_protobuf
from farm_ng.core.uri_pb2           import Uri

os.environ['QT_QPA_PLATFORM'] = 'xcb'
model = YOLO("my_model.pt")

async def process_rgb_stream(client):
    """Processa o stream de imagens RGB"""
    sub = SubscribeRequest(uri=Uri(path="/rgb", query="service_name=oak0"), every_n=1)
    
    try:
        async for event, message in client.subscribe(sub, decode=True):
            # cast image data bytes to numpy and decode
            image: np.ndarray = cv2.imdecode(np.frombuffer(message.image_data, dtype="uint8"), cv2.IMREAD_UNCHANGED)

            cv2.namedWindow("RGB Image", cv2.WINDOW_NORMAL)
            # cv2.imshow("RGB Image", annotate_image(image))
            cv2.imshow("RGB Image", image)
            cv2.waitKey(1)
    except asyncio.CancelledError:
        print("RGB stream foi cancelado")
        raise
    except Exception as e:
        print(f"Erro no RGB stream: {e}")

async def process_disparity_stream(client):
    """Processa o stream de disparidade"""
    disparity_sub = SubscribeRequest(uri=Uri(path="/disparity", query="service_name=oak0"), every_n=1)
    
    try:
        async for event, message in client.subscribe(disparity_sub, decode=True):
            # cast disparity data bytes to numpy and decode
            disparity: np.ndarray = cv2.imdecode(np.frombuffer(message.disparity_data, dtype="uint8"), cv2.IMREAD_UNCHANGED)

            cv2.namedWindow("Disparity", cv2.WINDOW_NORMAL)
            cv2.imshow("Disparity", disparity)
            print(f"Disparity shape: {disparity.shape}, dtype: {disparity.dtype}")
            cv2.waitKey(1)
    except asyncio.CancelledError:
        print("Disparity stream foi cancelado")
        raise
    except Exception as e:
        print(f"Erro no disparity stream: {e}")

async def get_gps(client):
    """Obtém a localização GPS do serviço"""
    gps_sub = SubscribeRequest(uri=Uri(path="/relposned", query="service_name=gps"), every_n=1)
    
    async for event, message in client.subscribe(gps_sub, decode=True):
        if message:
            print(f"GPS Data: {message}")
        else:
            print("No GPS data available")

async def main():
    # 1) Load all service configs
    config = proto_from_json_file("service_config.json", EventServiceConfig())
    client = EventClient(config)

    calibration: oak_pb2.OakCalibration = await client.request_reply("/calibration", Empty(), decode=True)
    print("Camera calibration:", calibration)

    # Criar tasks para ambos os streams
    rgb_task = asyncio.create_task(process_rgb_stream(client))
    gps_task = asyncio.create_task(get_gps(client))

    try:
        # Aguardar ambas as tasks (rode indefinidamente até interrupção)
        await asyncio.gather(rgb_task, gps_task)
    except KeyboardInterrupt:
        print("Interrompido pelo usuário")
        # Cancelar as tasks se necessário
        rgb_task.cancel()
        gps_task.cancel()
        
        # Aguardar o cancelamento
        await asyncio.gather(rgb_task, gps_task, return_exceptions=True)

if __name__ == "__main__":
    asyncio.run(main())