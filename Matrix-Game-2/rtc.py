from fastrtc import Stream
from twilio.rest import Client
import cv2
import os

account_sid = os.environ.get("TWILIO_ACCOUNT_SID")
auth_token = os.environ.get("TWILIO_AUTH_TOKEN")

client = Client(account_sid, auth_token)

token = client.tokens.create()

rtc_configuration = {
    "iceServers": token.ice_servers,
    "iceTransportPolicy": "relay",
}

# Video file path - update this to your static video file
VIDEO_FILE_PATH = "static_video.mp4"

def read_video():
    cap = cv2.VideoCapture(VIDEO_FILE_PATH)
    iterating = True
    while iterating:
        iterating, frame = cap.read()
        yield frame

stream = Stream(
    handler=read_video,
    rtc_configuration=rtc_configuration,
    modality="video",
    mode="receive",
)

stream.ui.launch(server_name="0.0.0.0", server_port=8000)