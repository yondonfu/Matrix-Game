from fastrtc import WebRTC
from twilio.rest import Client
import cv2
import os
import gradio as gr
import torch
from collections import deque

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

# Global action queue
action_queue = deque()

def read_video():
    cap = cv2.VideoCapture(VIDEO_FILE_PATH)
    iterating = True
    while iterating:
        # Pop the latest action from the queue and use get_action_cond
        if action_queue:
            latest_action = action_queue.pop()
            action_obj = get_action_cond("u", latest_action)
            print(action_obj)
        
        iterating, frame = cap.read()
        yield frame

def handle_move_up():
    action_queue.append("w")
    print("up")

def handle_move_down():
    action_queue.append("s")
    print("down")

def handle_move_left():
    action_queue.append("a")
    print("left")

def handle_move_right():
    action_queue.append("d")
    print("right")

def get_action_cond(idx_mouse, idx_keyboard):
    CAM_VALUE = 0.1
    CAMERA_VALUE_MAP = {
        "i":  [CAM_VALUE, 0],
        "k":  [-CAM_VALUE, 0],
        "j":  [0, -CAM_VALUE],
        "l":  [0, CAM_VALUE],
        "u":  [0, 0]
    }
    KEYBOARD_IDX = { 
        "w": [1, 0, 0, 0], "s": [0, 1, 0, 0], "a": [0, 0, 1, 0], "d": [0, 0, 0, 1],
        "q": [0, 0, 0, 0]
    }

    mouse_cond = torch.tensor(CAMERA_VALUE_MAP[idx_mouse]).cuda()
    keyboard_cond = torch.tensor(KEYBOARD_IDX[idx_keyboard]).cuda()

    return {
        "mouse": mouse_cond,
        "keyboard": keyboard_cond
    }

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            start_button = gr.Button("Start Stream", variant="primary")

            move_up_button = gr.Button("Up", variant="secondary")
            move_up_button.click(fn=handle_move_up)

            move_down_button = gr.Button("Down", variant="secondary")
            move_down_button.click(fn=handle_move_down)

            move_left_button = gr.Button("Left", variant="secondary")
            move_left_button.click(fn=handle_move_left)

            move_right_button = gr.Button("Right", variant="secondary")
            move_right_button.click(fn=handle_move_right)
        with gr.Column():
            video = WebRTC(
                modality="video",
                mode="receive",
                rtc_configuration=rtc_configuration
            )
        video.stream(fn=read_video, inputs=[], outputs=[video], trigger=start_button.click)

demo.launch(server_name="0.0.0.0", server_port=8888)