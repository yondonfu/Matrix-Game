import os
import argparse
import torch
import copy
import numpy as np
from typing import Generator
from collections import deque
from einops import rearrange
import gradio as gr

from omegaconf import OmegaConf
from torchvision.transforms import v2
from diffusers.utils import load_image

from pipeline import CausalInferenceRTCPipeline
from wan.vae.wanx_vae import get_wanx_vae_wrapper
from demo_utils.vae_block3 import VAEDecoderWrapper
from demo_utils.constant import ZERO_VAE_CACHE
from utils.conditions import *
from utils.misc import set_seed
from utils.wan_wrapper import WanDiffusionWrapper
from safetensors.torch import load_file

from fastrtc import WebRTC
from twilio.rest import Client

# Global action queue
move_queue = deque()
camera_queue = deque()

def handle_move_up():
    move_queue.append("w")
    print("move up")

def handle_move_down():
    move_queue.append("s")
    print("move down")

def handle_move_left():
    move_queue.append("a")
    print("move left")

def handle_move_right():
    move_queue.append("d")
    print("move right")

def handle_camera_up():
    camera_queue.append("i")
    print("camera up")

def handle_camera_down():
    camera_queue.append("k")
    print("camera down")

def handle_camera_left():
    camera_queue.append("j")
    print("camera left")

def handle_camera_right():
    camera_queue.append("l")
    print("camera right")

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

def get_current_actions():
    idx_mouse = "u"
    if camera_queue:
        idx_mouse = camera_queue.pop()

    idx_keyboard = "q"
    if move_queue:
        idx_keyboard = move_queue.pop()

    return get_action_cond(idx_mouse, idx_keyboard)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="configs/inference_yaml/inference_universal.yaml", help="Path to the config file")
    parser.add_argument("--checkpoint_path", type=str, default="", help="Path to the checkpoint")
    parser.add_argument("--img_path", type=str, default="demo_images/universal/0000.png", help="Path to input image")
    parser.add_argument("--max_num_output_frames", type=int, default=360,
                        help="Max number of output latent frames")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--pretrained_model_path", type=str, default="Matrix-Game-2.0", help="Path to the VAE model folder")
    parser.add_argument("--profile", action="store_true", help="Enable performance profiling")
    parser.add_argument("--vae-warmup", action="store_true", help="Enable VAE decoder torch.compile warmup")
    args = parser.parse_args()
    return args

class InteractiveGameInference:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda")
        self.weight_dtype = torch.bfloat16

        self._init_config()
        self._init_models()

        self.frame_process = v2.Compose([
            v2.Resize(size=(352, 640), antialias=True),
            v2.ToTensor(),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        self._init_input(args.img_path)

    def _init_config(self):
        self.config = OmegaConf.load(self.args.config_path)

    def _init_models(self):
        # Initialize pipeline
        generator = WanDiffusionWrapper(
            **getattr(self.config, "model_kwargs", {}), is_causal=True)
        current_vae_decoder = VAEDecoderWrapper()
        vae_state_dict = torch.load(os.path.join(self.args.pretrained_model_path, "Wan2.1_VAE.pth"), map_location="cpu")
        decoder_state_dict = {}
        for key, value in vae_state_dict.items():
            if 'decoder.' in key or 'conv2' in key:
                decoder_state_dict[key] = value
        current_vae_decoder.load_state_dict(decoder_state_dict)
        current_vae_decoder.to(self.device, torch.float16)
        current_vae_decoder.requires_grad_(False)
        current_vae_decoder.eval()
        # current_vae_decoder.compile(mode="max-autotune-no-cudagraphs")
        
        # Conditionally warm up the compiled VAE decoder
        if self.args.vae_warmup:
            print("Warming up VAE decoder torch.compile...")
            # Get runtime dimensions from config
            num_frame_per_block = getattr(self.config, "num_frame_per_block", 1)
            batch_size, num_channels, _, height, width = self.config.image_or_video_shape
            
            # Create dummy tensor matching exact runtime dimensions
            dummy_latent = torch.zeros([batch_size, num_frame_per_block, num_channels, height, width], 
                                     dtype=self.weight_dtype, device=self.device)
            print(f"Using dummy tensor dimensions: [{batch_size}, {num_frame_per_block}, {num_channels}, {height}, {width}] with dtype conversion {self.weight_dtype} -> .half()")
            
            dummy_cache = copy.deepcopy(ZERO_VAE_CACHE)
            for j in range(len(dummy_cache)):
                dummy_cache[j] = None
                
            with torch.no_grad():
                # Match exact runtime call pattern and capture warmed cache
                _, vae_cache = current_vae_decoder(dummy_latent.half(), *dummy_cache)
                    
            print("VAE decoder torch.compile warmup completed.")
            print("Captured warmed VAE cache for runtime use.")
            
            # Store warmed cache for later use
            self.vae_cache = vae_cache
        else:
            print("VAE decoder torch.compile warmup disabled.")
            self.vae_cache = None
        
        pipeline = CausalInferenceRTCPipeline(self.config, generator=generator, vae_decoder=current_vae_decoder)
        if self.args.checkpoint_path:
            print("Loading Pretrained Model...")
            state_dict = load_file(self.args.checkpoint_path)
            pipeline.generator.load_state_dict(state_dict)

        self.pipeline = pipeline.to(device=self.device, dtype=self.weight_dtype)
        self.pipeline.vae_decoder.to(torch.float16)

        vae = get_wanx_vae_wrapper(self.args.pretrained_model_path, torch.float16)
        vae.requires_grad_(False)
        vae.eval()
        self.vae = vae.to(self.device, self.weight_dtype)

    def _init_input(self, img_path):
        mode = self.config.mode
        
        image = load_image(img_path.strip())
        image = self._resizecrop(image, 352, 640)
        image = self.frame_process(image)[None, :, None, :, :].to(dtype=self.weight_dtype, device=self.device)
        # Encode the input image as the first latent
        padding_video = torch.zeros_like(image).repeat(1, 1, 4 * (self.args.max_num_output_frames - 1), 1, 1)
        img_cond = torch.concat([image, padding_video], dim=2)
        tiler_kwargs={"tiled": True, "tile_size": [44, 80], "tile_stride": [23, 38]}
        img_cond = self.vae.encode(img_cond, device=self.device, **tiler_kwargs).to(self.device)
        mask_cond = torch.ones_like(img_cond)
        mask_cond[:, :, 1:] = 0
        cond_concat = torch.cat([mask_cond[:, :4], img_cond], dim=1) 
        visual_context = self.vae.clip.encode_video(image)
        sampled_noise = torch.randn(
            [1, 16,self.args.max_num_output_frames, 44, 80], device=self.device, dtype=self.weight_dtype
        )
        num_frames = (self.args.max_num_output_frames - 1) * 4 + 1
        
        conditional_dict = {
            "cond_concat": cond_concat.to(device=self.device, dtype=self.weight_dtype),
            "visual_context": visual_context.to(device=self.device, dtype=self.weight_dtype)
        }
        
        if mode == 'universal':
            cond_data = Bench_actions_universal(num_frames)
            mouse_condition = cond_data['mouse_condition'].unsqueeze(0).to(device=self.device, dtype=self.weight_dtype)
            conditional_dict['mouse_cond'] = mouse_condition
        elif mode == 'gta_drive':
            cond_data = Bench_actions_gta_drive(num_frames)
            mouse_condition = cond_data['mouse_condition'].unsqueeze(0).to(device=self.device, dtype=self.weight_dtype)
            conditional_dict['mouse_cond'] = mouse_condition
        else:
            cond_data = Bench_actions_templerun(num_frames)

        keyboard_condition = cond_data['keyboard_condition'].unsqueeze(0).to(device=self.device, dtype=self.weight_dtype)
        conditional_dict['keyboard_cond'] = keyboard_condition

        self.sampled_noise = sampled_noise
        self.conditional_dict = conditional_dict

    def _resizecrop(self, image, th, tw):
        w, h = image.size
        if h / w > th / tw:
            new_w = int(w)
            new_h = int(new_w * th / tw)
        else:
            new_h = int(h)
            new_w = int(new_h * tw / th)
        left = (w - new_w) / 2
        top = (h - new_h) / 2
        right = (w + new_w) / 2
        bottom = (h + new_h) / 2
        image = image.crop((left, top, right, bottom))
        return image
    
    def stream_frames(self) -> Generator[np.ndarray, None, None]:
        mode = self.config.mode
        try:
            with torch.no_grad():
                for tensor_batch in self.pipeline.inference(
                    noise=self.sampled_noise,
                    conditional_dict=self.conditional_dict,
                    return_latents=False,
                    mode=mode,
                    profile=self.args.profile,
                    vae_cache=self.vae_cache,
                    get_current_actions=get_current_actions
                ):
                    # tensor_batch shape: (b, f, c, h, w)
                    _, num_frames, _, _, _ = tensor_batch.shape
                    
                    tensor_batch = rearrange(tensor_batch, "B T C H W -> B T H W C")
                    # Iterate through each frame in the batch
                    for frame_idx in range(num_frames):
                        # Extract single frame
                        frame = tensor_batch[0, frame_idx]  # (h, w, c)

                        # Convert from normalized range to uint8 (0-255)
                        # Assuming your frames are in range [-1, 1]
                        frame_numpy = ((frame.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)

                        # Convert from RGB to BGR because fastrtc expects a BGR np.ndarray
                        # https://github.com/gradio-app/fastrtc/blob/ce11e4d6f4e10bc7b3a8f879b7f0473b9d64dcb3/backend/fastrtc/tracks.py#L131
                        frame_numpy = frame_numpy[..., ::-1]
                        
                        yield frame_numpy
                        
        except Exception as e:
            print(f"Error during frame processing: {e}")

def main():
    """Main entry point for video generation."""
    args = parse_args()
    set_seed(args.seed)
    pipeline = InteractiveGameInference(args)

    account_sid = os.environ.get("TWILIO_ACCOUNT_SID")
    auth_token = os.environ.get("TWILIO_AUTH_TOKEN")

    client = Client(account_sid, auth_token)

    token = client.tokens.create()

    rtc_configuration = {
        "iceServers": token.ice_servers,
        "iceTransportPolicy": "relay",
    }

    # Create a wrapper function for the handler
    def frame_handler():
        return pipeline.stream_frames()
    
    # Create Gradio interface with buttons
    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column():
                start_button = gr.Button("Start Stream", variant="primary")

                move_up_button = gr.Button("🚶 Up", variant="secondary")
                move_up_button.click(fn=handle_move_up)

                move_down_button = gr.Button("🚶 Down", variant="secondary")
                move_down_button.click(fn=handle_move_down)

                move_left_button = gr.Button("🚶 Left", variant="secondary")
                move_left_button.click(fn=handle_move_left)

                move_right_button = gr.Button("🚶 Right", variant="secondary")
                move_right_button.click(fn=handle_move_right)

                camera_up_button = gr.Button("🎥 Up", variant="secondary")
                camera_up_button.click(fn=handle_camera_up)

                camera_down_button = gr.Button("🎥 Down", variant="secondary")
                camera_down_button.click(fn=handle_camera_down)

                camera_left_button = gr.Button("🎥 Left", variant="secondary")
                camera_left_button.click(fn=handle_camera_left)

                camera_right_button = gr.Button("🎥 Right", variant="secondary")
                camera_right_button.click(fn=handle_camera_right)

            with gr.Column():
                video = WebRTC(
                    modality="video",
                    mode="receive",
                    rtc_configuration=rtc_configuration
                )
            video.stream(fn=frame_handler, inputs=[], outputs=[video], trigger=start_button.click)

    demo.launch(server_name="0.0.0.0", server_port=8888)

if __name__ == "__main__":
    main()

