from typing import Optional, Generator, Union
import torch
import copy

from einops import rearrange
from utils.wan_wrapper import WanDiffusionWrapper
from demo_utils.constant import ZERO_VAE_CACHE

def cond_current(conditional_dict, current_start_frame, num_frame_per_block, replace=None, mode='universal'):
    
    new_cond = {}
    
    new_cond["cond_concat"] = conditional_dict["cond_concat"][:, :, current_start_frame: current_start_frame + num_frame_per_block]
    new_cond["visual_context"] = conditional_dict["visual_context"]
    if replace != None:
        if current_start_frame == 0:
            last_frame_num = 1 + 4 * (num_frame_per_block - 1)
        else:
            last_frame_num = 4 * num_frame_per_block
        final_frame = 1 + 4 * (current_start_frame + num_frame_per_block-1)
        if mode != 'templerun':
            conditional_dict["mouse_cond"][:, -last_frame_num + final_frame: final_frame] = replace['mouse'][None, None, :].repeat(1, last_frame_num, 1)
        conditional_dict["keyboard_cond"][:, -last_frame_num + final_frame: final_frame] = replace['keyboard'][None, None, :].repeat(1, last_frame_num, 1)
    if mode != 'templerun':
        new_cond["mouse_cond"] = conditional_dict["mouse_cond"][:, : 1 + 4 * (current_start_frame + num_frame_per_block - 1)]
    new_cond["keyboard_cond"] = conditional_dict["keyboard_cond"][:, : 1 + 4 * (current_start_frame + num_frame_per_block - 1)]

    if replace != None:
        return new_cond, conditional_dict
    else:
        return new_cond

class CausalInferenceRTCPipeline(torch.nn.Module):
    def __init__(
        self,
        args,
        device="cuda",
        vae_decoder=None,
        generator=None,
    ):
        super().__init__()
        # Step 1: Initialize all models
        self.generator = WanDiffusionWrapper(
            **getattr(args, "model_kwargs", {}), is_causal=True) if generator is None else generator
        self.vae_decoder = vae_decoder

        # Step 2: Initialize all causal hyperparmeters
        self.scheduler = self.generator.get_scheduler()
        self.denoising_step_list = torch.tensor(
            args.denoising_step_list, dtype=torch.long)
        if args.warp_denoising_step:
            timesteps = torch.cat((self.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32)))
            self.denoising_step_list = timesteps[1000 - self.denoising_step_list]

        self.num_transformer_blocks = 30
        self.frame_seq_length = 880 # 1590 # HW/4

        self.kv_cache1 = None
        self.kv_cache_mouse = None
        self.kv_cache_keyboard = None
        self.args = args
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        self.local_attn_size = self.generator.model.local_attn_size
        assert self.local_attn_size != -1
        print(f"KV inference with {self.num_frame_per_block} frames per block")

        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block

    def inference(
        self,
        noise: torch.Tensor,
        conditional_dict,
        initial_latent: Optional[torch.Tensor] = None,
        return_latents: bool = False,
        mode = 'universal',
        profile = False,
        vae_cache = None,
        get_current_actions = None,
    ) -> Generator[torch.Tensor, None, torch.Tensor]:
        """
        Perform inference on the given noise and text prompts.
        
        This method is a generator that yields decoded video frames as they become available.
        
        Inputs:
            noise (torch.Tensor): The input noise tensor of shape
                (batch_size, num_output_frames, num_channels, height, width).
            conditional_dict: Dictionary containing conditioning information.
            initial_latent (torch.Tensor): The initial latent tensor of shape
                (batch_size, num_input_frames, num_channels, height, width).
                If num_input_frames is 1, perform image to video.
                If num_input_frames is greater than 1, perform video extension.
            return_latents (bool): Whether to return the latents.
        
        Yields:
            video (torch.Tensor): Each decoded video frame block as it becomes available.
        """
        
        assert noise.shape[1] == 16
        batch_size, num_channels, num_frames, height, width = noise.shape
        
        assert num_frames % self.num_frame_per_block == 0
        num_blocks = num_frames // self.num_frame_per_block

        num_input_frames = initial_latent.shape[2] if initial_latent is not None else 0
        num_output_frames = num_frames + num_input_frames  # add the initial latent frames

        output = torch.zeros(
            [batch_size, num_channels, num_output_frames, height, width],
            device=noise.device,
            dtype=noise.dtype
        )
        # Use provided vae_cache or initialize fresh one
        if vae_cache is None:
            vae_cache = copy.deepcopy(ZERO_VAE_CACHE)
            for j in range(len(vae_cache)):
                vae_cache[j] = None
            print("Using fresh VAE cache")
        else:
            print("Using provided warmed VAE cache")
        # Set up profiling if requested
        self.kv_cache1=self.kv_cache_keyboard=self.kv_cache_mouse=self.crossattn_cache=None
        # Step 1: Initialize KV cache to all zeros
        if self.kv_cache1 is None:
            self._initialize_kv_cache(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device
            )
            self._initialize_kv_cache_mouse_and_keyboard(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device
            )
            
            self._initialize_crossattn_cache(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device
            )
        else:
            # reset cross attn cache
            for block_index in range(self.num_transformer_blocks):
                self.crossattn_cache[block_index]["is_init"] = False
            # reset kv cache
            for block_index in range(len(self.kv_cache1)):
                self.kv_cache1[block_index]["global_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)
                self.kv_cache1[block_index]["local_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)
                self.kv_cache_mouse[block_index]["global_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)
                self.kv_cache_mouse[block_index]["local_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)
                self.kv_cache_keyboard[block_index]["global_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)
                self.kv_cache_keyboard[block_index]["local_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)
        # Step 2: Cache context feature
        current_start_frame = 0
        if initial_latent is not None:
            timestep = torch.ones([batch_size, 1], device=noise.device, dtype=torch.int64) * 0
            
            # Assume num_input_frames is self.num_frame_per_block * num_input_blocks
            assert num_input_frames % self.num_frame_per_block == 0
            num_input_blocks = num_input_frames // self.num_frame_per_block

            for _ in range(num_input_blocks):
                current_ref_latents = \
                    initial_latent[:, :, current_start_frame:current_start_frame + self.num_frame_per_block]
                output[:, :, current_start_frame:current_start_frame + self.num_frame_per_block] = current_ref_latents
                self.generator(
                    noisy_image_or_video=current_ref_latents,
                    conditional_dict=cond_current(conditional_dict, current_start_frame, self.num_frame_per_block, replace=True),
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache1,
                    kv_cache_mouse=self.kv_cache_mouse,
                    kv_cache_keyboard=self.kv_cache_keyboard,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                )
                current_start_frame += self.num_frame_per_block

        # Step 3: Temporal denoising loop
        all_num_frames = [self.num_frame_per_block] * num_blocks
        
        last_video = None
        for current_num_frames in all_num_frames:
            current_actions = get_current_actions() if get_current_actions is not None else None
            new_act, _ = cond_current(conditional_dict, current_start_frame, self.num_frame_per_block, replace=current_actions, mode=mode)
            
            # Check if we have action input based on current_actions
            has_current_action = current_start_frame == 0 or self._has_action_input(new_act, mode, current_start_frame)
            if has_current_action:
                # Generate new block
                denoised_pred, video, vae_cache = self._generate_block(
                    noise, new_act, current_start_frame, current_num_frames,
                    num_input_frames, vae_cache, batch_size, mode, profile
                )
                output[:, :, current_start_frame:current_start_frame + current_num_frames] = denoised_pred
                last_video = video
            else:
                print("No action conditioning detected - reusing last static frame")
                
                # Ensure the video has the correct shape by taking last frame from previous video and repeating
                # We do not update the KV or VAE cache here because we pretend that this block was not generated
                if last_video is not None:
                    if hasattr(last_video, 'shape') and len(last_video.shape) >= 2:
                        # Take the last frame from the previous video
                        last_frame = last_video[:, -1:, ...]  # Shape: [B, 1, C, H, W] 
                        # Repeat it for the same number of frames as the previous video
                        num_video_frames = last_video.shape[1]
                        static_video = last_frame.repeat(1, num_video_frames, *([1] * (len(last_frame.shape) - 2)))
                        video = static_video
                    else:
                        video = last_video
                else:
                    video = last_video

                yield video
    
                # Continue without updating current_start_frame to pretend that this block was not generated yet
                continue

            current_start_frame += current_num_frames
            last_video = video
            
            # Yield each frame as it becomes available
            yield video
            
        # Return final output after all frames have been yielded
        if return_latents:
            return output

    def _initialize_kv_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU KV cache for the Wan model.
        """
        kv_cache1 = []
        if self.local_attn_size != -1:
            # Use the local attention size to compute the KV cache size
            kv_cache_size = self.local_attn_size * self.frame_seq_length
        else:
            # Use the default KV cache size
            kv_cache_size = 15 * 1 * self.frame_seq_length # 32760

        for _ in range(self.num_transformer_blocks):
            kv_cache1.append({
                "k": torch.zeros([batch_size, kv_cache_size, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, kv_cache_size, 12, 128], dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device)
            })

        self.kv_cache1 = kv_cache1  # always store the clean cache

    def _initialize_kv_cache_mouse_and_keyboard(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU KV cache for the Wan model.
        """
        kv_cache_mouse = []
        kv_cache_keyboard = []
        if self.local_attn_size != -1:
            kv_cache_size = self.local_attn_size
        else:
            kv_cache_size = 15 * 1
        for _ in range(self.num_transformer_blocks):
            kv_cache_keyboard.append({
                "k": torch.zeros([batch_size, kv_cache_size, 16, 64], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, kv_cache_size, 16, 64], dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device)
            })
            kv_cache_mouse.append({
                "k": torch.zeros([batch_size * self.frame_seq_length, kv_cache_size, 16, 64], dtype=dtype, device=device),
                "v": torch.zeros([batch_size * self.frame_seq_length, kv_cache_size, 16, 64], dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device)
            })
        self.kv_cache_keyboard = kv_cache_keyboard  # always store the clean cache
        self.kv_cache_mouse = kv_cache_mouse  # always store the clean cache

        

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU cross-attention cache for the Wan model.
        """
        crossattn_cache = []

        for _ in range(self.num_transformer_blocks):
            crossattn_cache.append({
                "k": torch.zeros([batch_size, 257, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, 257, 12, 128], dtype=dtype, device=device),
                "is_init": False
            })
        self.crossattn_cache = crossattn_cache

    def _has_action_input(self, conditional_dict, mode='universal', current_start_frame=0):
        """Check if current block has any keyboard/mouse input (non-zero values)"""
        # Extract only the conditioning for the current block's new frames
        # cond_current gives us all frames from start to current position
        # We need to check only the new frames added for this block
        
        if current_start_frame == 0:
            # First block, check from start
            if mode != 'templerun':
                current_mouse = conditional_dict["mouse_cond"][:, :1 + 4 * (self.num_frame_per_block - 1)]
            current_keyboard = conditional_dict["keyboard_cond"][:, :1 + 4 * (self.num_frame_per_block - 1)]
        else:
            # Subsequent blocks, check only the new frames added
            prev_frame_end = 1 + 4 * (current_start_frame - 1)
            curr_frame_end = 1 + 4 * (current_start_frame + self.num_frame_per_block - 1)
            
            if mode != 'templerun':
                current_mouse = conditional_dict["mouse_cond"][:, prev_frame_end:curr_frame_end]
            current_keyboard = conditional_dict["keyboard_cond"][:, prev_frame_end:curr_frame_end]
        
        if mode != 'templerun':
            mouse_input = current_mouse.abs().sum() > 0
        else:
            mouse_input = False
        
        keyboard_input = current_keyboard.abs().sum() > 0
        return mouse_input or keyboard_input

    def _generate_block(self, noise, conditional_dict, current_start_frame, current_num_frames, num_input_frames, vae_cache, batch_size, mode='universal', profile=False):
        """Generate a single block - extracted from main loop to avoid duplication"""
        
        noisy_input = noise[
            :, :, current_start_frame - num_input_frames:current_start_frame + current_num_frames - num_input_frames]

        # Step 3.1: Spatial denoising loop
        if profile:
            diffusion_start = torch.cuda.Event(enable_timing=True)
            diffusion_end = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            diffusion_start.record()
            
        for index, current_timestep in enumerate(self.denoising_step_list):
            # set current timestep
            timestep = torch.ones(
                [batch_size, current_num_frames],
                device=noise.device,
                dtype=torch.int64) * current_timestep

            if index < len(self.denoising_step_list) - 1:
                _, denoised_pred = self.generator(
                    noisy_image_or_video=noisy_input,
                    conditional_dict=conditional_dict,
                    timestep=timestep,
                    kv_cache=self.kv_cache1,
                    kv_cache_mouse=self.kv_cache_mouse,
                    kv_cache_keyboard=self.kv_cache_keyboard,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length
                )
                next_timestep = self.denoising_step_list[index + 1]
                noisy_input = self.scheduler.add_noise(
                    rearrange(denoised_pred, 'b c f h w -> (b f) c h w'),
                    torch.randn_like(rearrange(denoised_pred, 'b c f h w -> (b f) c h w')),
                    next_timestep * torch.ones(
                        [batch_size * current_num_frames], device=noise.device, dtype=torch.long)
                )
                noisy_input = rearrange(noisy_input, '(b f) c h w -> b c f h w', b=denoised_pred.shape[0])
            else:
                # for getting real output
                _, denoised_pred = self.generator(
                    noisy_image_or_video=noisy_input,
                    conditional_dict=conditional_dict,
                    timestep=timestep,
                    kv_cache=self.kv_cache1,
                    kv_cache_mouse=self.kv_cache_mouse,
                    kv_cache_keyboard=self.kv_cache_keyboard,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length
                )

        # Step 3.3: rerun with timestep zero to update KV cache using clean context
        context_timestep = torch.ones_like(timestep) * self.args.context_noise
        
        self.generator(
            noisy_image_or_video=denoised_pred,
            conditional_dict=conditional_dict,
            timestep=context_timestep,
            kv_cache=self.kv_cache1,
            kv_cache_mouse=self.kv_cache_mouse,
            kv_cache_keyboard=self.kv_cache_keyboard,
            crossattn_cache=self.crossattn_cache,
            current_start=current_start_frame * self.frame_seq_length,
        )

        # VAE decoding
        denoised_pred_for_vae = denoised_pred.transpose(1,2)
        video, vae_cache = self.vae_decoder(denoised_pred_for_vae.half(), *vae_cache)
        
        if profile:
            torch.cuda.synchronize()
            diffusion_end.record()
            diffusion_time = diffusion_start.elapsed_time(diffusion_end)
            print(f"diffusion_time: {diffusion_time}", flush=True)
            fps = video.shape[1]*1000/ diffusion_time
            print(f"  - FPS: {fps:.2f}")

        return denoised_pred, video, vae_cache
