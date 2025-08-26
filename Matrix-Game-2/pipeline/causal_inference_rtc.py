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
        
        # Initialize profiling events if needed
        if profile:
            diffusion_start = torch.cuda.Event(enable_timing=True)
            diffusion_end = torch.cuda.Event(enable_timing=True)
            vae_start = torch.cuda.Event(enable_timing=True)
            vae_end = torch.cuda.Event(enable_timing=True)
            total_diffusion_time = 0.0
            total_vae_time = 0.0
            block_count = 0
        
        for current_num_frames in all_num_frames:
            noisy_input = noise[
                :, :, current_start_frame - num_input_frames:current_start_frame + current_num_frames - num_input_frames]

            current_actions = get_current_actions() if get_current_actions is not None else None
            new_act, conditional_dict = cond_current(conditional_dict, current_start_frame, self.num_frame_per_block, replace=current_actions, mode=mode)
            
            # Step 3.1: Spatial denoising loop
            if profile:
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
                        conditional_dict=new_act,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        kv_cache_mouse=self.kv_cache_mouse,
                        kv_cache_keyboard=self.kv_cache_keyboard,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length
                    )
                    next_timestep = self.denoising_step_list[index + 1]
                    noisy_input = self.scheduler.add_noise(
                        rearrange(denoised_pred, 'b c f h w -> (b f) c h w'),# .flatten(0, 1),
                        torch.randn_like(rearrange(denoised_pred, 'b c f h w -> (b f) c h w')),
                        next_timestep * torch.ones(
                            [batch_size * current_num_frames], device=noise.device, dtype=torch.long)
                    )
                    noisy_input = rearrange(noisy_input, '(b f) c h w -> b c f h w', b=denoised_pred.shape[0])
                else:
                    # for getting real output
                    _, denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=new_act,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        kv_cache_mouse=self.kv_cache_mouse,
                        kv_cache_keyboard=self.kv_cache_keyboard,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length
                    )

            # Step 3.2: record the model's output
            output[:, :, current_start_frame:current_start_frame + current_num_frames] = denoised_pred

            # Step 3.3: rerun with timestep zero to update KV cache using clean context
            context_timestep = torch.ones_like(timestep) * self.args.context_noise
            
            self.generator(
                noisy_image_or_video=denoised_pred,
                conditional_dict=new_act,
                timestep=context_timestep,
                kv_cache=self.kv_cache1,
                kv_cache_mouse=self.kv_cache_mouse,
                kv_cache_keyboard=self.kv_cache_keyboard,
                crossattn_cache=self.crossattn_cache,
                current_start=current_start_frame * self.frame_seq_length,
            )

            # End diffusion timing
            if profile:
                torch.cuda.synchronize()
                diffusion_end.record()

            # Step 3.4: update the start and end frame indices and decode with VAE
            current_start_frame += current_num_frames

            if profile:
                torch.cuda.synchronize()
                vae_start.record()
                
            denoised_pred = denoised_pred.transpose(1,2)
            # print(f"Runtime VAE decoder input dims: {list(denoised_pred.shape)} with dtype {denoised_pred.dtype} -> .half()")
            video, vae_cache = self.vae_decoder(denoised_pred.half(), *vae_cache)

            # End VAE timing immediately after decoder
            if profile:
                torch.cuda.synchronize()
                vae_end.record()
            
            # Calculate metrics
            if profile:
                # Calculate times for this block
                diffusion_time = diffusion_start.elapsed_time(diffusion_end)
                vae_time = vae_start.elapsed_time(vae_end)
                total_time = diffusion_time + vae_time
                
                # Update running totals
                total_diffusion_time += diffusion_time
                total_vae_time += vae_time
                block_count += 1
                
                # Calculate FPS metrics
                frames_generated = video.shape[1]  # Number of frames in this block
                diffusion_fps = frames_generated * 1000 / diffusion_time
                vae_fps = frames_generated * 1000 / vae_time  
                total_fps = frames_generated * 1000 / total_time
                
                # Print per-block breakdown
                print(f"\n=== Block {block_count} Performance Metrics ===")
                print(f"Frames generated: {frames_generated}")
                print(f"Diffusion time: {diffusion_time:.2f}ms ({diffusion_fps:.2f} FPS)")
                print(f"VAE decode time: {vae_time:.2f}ms ({vae_fps:.2f} FPS)")  
                print(f"Total time: {total_time:.2f}ms ({total_fps:.2f} FPS)")
                print(f"Time breakdown - Diffusion: {diffusion_time/total_time*100:.1f}% | VAE: {vae_time/total_time*100:.1f}%")
                
                # Print cumulative averages
                avg_diffusion_time = total_diffusion_time / block_count
                avg_vae_time = total_vae_time / block_count
                avg_total_time = avg_diffusion_time + avg_vae_time
                avg_diffusion_fps = frames_generated * 1000 / avg_diffusion_time
                avg_vae_fps = frames_generated * 1000 / avg_vae_time
                avg_total_fps = frames_generated * 1000 / avg_total_time
                
                print(f"\n--- Cumulative Averages (over {block_count} blocks) ---")
                print(f"Avg diffusion time: {avg_diffusion_time:.2f}ms ({avg_diffusion_fps:.2f} FPS)")
                print(f"Avg VAE time: {avg_vae_time:.2f}ms ({avg_vae_fps:.2f} FPS)")
                print(f"Avg total time: {avg_total_time:.2f}ms ({avg_total_fps:.2f} FPS)")
                print(f"Avg breakdown - Diffusion: {avg_diffusion_time/avg_total_time*100:.1f}% | VAE: {avg_vae_time/avg_total_time*100:.1f}%")
                print("=" * 50)
            
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
