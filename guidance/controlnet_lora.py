from . import BaseGuidance
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, CLIPTextModel
from diffusers import (
	DDIMScheduler,
	DDPMScheduler,
	StableDiffusionPipeline,
	StableDiffusionControlNetImg2ImgPipeline,
	PNDMScheduler,
	ControlNetModel
)
from diffusers.image_processor import VaeImageProcessor
from diffusers.utils.import_utils import is_xformers_available

from utils.typing import *
from utils.ops import perpendicular_component
from utils.misc import C
from rich.console import Console

from PIL import Image

from guidance.control_lora import ControlLoRA
console = Console()

class StableDiffusionGuidance(BaseGuidance):
	def __init__(self, cfg) -> None:
		super().__init__(cfg)
		self.weights_dtype = (
			torch.float16 if self.cfg.half_precision_weights else torch.float32
		)

		if self.cfg.keep_complete_pipeline:
			pipe_kwargs = {
				"torch_dtype": self.weights_dtype,
			}
		else:
			pipe_kwargs = {
				"tokenizer": None,
				"safety_checker": None,
				"feature_extractor": None,
				"requires_safety_checker": False,
				"torch_dtype": self.weights_dtype,
				"cache_dir": "./.cache",
			}
			pipe_lora_kwargs = {
				"tokenizer": None,
				"safety_checker": None,
				"feature_extractor": None,
				"requires_safety_checker": False,
				"torch_dtype": self.weights_dtype,
				"cache_dir": "./.cache",
			}

		if self.cfg.repeat_until_success:
			success = False
			while not success:
				try:
					if self.cfg.controlled:
						controlnet = ControlNetModel.from_pretrained(
							self.cfg.controlnet_model_name_or_path,
							torch_dtype=torch.float16
						)
						self.pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
							self.cfg.pretrained_model_name_or_path,
							controlnet=controlnet,
							**pipe_kwargs
						).to(self.device)
						self.pipe_lora = StableDiffusionPipeline.from_pretrained(
							self.cfg.pretrained_model_name_or_path,
							**pipe_kwargs
						).to(self.device)
						del self.pipe_lora.vae
						self.pipe_lora.vae = self.pipe.vae
					else:
						self.pipe = StableDiffusionPipeline.from_pretrained(
							self.cfg.pretrained_model_name_or_path,
							**pipe_lora_kwargs
						).to(self.device)

				except KeyboardInterrupt:
					raise KeyboardInterrupt
				except:
					console.print(".", end="")
				else:
					success = True
					break
		else:
			if self.cfg.controlled:
				controlnet = ControlNetModel.from_pretrained(
					self.cfg.controlnet_model_name_or_path,
					torch_dtype=torch.float16
				)
				self.pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
					self.cfg.pretrained_model_name_or_path,
					controlnet=controlnet,
					**pipe_kwargs
				).to(self.device)
				self.pipe_lora = StableDiffusionPipeline.from_pretrained(
							self.cfg.pretrained_model_name_or_path,
							**pipe_lora_kwargs
						).to(self.device)
			else:
				self.pipe = StableDiffusionPipeline.from_pretrained(
					self.cfg.pretrained_model_name_or_path,
					**pipe_kwargs
				).to(self.device)
	
		n_ch = len(self.pipe_lora.unet.config.block_out_channels)
		control_ids = [i for i in range(n_ch)]
		cross_attention_dims = {i: [] for i in range(n_ch)}
		for name in self.pipe_lora.unet.attn_processors.keys():
			cross_attention_dim = None if name.endswith("attn1.processor") else self.pipe_lora.unet.config.cross_attention_dim
			if name.startswith("mid_block"):
				control_id = control_ids[-1]
			elif name.startswith("up_blocks"):
				block_id = int(name[len("up_blocks.")])
				control_id = list(reversed(control_ids))[block_id]
			elif name.startswith("down_blocks"):
				block_id = int(name[len("down_blocks.")])
				control_id = control_ids[block_id]
			cross_attention_dims[control_id].append(cross_attention_dim)
		cross_attention_dims = tuple([cross_attention_dims[control_id] for control_id in control_ids])

		self.control_lora = ControlLoRA.from_config("./conf/control-lora.yaml")

		self.vae = self.pipe.vae.to(self.device, torch.float16)
		self.vae_lora = self.vae
		self.unet = self.pipe.unet
		self.unet_lora = self.pipe_lora.unet
		if self.cfg.controlled:
			self.controlnet = self.pipe.controlnet

		for p in self.vae.parameters():
			p.requires_grad_(False)
		for p in self.unet.parameters():
			p.requires_grad_(False)
		if self.cfg.controlled:
			for p in self.controlnet.parameters():
				p.requires_grad_(False)
		for p in self.unet_lora.parameters():
			p.requires_grad_(False)


		# Set correct lora layers
		lora_attn_procs = {}
		lora_layers_list = list([list(layer_list) for layer_list in self.control_lora.lora_layers])
		for name in self.unet_lora.attn_processors.keys():
			cross_attention_dim = None if name.endswith("attn1.processor") else self.unet_lora.config.cross_attention_dim
			if name.startswith("mid_block"):
				control_id = control_ids[-1]
			elif name.startswith("up_blocks"):
				block_id = int(name[len("up_blocks.")])
				control_id = list(reversed(control_ids))[block_id]
			elif name.startswith("down_blocks"):
				block_id = int(name[len("down_blocks.")])
				control_id = control_ids[block_id]

			lora_layers = lora_layers_list[control_id]
			if len(lora_layers) != 0:
				lora_layer = lora_layers.pop(0)
				lora_attn_procs[name] = lora_layer

		self.unet_lora.set_attn_processor(lora_attn_procs)

		# if self.cfg.controlled:
			# self.controlnet = self.pipe.controlnet.eval()
			# self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
			# self.control_image_processor = VaeImageProcessor(
			# 	vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True, do_normalize=False
			# )
			# self.render_image_processor = VaeImageProcessor(
			# 	vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True, do_normalize=False
			# )
		
		# TODO: make this configurable
		scheduler = self.cfg.scheduler.type.lower()
		if scheduler == "ddim":
			self.scheduler = DDIMScheduler.from_pretrained(
				self.cfg.pretrained_model_name_or_path,
				subfolder="scheduler",
				torch_dtype=self.weights_dtype,
			)
			self.scheduler_lora = DDIMScheduler.from_pretrained(
				self.cfg.pretrained_model_name_or_path_lora,
				subfolder="scheduler",
				torch_dtype=self.weights_dtype,
			)
			self.pipe.scheduler = self.scheduler
			self.pipe_lora.scheduler = self.scheduler
		elif scheduler == "pndm":
			self.scheduler = PNDMScheduler(**self.cfg.scheduler.args)
		else:
			raise NotImplementedError(f"Scheduler {scheduler} not implemented")

		self.num_train_timesteps = self.scheduler.config.num_train_timesteps
		self.step = 0
		self.max_steps = self.cfg.max_steps
		self.set_min_max_steps()
		self.lora_scale = 0.5
		self.grad_clip_val = None
		self.alphas = self.scheduler.alphas_cumprod.to(self.device)
		if self.cfg.enable_attention_slicing:
			# enable GPU VRAM saving, reference: https://huggingface.co/stabilityai/stable-diffusion-2
			self.pipe.enable_attention_slicing(1)

	@torch.cuda.amp.autocast(enabled=False)
	def set_min_max_steps(self, min_step_percent=0.02, max_step_percent=0.98, lora_scale=0.5, condition_scale=1.5):
		self.min_t_step = int(self.num_train_timesteps * min_step_percent)
		self.max_t_step = int(self.num_train_timesteps * max_step_percent)
		if lora_scale is not None:
			self.lora_scale = lora_scale
		if condition_scale is not None:
			self.condition_scale = condition_scale

	@torch.cuda.amp.autocast(enabled=False)
	def forward_unet(
		self,
		latents,
		control_image,
		t,
		encoder_hidden_states,
		condition_scale
	):
		input_dtype = latents.dtype
		if self.cfg.controlled:
			controlnet_prompt_embeds = encoder_hidden_states.to(self.weights_dtype)

			down_block_res_samples, mid_block_res_samples = self.controlnet(
				latents.to(self.weights_dtype),
				t.to(self.weights_dtype),
				encoder_hidden_states=controlnet_prompt_embeds,
				controlnet_cond=control_image,
				conditioning_scale=condition_scale,
				guess_mode=False,
				return_dict=False
			)

			return self.unet(
				latents.to(self.weights_dtype),
				t.to(self.weights_dtype),
				encoder_hidden_states=encoder_hidden_states.to(self.weights_dtype),
				down_block_additional_residuals=down_block_res_samples,
				mid_block_additional_residual=mid_block_res_samples,
			).sample.to(input_dtype)
		else:
			return self.unet(
				latents.to(self.weights_dtype),
				t.to(self.weights_dtype),
				encoder_hidden_states=encoder_hidden_states.to(self.weights_dtype)
			).sample.to(input_dtype)

	@torch.cuda.amp.autocast(enabled=False)
	def forward_unet_lora(
		self,
		latents,
		t,
		image_cond,
		encoder_hidden_states,
	):
		input_dtype = latents.dtype
		_ = self.control_lora(image_cond).control_states
		return self.unet_lora(
			latents.to(self.weights_dtype),
			t.to(self.weights_dtype),
			encoder_hidden_states=encoder_hidden_states.to(self.weights_dtype),
		).sample.to(input_dtype)

	@torch.cuda.amp.autocast(enabled=False)
	def encode_images(self, imgs, geo_cond):
		input_dtype = imgs.dtype
		imgs = imgs * 2.0 - 1.0
		posterior = self.vae.encode(imgs.to(self.weights_dtype)).latent_dist
		latents = posterior.sample() * self.vae.config.scaling_factor

		geo_cond = geo_cond * 2.0 - 1.0 if geo_cond is not None else imgs
		posterior_lora = self.vae_lora.encode(geo_cond.to(self.weights_dtype)).latent_dist
		latents_lora = posterior_lora.sample() * self.vae_lora.config.scaling_factor 
		return latents.to(input_dtype), latents_lora.to(input_dtype)

	@torch.cuda.amp.autocast(enabled=False)
	def decode_latents(
		self,
		latents,
		latent_height: int = 64,
		latent_width: int = 64,
	):
		input_dtype = latents.dtype
		latents = F.interpolate(
			latents, (latent_height, latent_width), mode="bilinear", align_corners=False
		)
		latents = 1 / self.vae.config.scaling_factor * latents
		image = self.vae.decode(latents.to(self.weights_dtype)).sample
		image = (image * 0.5 + 0.5).clamp(0, 1)
		return image.to(input_dtype)

	def compute_grad_sds(
		self,
		latents,
		latents_lora,
		control_image,
		t,
		prompt_embedding,
		elevation,
		azimuth,
		camera_distances,
	):
		batch_size = elevation.shape[0]

		if prompt_embedding.use_perp_negative:
			(
				text_embeddings,
				neg_guidance_weights,
			) = prompt_embedding.get_text_embeddings_perp_neg(
				elevation, azimuth, camera_distances, self.cfg.use_view_dependent_prompt
			)
			with torch.no_grad():
				noise = torch.randn_like(latents)
				latents_noisy = self.scheduler.add_noise(latents, noise, t)
				latent_model_input = torch.cat([latents_noisy] * 4, dim=0)
				noise_pred = self.forward_unet(
					latent_model_input,
					control_image,
					torch.cat([t] * 4),
					encoder_hidden_states=text_embeddings,
				)  # (4B, 3, 64, 64)

			noise_pred_text = noise_pred[:batch_size]
			noise_pred_uncond = noise_pred[batch_size : batch_size * 2]
			noise_pred_neg = noise_pred[batch_size * 2 :]

			e_pos = noise_pred_text - noise_pred_uncond
			accum_grad = 0
			n_negative_prompts = neg_guidance_weights.shape[-1]
			for i in range(n_negative_prompts):
				e_i_neg = noise_pred_neg[i::n_negative_prompts] - noise_pred_uncond
				accum_grad += neg_guidance_weights[:, i].view(
					-1, 1, 1, 1
				) * perpendicular_component(e_i_neg, e_pos)

			noise_pred = noise_pred_uncond + self.cfg.guidance_scale * (
				e_pos + accum_grad
			)
			noise_pred = noise_pred.to(torch.float32)
		else:
			neg_guidance_weights = None
			text_embeddings = prompt_embedding.get_text_embedding(
				elevation, azimuth, camera_distances, self.cfg.use_view_dependent_prompt
			)
			# predict the noise residual with unet, NO grad!
			with torch.no_grad():
				# add noise
				noise = torch.randn_like(latents)  # TODO: use torch generator
				latents_noisy = self.scheduler.add_noise(latents, noise, t)
				# pred noise
				latent_model_input = torch.cat([latents_noisy] * 2, dim=0)
				# latent_model_input = latents_noisy
				if self.cfg.controlled:
					control_image_input = torch.cat([control_image] * 2, dim=0)
					# control_image_input = control_image
				else:
					control_image_input = None
				noise_pred = self.forward_unet(
					latent_model_input,
					control_image_input,
					torch.cat([t] * 2),
					encoder_hidden_states=text_embeddings,
					condition_scale=self.condition_scale
				)
				# noise_pred = self.forward_unet(
				# 	latent_model_input,
				# 	control_image_input,
				# 	t,
				# 	encoder_hidden_states=text_embeddings,
				# 	condition_scale=self.condition_scale
				# )
				latents_noisy_lora = self.scheduler_lora.add_noise(latents_lora, noise, t)
				# latent_model_input_lora = latents_noisy_lora
				latent_model_input_lora = torch.cat([latents_noisy_lora] * 2, dim=0)
				# noise_pred_est = self.forward_unet_lora(
				# 	latent_model_input_lora,
				# 	t,
				# 	control_image_input,
				# 	encoder_hidden_states=text_embeddings,
				# )
				noise_pred_est = self.forward_unet_lora(
					latent_model_input_lora,
					torch.cat([t] * 2),
					control_image_input,
					encoder_hidden_states=text_embeddings,
				)

			# perform guidance (high scale from paper!)
			noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
			noise_pred = noise_pred_uncond + self.cfg.guidance_scale * (
				noise_pred_text - noise_pred_uncond
			)
			
			assert self.scheduler.config.prediction_type == "epsilon"
			if self.scheduler_lora.config.prediction_type == "v_prediction":
				alphas_cumprod = self.scheduler_lora.alphas_cumprod.to(
					device=latents_noisy.device, dtype=latents_noisy.dtype
				)
				alpha_t = alphas_cumprod[t] ** 0.5
				sigma_t = (1 - alphas_cumprod[t]) ** 0.5

				noise_pred_est = latent_model_input * torch.cat([sigma_t] * 2, dim=0).view(
					-1, 1, 1, 1
				) + noise_pred_est * torch.cat([alpha_t] * 2, dim=0).view(-1, 1, 1, 1)

			(
				noise_pred_est_text,
				noise_pred_est_uncond,
			) = noise_pred_est.chunk(2)

			# NOTE: guidance scale definition here is aligned with diffusers, but different from other guidance
			noise_pred_est = noise_pred_est_uncond + self.cfg.guidance_scale_lora * (
				noise_pred_est_text - noise_pred_est_uncond
			)

		w = (1 - self.alphas[t]).view(-1, 1, 1, 1)
		grad = w * ((noise_pred - noise) - self.lora_scale * (noise_pred_est - noise)) 

		guidance_eval_utils = {
			"use_perp_neg": prompt_embedding.use_perp_negative,
			"neg_guidance_weights": neg_guidance_weights,
			"text_embeddings": text_embeddings,
			"t_orig": t,
			"latents_noisy": latents_noisy,
			"noise_pred": noise_pred,
		}

		# if render_image is not None:
		# 	self.scheduler.set_timesteps(50, device=self.device)
		# 	latents = self.scheduler.step(noise_pred, t[0], latents, return_dict=False)[0]
		# 	image = self.vae.decode(latents.to(torch.float16) / self.vae.config.scaling_factor)[0]
		# 	image = self.render_image_processor.postprocess(image.detach(), output_type="pil", do_denormalize=[True]*4)

		# 	image_transform = transforms.Compose([
		# 		transforms.Resize(256),
		# 		transforms.CenterCrop(256),
		# 		transforms.ToTensor(),
		# 		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
		# 	])

		# 	render_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
		# 	images = torch.empty((0, )).to(self.device)
		# 	for i in range(4):
		# 		images = torch.cat((images, image_transform(image[i]).to(self.device).unsqueeze(0)), dim=0)
		# 	render_image = render_transform(render_image)

		return grad, guidance_eval_utils
	
	def train_lora(
		self,
		latents,
		image_cond,
		prompt_embedding,
		elevation, azimuth, camera_distances
	):
		B = latents.shape[0]
		latents = latents.detach().repeat(1, 1, 1, 1)
		text_embeddings = prompt_embedding.get_text_embedding(
				elevation, azimuth, camera_distances, self.cfg.use_view_dependent_prompt
			)
		t = torch.randint(
			int(self.num_train_timesteps * 0.0),
			int(self.num_train_timesteps * 1.0),
			[B],
			dtype=torch.long,
			device=self.device,
		)
		noise = torch.randn_like(latents)
		noisy_latents = self.scheduler_lora.add_noise(latents, noise, t)
		if self.scheduler_lora.config.prediction_type == "epsilon":
			target = noise
		elif self.scheduler_lora.config.prediction_type == "v_prediction":
			target = self.scheduler_lora.get_velocity(latents, noise, t)
		else:
			raise ValueError(
				f"Unknown prediction type {self.scheduler_lora.config.prediction_type}"
			)
		
		input_dtype = latents.dtype
		_ = self.control_lora(image_cond).control_states
		# noise_pred = self.unet_lora(
		# 	noisy_latents.to(self.weights_dtype),
		# 	t.to(self.weights_dtype),
		# 	encoder_hidden_states=text_embeddings.to(self.weights_dtype),
		# ).sample.to(input_dtype)
		noise_pred = self.unet_lora(
			torch.cat([noisy_latents] * 2, dim=0).to(self.weights_dtype),
			torch.cat([t] * 2).to(self.weights_dtype),
			encoder_hidden_states=text_embeddings.to(self.weights_dtype),
		).sample.to(input_dtype)

		return F.mse_loss(noise_pred.float(), torch.cat([target] * 2, dim=0).float(), reduction="mean")

	def prepare_control_image(
		self,
		image,
		width,
		height,
		device,
		dtype
	):
		image = self.control_image_processor.preprocess(image, height=height, width=width).to(dtype=torch.float32)

		image = image.to(device=device, dtype=dtype)

		return image
	
	def prepare_out_image(
		self,
		image,
		width,
		height,
		device,
		dtype
	):
		image = self.render_image_processor.preprocess(image, height=height, width=width).to(dtype=torch.float32)

		image = image.to(device=device, dtype=dtype)

		return image
	   
	def forward(
		self,
		rgb,
		control_image,
		prompt_embedding,
		elevation,
		azimuth,
		camera_distance,
		rgb_as_latents=False,
		guidance_eval=False,
		geo_cond=None,
		**kwargs,
	):
		bs = rgb.shape[0]

		rgb_BCHW = rgb.permute(0, 3, 1, 2)
		if rgb_as_latents:
			latents = F.interpolate(
				rgb_BCHW, (64, 64), mode="bilinear", align_corners=False
			)
		else:
			rgb_BCHW_512 = F.interpolate(
				rgb_BCHW, (512, 512), mode="bilinear", align_corners=False
			)
			if geo_cond is not None:
				geo_cond = geo_cond.permute(0, 3, 1, 2)
			# encode image into latents with vae
			latents, latents_lora = self.encode_images(rgb_BCHW_512, geo_cond)
		
		if self.cfg.controlled:
			control_image = F.interpolate(
				control_image, (512, 512), mode="bilinear", align_corners=False
			).to(self.device, dtype=torch.float16)
			# control_image = self.prepare_control_image(
			# 	control_image,
			# 	512,
			# 	512,
			# 	device=self.device,
			# 	dtype=torch.float16
			# )

		t = torch.randint(
			self.min_t_step,
			self.max_t_step + 1,
			[bs],
			dtype=torch.long,
			device=self.device,
		)

		grad, guidance_eval_utils = self.compute_grad_sds(
			latents, latents_lora, control_image, t, prompt_embedding, elevation, azimuth, camera_distance
		)

		grad = torch.nan_to_num(grad)
		# clip grad for stable training?
		if self.grad_clip_val is not None:
			grad = grad.clamp(-self.grad_clip_val, self.grad_clip_val)

		target = (latents - grad).detach()
		loss_sds = 0.5 * F.mse_loss(latents, target, reduction="sum") / bs
		loss_sds_each = 0.5 * F.mse_loss(latents, target, reduction="none").sum(
			dim=[1, 2, 3]
		)

		loss_lora = self.train_lora(latents_lora, control_image, prompt_embedding, elevation, azimuth, camera_distance)

		guidance_out = {
			"loss_sds": loss_sds,
			"loss_lora": loss_lora,
			"loss_sds_each": loss_sds_each,
			"grad_norm": grad.norm(),
			"min_t_step": self.min_t_step,
			"max_t_step": self.max_t_step,
		}

		if guidance_eval:
			guidance_eval_out = self.guidance_eval(**guidance_eval_utils)
			texts = []
			for n, e, a, c in zip(
				guidance_eval_out["noise_levels"], elevation, azimuth, camera_distance
			):
				texts.append(
					f"n{n:.02f}\ne{e.item():.01f}\na{a.item():.01f}\nc{c.item():.02f}"
				)
			guidance_eval_out.update({"texts": texts})
			guidance_out.update({"eval": guidance_eval_out})

		return guidance_out

	# def step(self, epoch: int, step: int):
	#     if self.cfg.grad_clip is not None:
	#         self.grad_clip_val = C(self.cfg.grad_clip, epoch, step)

	# vanilla scheduler use constant min max steps
	# self.set_min_max_steps()

	# def train_lora(
	# 	self,
	# 	rgb,
	# 	control_image,
	# 	prompt_embedding,
	# 	elevation,
	# 	azimuth,
	# 	camera_distance,
	# 	rgb_as_latents=False,
	# 	geo_cond=None,
	# 	**kwargs
	# ):
	# 	self.lora_optimizer.zero_grad()

	# 	bs = rgb.shape[0]

	# 	rgb_BCHW = rgb.permute(0, 3, 1, 2)
	# 	if rgb_as_latents:
	# 		latents = F.interpolate(
	# 			rgb_BCHW, (64, 64), mode="bilinear", align_corners=False
	# 		)
	# 	else:
	# 		rgb_BCHW_512 = F.interpolate(
	# 			rgb_BCHW, (512, 512), mode="bilinear", align_corners=False
	# 		)
	# 		if geo_cond is not None:
	# 			geo_cond = geo_cond.permute(0, 3, 1, 2)
	# 		# encode image into latents with vae
	# 		_, latents = self.encode_images(rgb_BCHW_512, geo_cond)

	# 	if self.cfg.controlled:
	# 		control_image = F.interpolate(
	# 			control_image, (512, 512), mode="bilinear", align_corners=False
	# 		).to(self.device, dtype=torch.float16)
		
	# 	B = latents.shape[0]
	# 	latents = latents.detach().repeat(1, 1, 1, 1)
	# 	text_embeddings = prompt_embedding.get_text_embedding(
	# 			elevation, azimuth, camera_distance, self.cfg.use_view_dependent_prompt
	# 		)
	# 	t = torch.randint(
	# 		int(self.num_train_timesteps * 0.0),
	# 		int(self.num_train_timesteps * 1.0),
	# 		[B],
	# 		dtype=torch.long,
	# 		device=self.device,
	# 	)
	# 	noise = torch.randn_like(latents)
	# 	noisy_latents = self.scheduler_lora.add_noise(latents, noise, t)
	# 	if self.scheduler_lora.config.prediction_type == "epsilon":
	# 		target = noise
	# 	elif self.scheduler_lora.config.prediction_type == "v_prediction":
	# 		target = self.scheduler_lora.get_velocity(latents, noise, t)
	# 	else:
	# 		raise ValueError(
	# 			f"Unknown prediction type {self.scheduler_lora.config.prediction_type}"
	# 		)

	# 	# noise_pred = self.forward_unet_lora(
	# 	# 	noisy_latents,
	# 	# 	t,
	# 	# 	control_image,
	# 	# 	encoder_hidden_states=text_embeddings,
	# 	# )
	# 	noise_pred = self.forward_unet_lora(
	# 		torch.cat([noisy_latents] * 2, dim=0),
	# 		torch.cat([t] * 2),
	# 		control_image,
	# 		encoder_hidden_states=text_embeddings,
	# 	)

	# 	loss = F.mse_loss(noise_pred, torch.cat([target] * 2, dim=0), reduction="mean")
	# 	loss.backward()
	# 	self.lora_optimizer.step()
	# 	self.lora_scheduler.step()
	# 	return loss  


	def update(self, step):
		self.step = step
		self.set_min_max_steps(
			min_step_percent=C(self.cfg.min_step_percent, self.step, self.max_steps),
			max_step_percent=C(self.cfg.max_step_percent, self.step, self.max_steps),
			lora_scale=C(self.cfg.lora_scale, self.step, self.max_steps),
			condition_scale=C(self.cfg.condition_scale, self.step, self.max_steps)
		)
		if self.cfg.grad_clip is not None:
			self.grad_clip_val = C(self.cfg.grad_clip, step, self.max_steps)

	def log(self, writer, step):
		writer.add_scalar("guidance/min_step", self.min_t_step, step)
		writer.add_scalar("guidance/max_step", self.max_t_step, step)
