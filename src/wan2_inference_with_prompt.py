import os
import argparse
import copy
import warnings
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from einops import rearrange
from omegaconf import OmegaConf

import pytorch_lightning as L
from pytorch_lightning.utilities import rank_zero_only

from transformers import AutoTokenizer, UMT5EncoderModel
from diffusers import AutoencoderKLWan, UniPCMultistepScheduler
from diffusers.utils import export_to_video
from diffusers import FlowMatchEulerDiscreteScheduler

from models.wan2.transformer_wan import WanTransformer3DModel
from models.wan2.custom_pipeline import CustomWanPipeline as WanPipeline
from models.wan2.attn_process import ConditionAttnProcessor2_0

from tools.my_schedule import FlowMatchScheduler, MyFlowMatchEulerDiscreteScheduler
from decord import VideoReader
from PIL import Image
from torchvision import transforms


@rank_zero_only
def silence_warnings():
    warnings.filterwarnings("ignore", category=UserWarning)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class PromptBasedInference(torch.nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.hparams = opt
        self.is_configured = False

    def configure_model(self):
        if self.is_configured:
            return
        self.is_configured = True

        model_id = self.hparams.model_id

        # tokenizer / text encoder
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, subfolder="tokenizer")
        self.text_encoder = UMT5EncoderModel.from_pretrained(
            model_id, subfolder="text_encoder", torch_dtype=torch.bfloat16
        )

        # VAE
        self.vae = AutoencoderKLWan.from_pretrained(
            model_id, subfolder="vae", torch_dtype=torch.bfloat16
        )

        # schedulers
        if self.hparams.use_DiffSynth:
            self.train_scheduler = FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True)
            self.train_scheduler.set_timesteps(1000, training=True)
        else:
            self.train_scheduler = MyFlowMatchEulerDiscreteScheduler.from_pretrained(
                model_id, subfolder="scheduler"
            )
        base_sampler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            model_id, subfolder="scheduler"
        )
        self.sample_scheduler = UniPCMultistepScheduler.from_config(
            base_sampler.config, flow_shift=5
        )

        # transformer
        self.transformer = WanTransformer3DModel.from_pretrained(
            model_id, subfolder="transformer", torch_dtype=torch.bfloat16
        )

        # 冻结主干
        self.text_encoder.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.transformer.requires_grad_(False)

        # gradient checkpoint
        if getattr(self.hparams.training, "gradient_checkpointing", False):
            self.transformer.gradient_checkpointing = True
            self.transformer.enable_gradient_checkpointing()

        # latents 标准化参数
        self.register_buffer(
            'latents_mean',
            torch.tensor(self.vae.config.latents_mean).float().view(1, self.vae.config.z_dim, 1, 1, 1),
            persistent=False
        )
        self.register_buffer(
            'latents_std',
            torch.tensor(self.vae.config.latents_std).float().view(1, self.vae.config.z_dim, 1, 1, 1),
            persistent=False
        )

        # 保存 config
        self.vae_config = self.vae.config
        self.model_config = self.transformer.module.config if hasattr(self.transformer, "module") else self.transformer.config

        # LoRA
        self.using_lora = bool(self.hparams.use_lora)
        if self.using_lora:
            from peft import LoraConfig
            transformer_lora_config = LoraConfig(
                r=96, lora_alpha=96, init_lora_weights=True,
                target_modules=["to_k", "to_q", "to_v", "to_out.0"],
            )
            self.transformer.add_adapter(transformer_lora_config)

        # 设置 ConditionAttnProcessor
        for blk in self.transformer.blocks:
            blk.attn1.set_processor(ConditionAttnProcessor2_0())

        # 额外 patch embedding
        self.transformer.patch_embedding_extra = copy.deepcopy(self.transformer.patch_embedding).requires_grad_(True)

    @torch.no_grad()
    def encode_prompt(self, prompt_list, device):
        max_sequence_length = 512
        text_inputs = self.tokenizer(
            prompt_list,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        ids, mask = text_inputs.input_ids.to(device), text_inputs.attention_mask.to(device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        text_embeds = self.text_encoder(ids, mask).last_hidden_state
        text_embeds = [u[:v] for u, v in zip(text_embeds, seq_lens)]
        text_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))]) for u in text_embeds], dim=0
        )
        return text_embeds

    def _load_lora_from_ckpt(self, ckpt_path, device):
        if ckpt_path in (None, "", "None", "null"):
            print("[Infer] No ckpt_path provided. Skip loading LoRA.")
            return

        print(f"[Infer] Loading checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        if "state_dict" not in ckpt:
            print("[Infer] checkpoint has no 'state_dict' key; skip.")
            return

        sd_all = ckpt["state_dict"]

        # LoRA/attn processor
        if "transformer_processor" in sd_all:
            sd = sd_all["transformer_processor"]
            cur = self.transformer.state_dict()
            filtered = {k: v for k, v in sd.items() if (k in cur and cur[k].shape == v.shape)}
            skipped = [k for k in sd.keys() if k not in filtered]
            print(f"[Infer][LoRA] Load {len(filtered)}/{len(sd)} keys. Skipped {len(skipped)} mismatched keys.")
            self.transformer.load_state_dict(filtered, strict=False)
        else:
            print("[Infer] 'transformer_processor' not found in ckpt.state_dict; skip LoRA.")

        # patch_embedding_extra
        if "patch_embedding_extra" in sd_all:
            sd2 = sd_all["patch_embedding_extra"]
            cur2 = self.transformer.state_dict()
            filtered2 = {k: v for k, v in sd2.items() if (k in cur2 and cur2[k].shape == v.shape)}
            skipped2 = [k for k in sd2.keys() if k not in filtered2]
            print(f"[Infer][patch_embedding_extra] Load {len(filtered2)}/{len(sd2)} keys. Skipped {len(skipped2)} mismatched keys.")
            self.transformer.load_state_dict(filtered2, strict=False)
        else:
            print("[Infer] 'patch_embedding_extra' not found in ckpt.state_dict; skip.")

    @torch.no_grad()
    def inference_chunk(self, pixel_values, prompt, pipeline, device):
        """
        对一个 chunk 的 pixel_values [F, C, H, W] 推理
        """
        height = self.hparams.dataset.height
        width = self.hparams.dataset.width
        num_frames = self.hparams.dataset.sample_n_frames

        actual_frames = pixel_values.shape[0]

        # padding 到 num_frames
        if actual_frames < num_frames:
            padding = num_frames - actual_frames
            last_frame = pixel_values[-1:].repeat(padding, 1, 1, 1)
            pixel_values = torch.cat([pixel_values, last_frame], dim=0)

        # 转换为 [1, C, F, H, W]
        model_input = pixel_values.permute(1, 0, 2, 3).unsqueeze(0).to(device).to(torch.bfloat16)

        # encode 到 latent
        model_input_lat = self.vae.encode(model_input).latent_dist.sample()
        model_input_lat = (model_input_lat - self.latents_mean) / self.latents_std

        # first_frames 占位
        first_frames_lat = model_input_lat[:, :, :1].detach() * 0

        attention_kwargs = {
            'encoder_contion_states': model_input_lat,
            'encoder_first_states': first_frames_lat,
        }

        # 推理
        out = pipeline(
            prompt=[prompt],
            height=height,
            width=width,
            num_frames=num_frames,
            guidance_scale=5.0,
            attention_kwargs=attention_kwargs,
        )
        generated_video = out.frames[0]  # numpy array [F, H, W, C]

        # 裁剪回实际帧数
        generated_video = generated_video[:actual_frames]

        return generated_video

    @torch.no_grad()
    def inference_with_prompt(self, video_path, prompt, pipeline, device):
        """
        对单个视频用给定 prompt 做滑窗推理，返回 chunk 列表
        """
        height = self.hparams.dataset.height
        width = self.hparams.dataset.width
        chunk_size = self.hparams.dataset.sample_n_frames  # 49

        # 读取视频
        video_reader = VideoReader(video_path)
        total_frames = len(video_reader)

        print(f"[Infer] Video: {video_path}, total_frames: {total_frames}, prompt: '{prompt}'")

        # 准备 transforms
        video_transforms = transforms.Compose([
            transforms.CenterCrop((height, width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        # 读取全部帧并 transform
        all_frames = video_reader.get_batch(np.arange(total_frames)).asnumpy()
        all_pixel_values = []
        for frame in all_frames:
            img = Image.fromarray(frame)
            all_pixel_values.append(video_transforms(img))
        all_pixel_values = torch.stack(all_pixel_values)  # [T, C, H, W]

        # 滑窗切分 (不重叠)
        num_chunks = (total_frames + chunk_size - 1) // chunk_size
        print(f"[Infer] Splitting into {num_chunks} chunks of {chunk_size} frames")

        chunk_results = []
        for chunk_idx in range(num_chunks):
            start = chunk_idx * chunk_size
            end = min(start + chunk_size, total_frames)
            chunk_pv = all_pixel_values[start:end]

            print(f"[Infer]   Chunk {chunk_idx+1}/{num_chunks}: frames [{start}, {end})")
            gen_chunk = self.inference_chunk(chunk_pv, prompt, pipeline, device)
            chunk_results.append(gen_chunk)

        return chunk_results

    @torch.no_grad()
    def run_infer(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.configure_model()
        self.to(device)

        # 加载 LoRA
        if self.using_lora:
            ckpt_path = getattr(self.hparams, "ckpt_path", None)
            self._load_lora_from_ckpt(ckpt_path, device)

        # 采样管线
        pipeline = WanPipeline(
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            transformer=self.transformer,
            scheduler=self.sample_scheduler,
        )

        save_root = os.path.join(self.hparams.output_root, self.hparams.experiment_name)

        video_path = self.hparams.video_path
        prompt = self.hparams.prompt
        view_type = self.hparams.view_type  # 'front', 'wrist', or 'back'
        sample_id = self.hparams.sample_id  # e.g. '0', '1', ...
        fps = self.hparams.dataset.fps
        height = self.hparams.dataset.height
        width = self.hparams.dataset.width
        chunk_size = self.hparams.dataset.sample_n_frames  # 49

        # 按 view_type 分文件夹，每个 sample 一个子目录
        view_dir = os.path.join(save_root, view_type, sample_id)
        os.makedirs(view_dir, exist_ok=True)
        print(f"[Infer] Save to: {view_dir}")

        # 读取视频
        video_reader = VideoReader(video_path)
        total_frames = len(video_reader)
        print(f"[Infer] Video: {video_path}, total_frames: {total_frames}, prompt: '{prompt}'")

        # 准备 transforms
        video_transforms = transforms.Compose([
            transforms.CenterCrop((height, width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        # 读取全部帧并 transform
        all_frames = video_reader.get_batch(np.arange(total_frames)).asnumpy()
        all_pixel_values = []
        for frame in all_frames:
            img = Image.fromarray(frame)
            all_pixel_values.append(video_transforms(img))
        all_pixel_values = torch.stack(all_pixel_values)  # [T, C, H, W]

        # 滑窗切分 + 逐 chunk 推理并立刻保存
        num_chunks = (total_frames + chunk_size - 1) // chunk_size
        print(f"[Infer] Splitting into {num_chunks} chunks of {chunk_size} frames")

        chunk_paths = []
        for chunk_idx in range(num_chunks):
            start = chunk_idx * chunk_size
            end = min(start + chunk_size, total_frames)
            chunk_pv = all_pixel_values[start:end]

            print(f"[Infer]   Chunk {chunk_idx+1}/{num_chunks}: frames [{start}, {end})")
            gen_chunk = self.inference_chunk(chunk_pv, prompt, pipeline, device)

            # 立刻保存这个 chunk
            save_path = os.path.join(view_dir, f"chunk_{chunk_idx:03d}.mp4")
            export_to_video(gen_chunk, output_video_path=save_path, fps=fps)
            chunk_paths.append(save_path)
            print(f"[Infer] Saved: {save_path}")

        print(f"[Infer] Completed sample={sample_id} view={view_type}: {len(chunk_paths)} chunks saved to {view_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/inference_config.yaml")
    parser.add_argument("--video_path", type=str, required=True, help="path to input video")
    parser.add_argument("--ckpt_path", type=str, required=True, help="path to LoRA checkpoint")
    parser.add_argument("--prompt", type=str, required=True, help="prompt for generation")
    parser.add_argument("--view_type", type=str, required=True, choices=['front', 'wrist', 'back'], help="view type")
    parser.add_argument("--sample_id", type=str, required=True, help="sample identifier for output filename")
    parser.add_argument("--seed", type=int, default=42)
    args, extras = parser.parse_known_args()
    args = vars(args)

    # 合并配置
    opt = OmegaConf.merge(
        OmegaConf.load(args['config']),
        OmegaConf.from_cli(extras),
        OmegaConf.create(args),
        OmegaConf.create({"num_nodes": int(os.environ.get("NUM_NODES", 1))}),
        OmegaConf.create({"num_gpus": int(torch.cuda.device_count())}),
    )

    # 设定随机种子
    L.seed_everything(opt.seed)

    # 跑推理
    system = PromptBasedInference(opt)
    system.run_infer()


if __name__ == "__main__":
    main()
