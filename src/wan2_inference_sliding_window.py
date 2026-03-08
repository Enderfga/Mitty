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


class SlidingWindowInference(torch.nn.Module):
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
    def inference_single_video(self, video_path, pipeline, device):
        """
        对单个视频按81帧切分，每段独立推理
        """
        chunk_size = 81  # 固定81帧一段

        height = self.hparams.dataset.height
        width = self.hparams.dataset.width

        # 读取视频
        video_reader = VideoReader(video_path)
        total_frames = len(video_reader)

        print(f"[Infer] Video: {video_path}, total_frames: {total_frames}")

        # 准备transforms
        video_transforms = transforms.Compose([
            transforms.CenterCrop((height, width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        # 计算需要多少个chunk（每81帧一段，不重叠）
        num_chunks = total_frames // chunk_size
        if total_frames % chunk_size > 0:
            num_chunks += 1

        print(f"[Infer] Will generate {num_chunks} separate 81-frame videos")

        all_results = []

        for chunk_idx in range(num_chunks):
            start_frame = chunk_idx * chunk_size
            end_frame = min(start_frame + chunk_size, total_frames)
            actual_frames = end_frame - start_frame

            print(f"[Infer] Chunk {chunk_idx+1}/{num_chunks}: frames [{start_frame}, {end_frame}), actual={actual_frames}")

            # 读取帧
            frame_indices = np.arange(start_frame, end_frame)
            frames = video_reader.get_batch(frame_indices).asnumpy()
            frames = [Image.fromarray(frame) for frame in frames]
            pixel_values = [video_transforms(frame) for frame in frames]
            pixel_values = torch.stack(pixel_values)  # [F, C, H, W]

            # 如果不足81帧，padding到81帧
            if actual_frames < chunk_size:
                padding = chunk_size - actual_frames
                last_frame = pixel_values[-1:].repeat(padding, 1, 1, 1)
                pixel_values = torch.cat([pixel_values, last_frame], dim=0)

            # 转换为 [1, C, F, H, W]
            model_input = pixel_values.permute(1, 0, 2, 3).unsqueeze(0).to(device).to(torch.bfloat16)

            # encode到latent
            model_input_lat = self.vae.encode(model_input).latent_dist.sample()
            model_input_lat = (model_input_lat - self.latents_mean) / self.latents_std

            # first_frames占位
            first_frames_lat = model_input_lat[:, :, :1].detach() * 0

            attention_kwargs = {
                'encoder_contion_states': model_input_lat,
                'encoder_first_states': first_frames_lat,
            }

            # 推理
            prompt = [""]  # 空prompt
            out = pipeline(
                prompt=prompt,
                height=height,
                width=width,
                num_frames=chunk_size,
                guidance_scale=5.0,
                attention_kwargs=attention_kwargs,
            )
            video_chunk = out.frames[0]  # numpy array [F, H, W, C]

            # 裁剪回实际帧数
            video_chunk = video_chunk[:actual_frames]

            all_results.append({
                'chunk_idx': chunk_idx,
                'start_frame': start_frame,
                'end_frame': end_frame,
                'video': video_chunk
            })

        return all_results

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
        os.makedirs(save_root, exist_ok=True)
        print(f"[Infer] Save to: {save_root}")

        video_path = self.hparams.video_path
        video_name = os.path.basename(video_path).replace('.mp4', '')

        # 推理
        all_results = self.inference_single_video(video_path, pipeline, device)

        # 为每个chunk保存独立的视频
        video_reader = VideoReader(video_path)
        height = self.hparams.dataset.height
        width = self.hparams.dataset.width

        for result in all_results:
            chunk_idx = result['chunk_idx']
            start_frame = result['start_frame']
            end_frame = result['end_frame']
            generated_video = result['video']

            # 保存生成的视频
            save_path = os.path.join(save_root, f"{video_name}_chunk{chunk_idx:03d}_generated.mp4")
            export_to_video(generated_video, output_video_path=save_path, fps=self.hparams.dataset.fps)
            print(f"[Infer] Saved chunk {chunk_idx}: {save_path}")

            # 保存输入视频的对应部分（用于对比）
            input_frames = video_reader.get_batch(np.arange(start_frame, end_frame)).asnumpy()
            input_frames_resized = []
            for frame in input_frames:
                img = Image.fromarray(frame)
                img = transforms.CenterCrop((height, width))(img)
                input_frames_resized.append(np.array(img))
            input_frames_resized = np.stack(input_frames_resized)

            # 拼接input和output (转换为float32 [0,1]格式以避免export_to_video的自动*255)
            input_frames_float = input_frames_resized.astype(np.float32) / 255.0
            concat_video = np.concatenate([input_frames_float, generated_video], axis=2)
            save_path_concat = os.path.join(save_root, f"{video_name}_chunk{chunk_idx:03d}_concat.mp4")
            export_to_video(concat_video, output_video_path=save_path_concat, fps=self.hparams.dataset.fps)
            print(f"[Infer] Saved concat chunk {chunk_idx}: {save_path_concat}")

        print(f"[Infer] Completed all {len(all_results)} chunks for {video_name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/inference_config.yaml")
    parser.add_argument("--video_path", type=str, required=True, help="path to input video")
    parser.add_argument("--ckpt_path", type=str, required=True, help="path to LoRA checkpoint")
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
    system = SlidingWindowInference(opt)
    system.run_infer()


if __name__ == "__main__":
    main()
