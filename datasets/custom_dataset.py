from torch.utils.data import Dataset
import os
from decord import VideoReader
from torchvision import transforms
import numpy as np
from PIL import Image
import torch


class CustomDataset(Dataset):
    def __init__(
        self,
        video_root,
        video_root2,
        first_root,
        height=512,
        width=512,
        sample_n_frames=49,
        is_one2three=False,
        training_len=-1,
        caption_ext=".txt",  # 文本文件后缀
        # 子文件夹模式的参数
        subfolder_mode=False,
        human_filename="video_L.mp4",
        robot_filename="2.mp4",
    ):
        self.training_len = training_len
        self.is_one2three = is_one2three
        self.caption_ext = caption_ext
        self.subfolder_mode = subfolder_mode

        # --- 可选首帧：目录存在且有 >=1 个可读图片才开启 ---
        img_exts = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
        if first_root and os.path.isdir(first_root):
            first_list_all = sorted(os.listdir(first_root))
            first_list = [x for x in first_list_all if x.lower().endswith(img_exts)]
            if len(first_list) > 0:
                self.first_root = first_root
                self.first_paths = [os.path.join(first_root, x) for x in first_list]
                self.use_first = True
            else:
                self.first_root = None
                self.first_paths = []
                self.use_first = False
        else:
            self.first_root = None
            self.first_paths = []
            self.use_first = False

        video_exts = (".mp4", ".avi", ".mov", ".mkv")

        if subfolder_mode:
            # === 子文件夹模式 ===
            # video_root 和 video_root2 可以是多个目录（用逗号分隔）
            roots = []
            if video_root:
                roots.extend([r.strip() for r in video_root.split(",") if r.strip()])
            if video_root2:
                roots.extend([r.strip() for r in video_root2.split(",") if r.strip()])

            self.video_pairs = []  # [(human_path, robot_path), ...]

            for root in roots:
                if not os.path.isdir(root):
                    continue
                # 遍历子文件夹
                for subfolder in sorted(os.listdir(root)):
                    subfolder_path = os.path.join(root, subfolder)
                    if not os.path.isdir(subfolder_path):
                        continue
                    human_path = os.path.join(subfolder_path, human_filename)
                    robot_path = os.path.join(subfolder_path, robot_filename)
                    # 两个视频都存在才加入
                    if os.path.exists(human_path) and os.path.exists(robot_path):
                        self.video_pairs.append((human_path, robot_path))

            print(f"CustomDataset [subfolder_mode]: 找到 {len(self.video_pairs)} 对视频")
            self.inference_mode = False
            self.use_first = False  # 子文件夹模式暂不支持首帧
        else:
            # === 原始模式 ===
            self.video_root = video_root
            self.video_root2 = video_root2

            print(
                f"CustomDataset: video_root: {video_root}, "
                f"video_root2: {video_root2}, first_root: {self.first_root or ''}"
            )

            # 视频列表（只收视频后缀）
            video_list = sorted(
                [x for x in os.listdir(self.video_root) if x.lower().endswith(video_exts)]
            )
            self.video_paths = [os.path.join(self.video_root, v) for v in video_list]

            # video_root2 可选（推理时可能没有 gt）
            if self.video_root2 and os.path.isdir(self.video_root2):
                video_list2 = sorted(
                    [x for x in os.listdir(self.video_root2) if x.lower().endswith(video_exts)]
                )
                self.video_paths2 = [os.path.join(self.video_root2, v) for v in video_list2]
                self.inference_mode = False
            else:
                self.video_paths2 = self.video_paths  # 推理模式：复用 video_paths
                self.inference_mode = True

            self.len_videos = len(self.video_paths)
            self.len_videos2 = len(self.video_paths2)
            self.len_firsts = len(self.first_paths)

            # 两路视频必须一一对应（推理模式下跳过）
            if not self.inference_mode:
                assert self.len_videos == self.len_videos2, "mismatch in first videos and third videos"

        self.height = height
        self.width = width
        self.train_video_transforms = transforms.Compose(
            [
                transforms.CenterCrop((height, width)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

        self.sample_n_frames = sample_n_frames

    def __len__(self):
        if self.training_len != -1:
            return self.training_len
        if self.subfolder_mode:
            return len(self.video_pairs)
        if self.use_first:
            return min(self.len_videos, self.len_videos2, self.len_firsts)
        else:
            return min(self.len_videos, self.len_videos2)

    def _caption_path_for(self, video2_path: str):
        stem, _ = os.path.splitext(video2_path)
        return stem + self.caption_ext

    def _load_caption(self, video2_path: str) -> str:
        cap_path = self._caption_path_for(video2_path)
        if os.path.exists(cap_path) and os.path.isfile(cap_path):
            try:
                with open(cap_path, "r", encoding="utf-8") as f:
                    return f.read().strip()
            except Exception:
                return ""
        return ""

    def _sample_frames_full_coverage(self, video_length, sample_n_frames):
        """
        完整进度采样：计算能覆盖整个视频的 stride，随机选择 offset
        例如：视频 810 帧，采样 81 帧，stride=10，offset 可选 0-9
        """
        # 计算能覆盖完整视频的 stride
        stride = max(1, video_length // sample_n_frames)

        # 随机选择 offset（数据增强）
        offset = np.random.randint(0, stride)

        # 生成帧索引
        frame_indices = offset + np.arange(sample_n_frames) * stride

        # 确保不越界
        frame_indices = np.clip(frame_indices, 0, video_length - 1)

        return frame_indices

    def __getitem__(self, index):
        if self.subfolder_mode:
            # === 子文件夹模式 ===
            index = index % len(self.video_pairs)
            human_path, robot_path = self.video_pairs[index]

            # 读取人类视频
            video_reader_human = VideoReader(human_path)
            video_length_human = len(video_reader_human)

            # 读取机械臂视频
            video_reader_robot = VideoReader(robot_path)
            video_length_robot = len(video_reader_robot)

            # 各自采样，覆盖完整进度
            frame_indices_human = self._sample_frames_full_coverage(
                video_length_human, self.sample_n_frames
            )
            frame_indices_robot = self._sample_frames_full_coverage(
                video_length_robot, self.sample_n_frames
            )

            # 读取人类视频帧
            video_human = video_reader_human.get_batch(frame_indices_human).asnumpy()
            video_human = [Image.fromarray(frame) for frame in video_human]
            pixel_values = [self.train_video_transforms(frame) for frame in video_human]
            pixel_values = torch.stack(pixel_values)

            # 读取机械臂视频帧
            video_robot = video_reader_robot.get_batch(frame_indices_robot).asnumpy()
            video_robot = [Image.fromarray(frame) for frame in video_robot]
            pixel_values2 = [self.train_video_transforms(frame) for frame in video_robot]
            pixel_values2 = torch.stack(pixel_values2)

            # 文本
            prompt = self._load_caption(robot_path)

            sample = {
                "pixel_values": pixel_values.permute(1, 0, 2, 3),   # C, F, H, W
                "pixel_values2": pixel_values2.permute(1, 0, 2, 3), # C, F, H, W
                "prompts": prompt,
            }
            return sample

        # === 原始模式 ===
        # 根据是否启用首帧决定取模长度，避免越界
        if self.use_first:
            min_len = min(self.len_videos, self.len_videos2, self.len_firsts)
        else:
            min_len = min(self.len_videos, self.len_videos2)

        # 如果真的没有数据，给出清晰报错（避免除以 0 或空取模）
        if min_len <= 0:
            raise RuntimeError(
                f"No valid samples: "
                f"len_videos={self.len_videos}, len_videos2={self.len_videos2}, len_firsts={self.len_firsts}."
            )

        index = index % min_len

        # video A (人类)
        video_path = self.video_paths[index]
        video_reader = VideoReader(video_path)
        video_length = len(video_reader)

        # video B (机械臂 GT)
        video_path2 = self.video_paths2[index]
        video_reader2 = VideoReader(video_path2)
        video_length2 = len(video_reader2)

        # 可选的首帧
        first_frame = None
        if self.use_first:
            first_frame_path = self.first_paths[index]
            first_frame = Image.open(first_frame_path).convert("RGB")
            first_frame = self.train_video_transforms(first_frame)

        # 各自采样，覆盖完整进度（支持不同长度的视频）
        frame_indices_human = self._sample_frames_full_coverage(
            video_length, self.sample_n_frames
        )
        frame_indices_robot = self._sample_frames_full_coverage(
            video_length2, self.sample_n_frames
        )

        # 读取视频 A (人类)
        video = video_reader.get_batch(frame_indices_human).asnumpy()
        video = [Image.fromarray(frame) for frame in video]
        pixel_values = [self.train_video_transforms(frame) for frame in video]
        pixel_values = torch.stack(pixel_values)

        # 读取视频 B (机械臂 GT)
        video2 = video_reader2.get_batch(frame_indices_robot).asnumpy()
        video2 = [Image.fromarray(frame) for frame in video2]
        pixel_values2 = [self.train_video_transforms(frame) for frame in video2]
        pixel_values2 = torch.stack(pixel_values2)

        # 文本：优先读取 video_root2 同名 .txt；没有则返回空字符串
        prompt = self._load_caption(video_path2)

        sample = {
            "pixel_values": pixel_values.permute(1, 0, 2, 3),       # C, F, H, W
            "pixel_values2": pixel_values2.permute(1, 0, 2, 3),     # C, F, H, W
            "prompts": prompt,
        }
        if self.use_first:
            sample["first_frames"] = first_frame  # C, H, W

        return sample
