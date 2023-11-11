import random
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import polars as pl
import torch
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset
from torchvision.transforms.functional import resize

from src.utils.common import pad_if_needed


###################
# Load Functions
###################
def load_features(
    feature_names: list[str],
    series_ids: Optional[list[str]],
    processed_dir: Path,
    phase: str,
) -> dict[str, np.ndarray]:
    features = {}

    if series_ids is None:
        series_ids = [series_dir.name for series_dir in (processed_dir / phase).glob("*")]
        '''
        glob("*") 方法获取 processed_dir / phase 目录下的所有子目录（每个系列一个目录），并将它们的名称作为系列标识符列表
        每个系列的数据都存储在以系列 ID 命名的文件夹中。
        '''

    for series_id in series_ids:
        series_dir = processed_dir / phase / series_id
        this_feature = []
        for feature_name in feature_names:
            this_feature.append(np.load(series_dir / f"{feature_name}.npy"))
        features[series_dir.name] = np.stack(this_feature, axis=1)

    return features


#将时间序列数据分割成长度为 duration 的连续块
def load_chunk_features(
    duration: int,
    feature_names: list[str],
    series_ids: Optional[list[str]],
    processed_dir: Path,
    phase: str,
) -> dict[str, np.ndarray]:
    features = {}

    if series_ids is None:
        series_ids = [series_dir.name for series_dir in (processed_dir / phase).glob("*")]

    for series_id in series_ids:
        series_dir = processed_dir / phase / series_id
        this_feature = []
        for feature_name in feature_names:
            this_feature.append(np.load(series_dir / f"{feature_name}.npy"))
        this_feature = np.stack(this_feature, axis=1)
        num_chunks = (len(this_feature) // duration) + 1
        for i in range(num_chunks):
            chunk_feature = this_feature[i * duration : (i + 1) * duration]
            chunk_feature = pad_if_needed(chunk_feature, duration, pad_value=0)  # type: ignore
            features[f"{series_id}_{i:07}"] = chunk_feature

    return features  # type: ignore


###################
# Augmentation
###################
#random_crop 函数的作用是在一个给定的时间序列中随机选择一段长度为 duration 的子序列
# 这个子序列包含了一个特定的位置 pos
def random_crop(pos: int, duration: int, max_end) -> tuple[int, int]:
    """Randomly crops with duration length including pos.
    However, 0<=start, end<=max_end
    """
    start = random.randint(max(0, pos - duration), min(pos, max_end - duration))
    #既包含了指定的位置 pos，又不会超出整个时间序列的范围
    end = start + duration
    return start, end


###################
# Label
###################
def get_label(
    this_event_df: pd.DataFrame, num_frames: int, duration: int, start: int, end: int
) -> np.ndarray:
    # # (start, end)の範囲と(onset, wakeup)の範囲が重なるものを取得
    this_event_df = this_event_df.query("@start <= wakeup & onset <= @end")

    label = np.zeros((num_frames, 3))
    # onset, wakeup, sleepのラベルを作成
    for onset, wakeup in this_event_df[["onset", "wakeup"]].to_numpy():
        onset = int((onset - start) / duration * num_frames)
        wakeup = int((wakeup - start) / duration * num_frames)
        '''
        这一步将 onset  wakeup时间点转换为相对于数据块（由 start 到 end）的位置
        duration 数据块长度

        num_frames: 这通常是经过处理（如上采样或下采样）后的数据块中的帧数。
        num_frames 是根据上采样或下采样的比率计算得出的
        例如，如果原始数据（duration 长度的数据）被上采样或下采样，num_frames 将表示处理后数据块的帧数。

        将 onset 和 wakeup 时间点从整个时间序列的上下文中转换为相对于当前处理的数据块（子序列）的位置
        '''
        if onset >= 0 and onset < num_frames:
            label[onset, 1] = 1
        if wakeup < num_frames and wakeup >= 0:
            label[wakeup, 2] = 1

        onset = max(0, onset)
        wakeup = min(num_frames, wakeup)
        label[onset:wakeup, 0] = 1  # sleep
        '''
        每个标签由一个三维向量表示，分别对应于睡眠、入睡和醒来。
        然后，这个函数根据事件的发生时间（onset、wakeup）来填充这个标签数组。
        如果一个事件（入睡或醒来）在所选区间内发生，相应的标签位置被设置为 1。
        此外，从入睡时间到醒来时间之间的所有帧都被标记为睡眠（第一个维度）。
        '''

    return label


# ref: https://www.kaggle.com/competitions/dfl-bundesliga-data-shootout/discussion/360236#2004730
# 高斯核生成：这个函数生成了一个高斯（正态分布）核。这个核是用来平滑数据的
# 其中 sigma 表示高斯分布的标准差，决定了平滑的程度，而 length 决定了核的长度 
# 高斯核是一维的
def gaussian_kernel(length: int, sigma: int = 3) -> np.ndarray:
    x = np.ogrid[-length : length + 1]
    h = np.exp(-(x**2) / (2 * sigma * sigma))  # type: ignore
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def gaussian_label(label: np.ndarray, offset: int, sigma: int) -> np.ndarray:
    num_events = label.shape[1]
    for i in range(num_events):
        label[:, i] = np.convolve(label[:, i], gaussian_kernel(offset, sigma), mode="same")
        '''
        卷积是一种数学运算，用于将两个信号（在这里是标签向量和高斯核）结合起来
        通过卷积操作，原始的硬标签（在这种情况下是二进制标签，如 0 或 1，表示没有事件或事件发生）被转换为平滑的、连续的值
        这可以提供更丰富的信息，特别是对于模型学习边界情况或不确定性时。
        '''

    return label


#negative_sampling 函数能够从时间序列中的非事件部分随机选择一个位置，作为负样本
# 有助于模型学习区分事件和非事件，从而提高其在不平衡数据集上的表现。
def negative_sampling(this_event_df: pd.DataFrame, num_steps: int) -> int:
    """negative sampling

    Args:
        this_event_df (pd.DataFrame): event df
        num_steps (int): number of steps in this series

    Returns:
        int: negative sample position
    """
    # onsetとwakupを除いた範囲からランダムにサンプリング
    positive_positions = set(this_event_df[["onset", "wakeup"]].to_numpy().flatten().tolist())
    negative_positions = list(set(range(num_steps)) - positive_positions)
    return random.sample(negative_positions, 1)[0]


###################
# Dataset
###################
#帮助调整输入数据的大小，以确保它们适合特定的网络架构要求
def nearest_valid_size(input_size: int, downsample_rate: int) -> int:
    """
    (x // hop_length) % 32 == 0
    を満たすinput_sizeに最も近いxを返す
    """

    while (input_size // downsample_rate) % 32 != 0:
        input_size += 1
    assert (input_size // downsample_rate) % 32 == 0

    return input_size


class TrainDataset(Dataset):
    def __init__(
        self,
        cfg: DictConfig,
        event_df: pl.DataFrame,
        features: dict[str, np.ndarray],
    ):
        self.cfg = cfg
        self.event_df: pd.DataFrame = (
            event_df.pivot(index=["series_id", "night"], columns="event", values="step")
            .drop_nulls()
            .to_pandas()
        )
        '''
        每个“series_id”和“night”的组合将有一个行
        每个“event”的唯一值将有一个列
        而这些列中的数据将是对应的“step”值
        '''
        self.features = features
        self.num_features = len(cfg.features)
        self.upsampled_num_frames = nearest_valid_size(
            int(self.cfg.duration * self.cfg.upsample_rate), self.cfg.downsample_rate
        )

    def __len__(self):
        return len(self.event_df)

    def __getitem__(self, idx):
        '''
        由于 event_df 是通过 pivot 方法从原始 DataFrame 转换而来，并且其索引设置为 ["series_id", "night"] 的组合
        这意味着 idx 实际上是对 series_id 和 night 的组合索引的引用。
        每个 idx 标识了一个独特的序列及其在特定夜晚的数据，其中包含了这一夜的各种事件（如“入睡”、“醒来”等）的时间步骤。
        '''
        event = np.random.choice(["onset", "wakeup"], p=[0.5, 0.5])
        #随机选择 "onset"（入睡事件）或 "wakeup"（醒来事件）中的一个，两者被选择的概率均为 50%。
        pos = self.event_df.at[idx, event]
        series_id = self.event_df.at[idx, "series_id"]
        self.event_df["series_id"]
        this_event_df = self.event_df.query("series_id == @series_id").reset_index(drop=True)
        #某个序列的csv
        # extract data matching series_id
        this_feature = self.features[series_id]  # (n_steps, num_features)
        n_steps = this_feature.shape[0]

        # sample background
        #随机数小于背景采样率
        if random.random() < self.cfg.bg_sampling_rate:
            pos = negative_sampling(this_event_df, n_steps)
            #pos 被重新赋值为一个背景（非事件）位置

        # crop
        start, end = random_crop(pos, self.cfg.duration, n_steps)
        feature = this_feature[start:end]  # (duration, num_features)

        # upsample
        feature = torch.FloatTensor(feature.T).unsqueeze(0)  # (1, num_features, duration)
        '''
        crop是从整个时间序列中选择一个特定长度的子序列的过程
        上采样是指增加数据的采样点数，通常用于提高数据的时间分辨率
        原始的裁剪子序列 feature 在时间轴上有固定的点数，等于 self.cfg.duration
        上采样过程不是增加原始数据点，而是在现有数据点之间插入新的数据点，从而增加采样点的数量。这是通过插值算法实现的
        上采样的目的通常是为了使数据与特定的处理流程或模型结构兼容。
        '''
        feature = resize(
            feature,
            size=[self.num_features, self.upsampled_num_frames],
            antialias=False,
        ).squeeze(0)

        # from hard label to gaussian label
        num_frames = self.upsampled_num_frames // self.cfg.downsample_rate
        # 在创建标签时，需要一个与上采样后数据长度相匹配的标签数组。
        # 这个标签数组需要与数据的时间分辨率相对应，以便每个时间点都有一个标签
        # num_frames 就是这个标签数组的长度。
        #在保持数据详细度的同时，使其适应特定模型结构
        
        label = get_label(this_event_df, num_frames, self.cfg.duration, start, end)
        label[:, [1, 2]] = gaussian_label(
            label[:, [1, 2]], offset=self.cfg.offset, sigma=self.cfg.sigma
        )
        '''
        标签平滑的应用不仅限于单个事件（如入睡或醒来），而是同时考虑所有相关事件。
        通过应用高斯平滑（或高斯核卷积），我们能够使每个事件标签在其周围的时间点上展现出渐进的变化，而不是一个突然的跳变。
        这种方法更好地反映了现实世界中事件的逐渐变化特性，有助于提高模型的性能和泛化能力。
        模型能够更好地学习事件周围的上下文信息，而不是仅仅专注于特定的时间点。
        '''
        return {
            "series_id": series_id,
            "feature": feature,  # (num_features, upsampled_num_frames)
            "label": torch.FloatTensor(label),  # (pred_length, num_classes)
        }


class ValidDataset(Dataset):
    def __init__(
        self,
        cfg: DictConfig,
        chunk_features: dict[str, np.ndarray],
        event_df: pl.DataFrame,
    ):
        self.cfg = cfg
        self.chunk_features = chunk_features
        self.keys = list(chunk_features.keys())
        self.event_df = (
            event_df.pivot(index=["series_id", "night"], columns="event", values="step")
            .drop_nulls()
            .to_pandas()
        )
        self.num_features = len(cfg.features)
        self.upsampled_num_frames = nearest_valid_size(
            int(self.cfg.duration * self.cfg.upsample_rate), self.cfg.downsample_rate
        )

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        feature = self.chunk_features[key]
        feature = torch.FloatTensor(feature.T).unsqueeze(0)  # (1, num_features, duration)
        feature = resize(
            feature,
            size=[self.num_features, self.upsampled_num_frames],
            antialias=False,
        ).squeeze(0)

        series_id, chunk_id = key.split("_")
        chunk_id = int(chunk_id)
        start = chunk_id * self.cfg.duration
        end = start + self.cfg.duration
        num_frames = self.upsampled_num_frames // self.cfg.downsample_rate
        label = get_label(
            self.event_df.query("series_id == @series_id").reset_index(drop=True),
            num_frames,
            self.cfg.duration,
            start,
            end,
        )
        return {
            "key": key,
            "feature": feature,  # (num_features, duration)
            "label": torch.FloatTensor(label),  # (duration, num_classes)
        }


class TestDataset(Dataset):
    def __init__(
        self,
        cfg: DictConfig,
        chunk_features: dict[str, np.ndarray],
    ):
        self.cfg = cfg
        self.chunk_features = chunk_features
        self.keys = list(chunk_features.keys())
        self.num_features = len(cfg.features)
        self.upsampled_num_frames = nearest_valid_size(
            int(self.cfg.duration * self.cfg.upsample_rate), self.cfg.downsample_rate
        )

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        feature = self.chunk_features[key]
        feature = torch.FloatTensor(feature.T).unsqueeze(0)  # (1, num_features, duration)
        feature = resize(
            feature,
            size=[self.num_features, self.upsampled_num_frames],
            antialias=False,
        ).squeeze(0)

        return {
            "key": key,
            "feature": feature,  # (num_features, duration)
        }


###################
# DataModule
###################
class SegDataModule(LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.data_dir = Path(cfg.dir.data_dir)
        self.processed_dir = Path(cfg.dir.processed_dir)
        self.event_df = pl.read_csv(self.data_dir / "train_events.csv").drop_nulls()
        #polars 的缩写 polars可提供类似于 pandas 的 API 但在某些情况下速度更快
        self.train_event_df = self.event_df.filter(
            pl.col("series_id").is_in(self.cfg.split.train_series_ids)
        )
        #可在split处进行交叉验证 现在是固定划分
        self.valid_event_df = self.event_df.filter(
            pl.col("series_id").is_in(self.cfg.split.valid_series_ids)
        )
        # train data
        self.train_features = load_features(
            feature_names=self.cfg.features,
            series_ids=self.cfg.split.train_series_ids,
            processed_dir=self.processed_dir,
            phase="train",
        )

        # valid data
        self.valid_chunk_features = load_chunk_features(
            duration=self.cfg.duration,
            feature_names=self.cfg.features,
            series_ids=self.cfg.split.valid_series_ids,
            processed_dir=self.processed_dir,
            phase="train",
        )

    def train_dataloader(self):
        train_dataset = TrainDataset(
            cfg=self.cfg,
            event_df=self.train_event_df,
            features=self.train_features,
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        return train_loader

    def val_dataloader(self):
        valid_dataset = ValidDataset(
            cfg=self.cfg,
            chunk_features=self.valid_chunk_features,
            event_df=self.valid_event_df,
        )
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
        )
        return valid_loader
