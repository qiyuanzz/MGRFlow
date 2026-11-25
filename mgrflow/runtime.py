"""Small helpers shared by training and evaluation."""

import os

import torch
import torch.distributed as dist
from torch.utils.data import Sampler


def initialize(device_name: str = "auto") -> tuple[torch.device, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    if device.type == "cuda":
        if device.index is None:
            device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
        torch.cuda.set_device(device)
        torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    if world > 1:
        dist.init_process_group(backend="nccl" if device.type == "cuda" else "gloo")
    return device, rank, world


def sum_ranks(value: torch.Tensor) -> torch.Tensor:
    if dist.is_initialized():
        dist.all_reduce(value)
    return value


def finalize() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def flow_precision(device: torch.device) -> torch.dtype:
    return (
        torch.bfloat16
        if device.type == "cuda" and torch.cuda.is_bf16_supported()
        else torch.float32
    )


class TrainingTiles(Sampler):
    """Shuffle all tiles once. Zero-weight padding keeps DDP ranks in sync."""

    def __init__(self, size: int, rank: int, world: int, seed: int):
        self.size, self.rank, self.world, self.seed = size, rank, world, seed
        self.epoch = 0

    def __len__(self) -> int:
        return (self.size + self.world - 1) // self.world

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        order = torch.randperm(self.size, generator=generator).tolist()
        for position in range(self.rank, len(self) * self.world, self.world):
            yield order[position % self.size], position < self.size
