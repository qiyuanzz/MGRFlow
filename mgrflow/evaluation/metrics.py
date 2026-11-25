"""FP32 paired image metrics and marker-macro reduction."""

import torch
from torch.nn import functional as F

from ..runtime import sum_ranks

IMAGE_METRICS = ("pearson", "ssim", "psnr", "spearman")


def pearson(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x, y = x.flatten(1).float(), y.flatten(1).float()
    varies = (x.amax(1) > x.amin(1)) & (y.amax(1) > y.amin(1))
    x = x - x.mean(1, keepdim=True)
    y = y - y.mean(1, keepdim=True)
    norm = (x.square().sum(1) * y.square().sum(1)).sqrt()
    score = (x * y).sum(1) / norm.clamp_min(torch.finfo(torch.float32).tiny)
    return score.clamp(-1, 1).masked_fill(~varies | (norm == 0), torch.nan)


def average_ranks(values: torch.Tensor) -> torch.Tensor:
    """Assign each tied run its mean rank."""
    sorted_values, order = values.sort(dim=1)
    positions = torch.arange(values.shape[1], dtype=torch.float32, device=values.device)
    boundaries = sorted_values[:, 1:] != sorted_values[:, :-1]
    starts = F.pad(boundaries, (1, 0), value=True)
    ends = F.pad(boundaries, (0, 1), value=True)
    first = torch.where(starts, positions, 0).cummax(1).values
    last = torch.where(ends, positions, values.shape[1] - 1).flip(1).cummin(1).values.flip(1)
    return torch.empty_like(values).scatter_(1, order, (first + last) * 0.5)


def ssim(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    positions = torch.arange(-5, 6, device=x.device, dtype=torch.float32)
    gaussian = torch.exp(-positions.square() / (2 * 1.5**2))
    gaussian /= gaussian.sum()
    kernel = torch.outer(gaussian, gaussian)[None, None]
    signals = torch.cat([x, y, x.square(), y.square(), x * y])[:, None]
    mx, my, ex2, ey2, exy = F.conv2d(signals, kernel)[:, 0].chunk(5)
    vx = (ex2 - mx.square()).clamp_min(0)
    vy = (ey2 - my.square()).clamp_min(0)
    cov = exy - mx * my
    difference = (2 * mx * my + 0.01**2) * (2 * cov + 0.03**2)
    normalization = (mx.square() + my.square() + 0.01**2) * (vx + vy + 0.03**2)
    return (difference / normalization).mean(dim=(-2, -1))


def patch_scores(predicted: torch.Tensor, measured: torch.Tensor) -> torch.Tensor:
    x, y = predicted.float(), measured.float()
    mse = (x - y).square().mean(dim=(-2, -1))
    psnr = -10 * mse.log10()
    rho = pearson(average_ranks(x.flatten(1)), average_ranks(y.flatten(1)))
    return torch.stack((pearson(x, y), ssim(x, y), psnr, rho), dim=1)


class MarkerAverages:
    def __init__(self, keys: list[str], metrics: tuple[str, ...], device: torch.device):
        self.keys, self.metrics = keys, metrics
        self.sums = torch.zeros((len(keys), len(metrics)), device=device)
        self.finite = torch.zeros_like(self.sums, dtype=torch.long)
        self.infinite = torch.zeros_like(self.finite)
        self.count = torch.zeros(len(keys), device=device, dtype=torch.long)

    def update(self, ids: torch.Tensor, scores: torch.Tensor) -> None:
        finite = torch.isfinite(scores)
        self.sums.index_add_(0, ids, scores.masked_fill(~finite, 0))
        self.finite.index_add_(0, ids, finite.long())
        self.infinite.index_add_(0, ids, torch.isposinf(scores).long())
        self.count.index_add_(0, ids, torch.ones_like(ids))

    def result(self) -> dict:
        per_marker = {key: {} for key in self.keys}
        count = sum_ranks(self.count).tolist()
        avg = sum_ranks(self.sums) / sum_ranks(self.finite)
        avg = avg.masked_fill(sum_ranks(self.infinite) > 0, torch.inf)
        macro = {}
        for col, name in enumerate(self.metrics):
            values = avg[:, col]
            macro[name] = float(values.nanmean()) if values.isfinite().any() else None
            if values.isposinf().any():
                macro[name] = None
                macro[f"{name}_status"] = "positive_infinity"
            for key, value in zip(self.keys, values.tolist(), strict=True):
                per_marker[key][name] = value if -float("inf") < value < float("inf") else None
                if value == float("inf"):
                    per_marker[key][f"{name}_status"] = "positive_infinity"
        for key, n in zip(self.keys, count, strict=True):
            per_marker[key]["pairs"] = n
        return {"marker_macro": macro, "per_marker": per_marker, "evaluated_pairs": sum(count)}
