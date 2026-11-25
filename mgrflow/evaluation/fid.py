"""Pool-3 Fréchet Inception Distance, computed separately for each marker."""

import torch

from ..runtime import sum_ranks


class Moments:
    """Track mean and centered scatter without retaining the full feature matrix."""

    def __init__(self, device: torch.device, dimension: int = 2048):
        self.count = 0
        self.mean = torch.zeros(dimension, dtype=torch.float32, device=device)
        self.scatter = torch.zeros((dimension, dimension), dtype=torch.float32, device=device)

    def update(self, features: torch.Tensor) -> None:
        size = len(features)
        mean = features.mean(dim=0)
        centered = features - mean
        delta = mean - self.mean
        new_count = self.count + size
        self.scatter.add_(centered.T @ centered)
        self.scatter.add_(torch.outer(delta, delta), alpha=self.count * size / new_count)
        self.mean.add_(delta, alpha=size / new_count)
        self.count = new_count

    def finish(self) -> tuple[torch.Tensor, torch.Tensor] | None:
        count = int(sum_ranks(self.mean.new_tensor(self.count, dtype=torch.long)))
        if count < 2:
            return None
        mean = sum_ranks(self.mean * self.count) / count
        delta = self.mean - mean
        self.scatter.add_(torch.outer(delta, delta), alpha=self.count)
        covariance = sum_ranks(self.scatter) / (count - 1)
        return mean, covariance


def frechet(mean1, covariance1, mean2, covariance2) -> float:
    # The symmetric PSD expression allows a real-valued FP32 eigendecomposition.
    a = (covariance1 + covariance1.T) * 0.5
    b = (covariance2 + covariance2.T) * 0.5
    if torch.equal(a, b):
        return float((mean1 - mean2).square().sum())
    eigenvalues, basis = torch.linalg.eigh(a)
    root = (basis * eigenvalues.clamp_min(0).sqrt()) @ basis.T
    product = root @ b @ root
    eigenvalues = torch.linalg.eigvalsh((product + product.T) * 0.5)
    trace_root = eigenvalues.clamp_min(0).sqrt().sum()
    score = (mean1 - mean2).square().sum() + a.trace() + b.trace() - 2 * trace_root
    return float(score.clamp_min(0))


class MarkerFID:
    def __init__(self, keys: list[str], device: torch.device):
        from pytorch_fid.inception import InceptionV3

        self.inception = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).eval().to(device)
        self.keys = keys
        self.device = device
        self.predicted = {}
        self.measured = {}

    def features(self, images: torch.Tensor) -> torch.Tensor:
        rgb = images.float().clamp(0, 1)[:, None].expand(-1, 3, -1, -1).contiguous()
        return self.inception(rgb)[0].flatten(1)

    @torch.no_grad()
    def update(self, ids, predicted, measured) -> None:
        for marker in ids.unique().tolist():
            key = self.keys[marker]
            if key not in self.predicted:
                self.predicted[key] = Moments(self.device)
                self.measured[key] = Moments(self.device)
            group = ids == marker
            self.predicted[key].update(predicted[group])
            self.measured[key].update(measured[group])

    @torch.no_grad()
    def result(self, rank: int) -> dict:
        scores = {}
        for key in self.keys:
            predicted = self.predicted.pop(key, None) or Moments(self.device)
            measured = self.measured.pop(key, None) or Moments(self.device)
            first, second = predicted.finish(), measured.finish()
            if rank == 0:
                scores[key] = frechet(*first, *second) if first is not None else None
        return scores
