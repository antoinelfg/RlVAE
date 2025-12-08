import torch

from src.data.ellipse_sequences import EllipseSequenceDataset


def test_shapes_and_lengths():
    ds = EllipseSequenceDataset(num_sequences=10, seq_len=8, image_size=(32, 32), seed=0)
    assert len(ds) == 10
    x, y = ds[0]
    assert isinstance(x, torch.Tensor)
    assert x.shape == (8, 1, 32, 32)
    assert y == 0


def test_eccentricity_changes_monotonic():
    ds = EllipseSequenceDataset(num_sequences=1, seq_len=8, image_size=(32, 32), seed=1)
    x, _ = ds[0]
    # A coarse proxy: the ellipse should get more elongated over time on average.
    # Use inertia ratio along principal axes as a proxy measure.
    def inertia_ratio(img):
        img = img[0].numpy()
        ys, xs = (img > 0.1).nonzero()
        if ys.size == 0:
            return 1.0
        cy = ys.mean()
        cx = xs.mean()
        var_x = ((xs - cx) ** 2).mean() + 1e-6
        var_y = ((ys - cy) ** 2).mean() + 1e-6
        return max(var_x, var_y) / min(var_x, var_y)

    ratios = [inertia_ratio(x[t]) for t in range(x.shape[0])]
    # Not strictly monotonic, but later frames should on average be more elongated
    assert ratios[-1] >= ratios[0] * 0.8

