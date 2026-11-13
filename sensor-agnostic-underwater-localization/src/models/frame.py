"""Positions relative to the array centroid (model.frame=centroid).

CentroidFrame wraps a model so that, internally, labels are expressed relative to
the centroid of the deployment's sensors. Callers still pass and receive labels
in the normalized absolute frame. The shift, (y_mean - centroid) / y_std, is
applied to x and y only and leaves the scale unchanged, so the predictive
variance and the NLL need no correction.
"""
import inspect

import torch as t
import torch.nn as nn


class CentroidFrame(nn.Module):
    def __init__(self, inner, output_dim):
        super().__init__()
        self.inner = inner
        # Label normalization used by the caller; set by the trainer.
        self.register_buffer("y_mean", t.zeros(output_dim))
        self.register_buffer("y_std", t.ones(output_dim))
        self._sig = inspect.signature(inner.forward)

    def set_y_stats(self, y_mean=None, y_std=None):
        if y_mean is not None:
            self.y_mean.copy_(t.as_tensor(y_mean, dtype=t.float32))
        if y_std is not None:
            self.y_std.copy_(t.as_tensor(y_std, dtype=t.float32))

    def __getattr__(self, name):
        # Fall through to the wrapped model (e.g. spatial_encoder).
        try:
            return super().__getattr__(name)
        except AttributeError:
            if name == "inner" or name.startswith("_"):
                raise
            return getattr(super().__getattr__("inner"), name)

    def forward(self, *args, **kwargs):
        a = self._sig.bind(*args, **kwargs)
        sp = a.arguments.get("sensor_pos")
        if sp is None:
            raise ValueError("model.frame=centroid needs sensor_pos")
        c = sp[..., :2].mean(1)                                      # (B, 2), all sensors
        off = ((self.y_mean[:2] - c) / self.y_std[:2])[:, None]      # (B, 1, 2)
        for k in ("context_y", "target_y"):
            y = a.arguments.get(k)
            if y is not None:
                y = y.clone()
                y[..., :2] = y[..., :2] + off
                a.arguments[k] = y
        mean, *rest = self.inner(*a.args, **a.kwargs)
        mean = mean.clone()
        mean[..., :2] = mean[..., :2] - off
        return (mean, *rest)
