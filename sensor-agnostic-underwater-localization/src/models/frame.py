"""src/models/frame.py -- run a model with its positions relative to the array
centroid (model.frame=centroid).

The channel depends only on source-sensor geometry, so where a deployment sits in
the data frame carries no information. A model that predicts absolute positions
must still learn that offset from sensor_pos; the per-point regressor that matches
the MFP on B1 (scripts/eval/acoustic_regressor.py) predicts relative to the array
centroid instead. CentroidFrame gives any model that frame without touching the
trainer's or the evaluators' calling convention: they keep passing labels in the
normalized absolute frame and get predictions back in it.

Inside, each label is shifted by the deployment's offset in model units,
(y_mean - centroid) / y_std, on x and y only. The scale is unchanged, so the
predictive variance, the variance floor and the NLL need no correction.
"""
import inspect

import torch as t
import torch.nn as nn


class CentroidFrame(nn.Module):
    def __init__(self, inner, output_dim):
        super().__init__()
        self.inner = inner
        # Normalization of the labels the caller passes (identity if unnormalized);
        # set by the trainer, restored from the checkpoint's state_dict.
        self.register_buffer("y_mean", t.zeros(output_dim))
        self.register_buffer("y_std", t.ones(output_dim))
        self._sig = inspect.signature(inner.forward)

    def set_y_stats(self, y_mean=None, y_std=None):
        if y_mean is not None:
            self.y_mean.copy_(t.as_tensor(y_mean, dtype=t.float32))
        if y_std is not None:
            self.y_std.copy_(t.as_tensor(y_std, dtype=t.float32))

    def __getattr__(self, name):
        # spatial_encoder and the other submodules the trainer and the evaluators
        # look up live on the wrapped model.
        try:
            return super().__getattr__(name)
        except AttributeError:
            if name == "inner" or name.startswith("_"):     # half-built (copy / pickle)
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
