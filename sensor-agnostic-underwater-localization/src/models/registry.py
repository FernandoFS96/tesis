"""src/models/registry.py -- the ONE place that turns an arch name (or a saved
checkpoint) into a model.

The trainer and every eval script build models through here, so a new model
option is wired in once instead of once per script. (Separate copies of this
factory drifted twice: PFN ``readout`` and ``attn_ffn`` were not forwarded by the
eval copy, so those checkpoints could not be rebuilt.)

Unsupported option combinations raise instead of being silently ignored.
"""

from . import anp as anp_mod
from . import r_anp as ranp_mod
from . import online_r_anp as online_mod
from . import pfn as pfn_mod
from .frame import CentroidFrame

ARCHS = ("cnp", "anp", "pfn", "ranp", "rcnp", "online_ranp")
LATENT_ARCHS = ("anp", "ranp", "online_ranp")   # posterior + KL term


def is_latent(name):
    return name.lower() in LATENT_ARCHS


def build_model(name, num_hidden, input_dim, output_dim, *,
                rnn_type="lstm", rnn_layers=1, rnn_dropout=0.0, dropout=0.1,
                max_context=128, spatial_cfg=None,
                n_layers=6, n_heads=8, ffn_mult=4,
                readout="joint", n_cross_layers=2, full_cov=False,
                attn_ffn=False, latent_pooling=None, frame="absolute"):
    """Returns (model, convention). convention is how the trainer calls forward:
    'split' (context/target tensors), 'indexed' (full sequence + indices) or
    'online' (streaming). latent_pooling=None keeps each latent model's own
    default ('mean' for anp, 'attention' for ranp / online_ranp).
    frame='centroid' wraps the model in frame.CentroidFrame (labels relative to
    the array centroid inside the model; the calling convention is unchanged)."""
    frame = str(frame).lower()
    if frame not in ("absolute", "centroid"):
        raise ValueError(f"frame must be absolute|centroid, got {frame}")
    model, conv = _build(name, num_hidden, input_dim, output_dim, rnn_type=rnn_type,
                         rnn_layers=rnn_layers, rnn_dropout=rnn_dropout, dropout=dropout,
                         max_context=max_context, spatial_cfg=spatial_cfg,
                         n_layers=n_layers, n_heads=n_heads, ffn_mult=ffn_mult,
                         readout=readout, n_cross_layers=n_cross_layers,
                         full_cov=full_cov, attn_ffn=attn_ffn, latent_pooling=latent_pooling)
    if frame == "centroid":
        if not (spatial_cfg and spatial_cfg.get("enabled", False)) or conv == "online":
            raise ValueError("frame=centroid needs a spatial model (it reads sensor_pos)")
        model = CentroidFrame(model, output_dim)
    return model, conv


def _build(name, num_hidden, input_dim, output_dim, *, rnn_type, rnn_layers, rnn_dropout,
           dropout, max_context, spatial_cfg, n_layers, n_heads, ffn_mult, readout,
           n_cross_layers, full_cov, attn_ffn, latent_pooling):
    name = name.lower()
    if name not in ARCHS:
        raise ValueError(f"unknown model '{name}' (use {'|'.join(ARCHS)})")
    if latent_pooling is not None and not is_latent(name):
        raise ValueError(f"latent_pooling applies to latent models only, not '{name}'")
    pool_kw = {} if latent_pooling is None else {"latent_pooling": latent_pooling}

    if name == "pfn":  # masked single-stack transformer (spatial_cfg -> spatial_pfn)
        # attn_ffn is not an option here: every PFN block already has its FFN.
        if full_cov:
            raise ValueError("full_cov is not implemented for pfn")
        return pfn_mod.PFNModel(num_hidden, input_dim, output_dim, dropout=dropout,
                                spatial_cfg=spatial_cfg, n_layers=n_layers,
                                n_heads=n_heads, ffn_mult=ffn_mult,
                                readout=readout, n_cross_layers=n_cross_layers), "split"
    if name == "cnp":
        return anp_mod.DeterministicModel(num_hidden, input_dim, output_dim,
                                          dropout=dropout, spatial_cfg=spatial_cfg,
                                          full_cov=full_cov, attn_ffn=attn_ffn), "split"
    if name == "anp":
        return anp_mod.LatentModel(num_hidden, input_dim, output_dim,
                                   dropout=dropout, spatial_cfg=spatial_cfg,
                                   full_cov=full_cov, attn_ffn=attn_ffn, **pool_kw), "split"
    if name == "ranp":  # spatial_cfg enables the spatial_ranp front end
        return ranp_mod.LatentModel(num_hidden, input_dim, output_dim,
                                    rnn_type=rnn_type, rnn_layers=rnn_layers,
                                    rnn_dropout=rnn_dropout, dropout=dropout,
                                    spatial_cfg=spatial_cfg, full_cov=full_cov,
                                    attn_ffn=attn_ffn, **pool_kw), "indexed"
    if name == "rcnp":  # recurrent CNP (spatial_cfg -> spatial_rcnp)
        return ranp_mod.DeterministicModel(num_hidden, input_dim, output_dim,
                                           rnn_type=rnn_type, rnn_layers=rnn_layers,
                                           rnn_dropout=rnn_dropout, dropout=dropout,
                                           spatial_cfg=spatial_cfg, full_cov=full_cov,
                                           attn_ffn=attn_ffn), "indexed"
    # online_ranp: streaming / online-deployable latent RANP (a sketch; see its docstring)
    unsupported = [k for k, v in (("spatial", spatial_cfg), ("full_cov", full_cov),
                                  ("attn_ffn", attn_ffn)) if v]
    if unsupported:
        raise ValueError(f"online_ranp does not support: {', '.join(unsupported)}")
    if latent_pooling not in (None, "attention"):
        raise ValueError("online_ranp always uses attention latent pooling")
    return online_mod.OnlineLatentModel(num_hidden, input_dim, output_dim,
                                        rnn_type=rnn_type, rnn_layers=rnn_layers,
                                        rnn_dropout=rnn_dropout, dropout=dropout,
                                        max_context=max_context), "online"


def model_from_checkpoint(ck, device="cpu"):
    """Rebuild a model from a checkpoint written by train_np_geometry.py, load its
    weights strictly and put it in eval mode. Returns (model, convention).

    ck: the dict from torch.load(best.pt / last.pt)."""
    cfg = ck.get("config", {})
    name = ck.get("model_name", "cnp")
    sd = ck["model"]
    # attn_ffn entered the checkpoint config only after some runs were trained:
    # when it is absent, infer it from the weights (the '*.ffn.*' parameters exist
    # only when it was on; PFN blocks always have them, so PFN is excluded).
    attn_ffn = cfg.get("attn_ffn", None)
    if attn_ffn is None:
        attn_ffn = name != "pfn" and any(
            ".ffn." in k or k.endswith(".ffn_norm.weight") for k in sd)
    model, conv = build_model(
        name, cfg.get("num_hidden", 128), ck.get("feat_dim"), ck.get("out_dim", 3),
        rnn_type=cfg.get("rnn_type", "lstm"), rnn_layers=cfg.get("rnn_layers", 1),
        rnn_dropout=cfg.get("rnn_dropout", 0.0), dropout=cfg.get("dropout", 0.1),
        max_context=cfg.get("max_context", 128), spatial_cfg=cfg.get("spatial", None),
        n_layers=cfg.get("n_layers", 6), n_heads=cfg.get("n_heads", 8),
        ffn_mult=cfg.get("ffn_mult", 4), readout=cfg.get("readout", "joint"),
        n_cross_layers=cfg.get("n_cross_layers", 2),
        full_cov=cfg.get("full_cov", False), attn_ffn=bool(attn_ffn),
        latent_pooling=cfg.get("latent_pooling", None),
        frame=cfg.get("frame", "absolute"))
    model.to(device).load_state_dict(sd)
    # The variance floor is not in the state_dict: re-apply the one the run used.
    vf = cfg.get("var_floor_phys", None)
    if vf is not None:
        y_std = ck.get("y_std") if cfg.get("normalize_y", False) else None
        anp_mod.set_var_floor(model, anp_mod.physical_var_floor(vf, y_std))
    model.eval()
    return model, ck.get("convention") or conv
