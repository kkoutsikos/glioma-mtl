"""Explanations for :class:`MultiTaskGliomaNet`: fusion attention, token (modality) occlusion, 3D Grad-CAM.

* **Fusion attention**: attention of each task token over the image / clinical / radiomics tokens,
  recomputed layer by layer from the trained weights (the thesis did the same by hand; here a unit
  test checks that the recomputation reproduces the model's own output).
* **Token occlusion**: drop in the true-class probability when one context token is zeroed, the same
  operation as modality dropout during training. Works for every fusion type, including ``concat``.
* **Grad-CAM**: gradient-weighted activation map of a chosen feature map (an encoder stage or the decoder
  features the classifier pools), upsampled to the crop. Raises if no gradient reaches that map
  instead of returning an empty heatmap.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .models.classification import MultiTaskGliomaNet


@contextmanager
def _unfrozen(model: MultiTaskGliomaNet):
    """Grad-CAM needs a graph through the backbone even when it was frozen during training."""
    frozen = model.freeze_backbone
    model.freeze_backbone = False
    try:
        yield
    finally:
        model.freeze_backbone = frozen


@torch.no_grad()
def fusion_attention(model: MultiTaskGliomaNet, context: torch.Tensor):
    """Per fusion layer, the attention of the task tokens: list of [B, n_tasks, n_keys], and the fused tokens.

    Keys are ``task tokens + context tokens`` for ``transformer`` fusion and the context tokens for
    ``cross_attention``. Returns ``(None, None)`` for ``concat`` (no attention).
    """
    n = len(model.tasks)
    x = model.task_tokens.expand(context.shape[0], -1, -1)
    maps = []
    if model.fusion == "transformer":
        x = torch.cat([x, context], dim=1)
        for layer in model.fuser.layers:
            q = layer.norm1(x) if layer.norm_first else x
            maps.append(layer.self_attn(q, q, q, need_weights=True, average_attn_weights=True)[1][:, :n])
            x = layer(x)
        return maps, x[:, :n]
    if model.fusion == "cross_attention":
        for layer in model.fuser.layers:
            t = layer.norm1(x) if layer.norm_first else x
            sa = layer.dropout1(layer.self_attn(t, t, t, need_weights=False)[0])
            x_sa = x + sa if layer.norm_first else layer.norm1(x + sa)
            q = layer.norm2(x_sa) if layer.norm_first else x_sa
            maps.append(layer.multihead_attn(q, context, context, need_weights=True, average_attn_weights=True)[1])
            x = layer(x, context)
        return maps, x
    return None, None


def attention_keys(model: MultiTaskGliomaNet) -> list[str]:
    tokens = model.token_names
    return [f"task_{t}" for t in model.tasks] + tokens if model.fusion == "transformer" else tokens


@torch.no_grad()
def token_occlusion(model: MultiTaskGliomaNet, enc: Mapping) -> tuple[dict, dict]:
    """Class probabilities with all tokens ``{task: [B, k]}`` and with each token zeroed ``{task: [B, n_ctx, k]}``."""
    it, tt = enc["image_tokens"], enc["tab_tokens"]
    base, _ = model.fuse(model.context(it, tt))
    dropped = [model.fuse(model.context(it, tt, drop=[i]))[0] for i in range(len(it) + len(tt))]
    return ({t: torch.softmax(v.float(), 1) for t, v in base.items()},
            {t: torch.stack([torch.softmax(d[t].float(), 1) for d in dropped], 1) for t in model.tasks})


def cam_layers(model: MultiTaskGliomaNet) -> list[str]:
    """The feature maps the classifier pools, i.e. the ones Grad-CAM can meaningfully explain."""
    return [f"stage{s}" for s in model.encoder_stages] + (["decoder"] if model.use_decoder_token else [])


def grad_cam(model: MultiTaskGliomaNet, image: torch.Tensor, tabular: Mapping | None, task: str,
             layer: str = "decoder", target: int | None = None, mask: torch.Tensor | None = None):
    """3D Grad-CAM for one sample (batch of 1). Returns ``(cam [D,H,W] in [0, 1], probs, class)``."""
    model.eval()
    # A frozen backbone has requires_grad=False weights; an input that requires grad still gives the
    # activations a graph, so Grad-CAM works without touching the weights.
    image = image.detach().requires_grad_(True)
    with _unfrozen(model), torch.enable_grad():
        enc = model.encode(image, tabular, mask)
        fmap = enc["feats"].decoder if layer == "decoder" else enc["feats"].hidden[int(layer.removeprefix("stage"))]
        fmap.retain_grad()
        logits, _ = model.fuse(model.context(enc["image_tokens"], enc["tab_tokens"]))
        scores = logits[task][0]
        cls = int(scores.argmax()) if target is None else int(target)
        model.zero_grad(set_to_none=True)
        scores[cls].backward()
        if fmap.grad is None or not torch.any(fmap.grad):
            raise RuntimeError(f"no gradient reaches {layer} from the {task} logit; choose one of {cam_layers(model)}")
        weights = fmap.grad.mean(dim=(2, 3, 4), keepdim=True)
        cam = F.relu((weights * fmap).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=image.shape[2:], mode="trilinear", align_corners=False)[0, 0]
        cam = cam / cam.max().clamp_min(1e-8)
    return cam.detach().cpu().numpy(), torch.softmax(scores.detach().float(), 0).cpu().numpy(), cls


def explain_classifier(model: MultiTaskGliomaNet, loader, device: torch.device, class_names: Mapping[str, Sequence],
                       out_dir: str | Path, mask_key: str = "label", gradcam: bool = True) -> dict[str, pd.DataFrame]:
    """Attention, occlusion and Grad-CAM over a (test) loader with batch size 1. Writes CSVs and figures."""
    from . import plotting

    out_dir = Path(out_dir)
    model.eval().to(device)
    keys = attention_keys(model)
    attn_rows, occ_rows = [], []
    for batch in tqdm(loader, desc="explain", leave=False):
        image = batch["image"].to(device)
        tabular = {n: batch[f"tab_{n}"].to(device) for n in model.tabular_dims}
        mask = batch[mask_key].to(device) if model.mask_mode == "input" else None
        pid = batch["patient_id"][0]
        with torch.no_grad():
            enc = model.encode(image, tabular, mask)
            maps, _ = fusion_attention(model, model.context(enc["image_tokens"], enc["tab_tokens"]))
            base, dropped = token_occlusion(model, enc)
        for t_idx, task in enumerate(model.tasks):
            y = int(torch.as_tensor(batch[f"y_{task}"]).view(-1)[0])
            pred = int(base[task][0].argmax())
            if maps is not None:
                for layer, w in enumerate(maps):
                    attn_rows += [{"patient_id": pid, "task": task, "layer": layer, "key": k, "attention": float(v)}
                                  for k, v in zip(keys, w[0, t_idx].cpu())]
            if y >= 0:
                p_full = float(base[task][0, y])
                occ_rows += [{"patient_id": pid, "task": task, "token": name, "y": y, "pred": pred, "p_true": p_full,
                              "p_true_without": float(dropped[task][0, i, y]),
                              "importance": p_full - float(dropped[task][0, i, y])}
                             for i, name in enumerate(model.token_names)]
            if gradcam:
                seg = (torch.sigmoid(enc["feats"].logits.float()) > 0.5)[0, 0].cpu().numpy()
                gt = batch["label"][0, 0].cpu().numpy() if "label" in batch else None
                for layer in cam_layers(model):
                    cam, probs, cls = grad_cam(model, image, tabular, task, layer, mask=mask)
                    title = (f"{pid} | {task}: true {class_names[task][y] if y >= 0 else '?'}, "
                             f"pred {class_names[task][cls]} ({probs[cls]:.2f}) | Grad-CAM {layer}")
                    plotting.plot_cam(image[0, 0].cpu().numpy(), cam, gt, seg,
                                      out_dir / "gradcam" / f"{pid}_{task}_{layer}.png", title)

    results = {}
    if occ_rows:
        occ = pd.DataFrame(occ_rows)
        occ.to_csv(out_dir / "occlusion.csv", index=False)
        summary = occ.groupby(["task", "token"])["importance"].agg(["mean", "std", "count"]).reset_index()
        summary.to_csv(out_dir / "occlusion_summary.csv", index=False)
        for task, sub in summary.groupby("task"):
            plotting.plot_bars(sub["token"], sub["mean"], sub["std"], out_dir / f"occlusion_{task}.png",
                               f"Token occlusion ({task}): drop in true-class probability", "Δ p(true class)")
        results["occlusion"] = occ
    if attn_rows:
        attn = pd.DataFrame(attn_rows)
        attn.to_csv(out_dir / "attention.csv", index=False)
        mean = attn.groupby(["layer", "task", "key"], sort=False)["attention"].mean().reset_index()
        for layer, sub in mean.groupby("layer"):
            mat = sub.pivot(index="task", columns="key", values="attention").reindex(index=list(model.tasks), columns=keys)
            plotting.plot_heatmap(mat.values, list(mat.index), list(mat.columns), out_dir / f"attention_layer{layer}.png",
                                  f"Mean fusion attention, layer {layer} (n={attn['patient_id'].nunique()})")
        results["attention"] = attn
    return results


def panel_patients(calls: Sequence[Mapping], n_classes: int) -> list[Mapping]:
    """One patient per true class: the correctly called one with the highest probability or, for a class with no
    correct call, the one with the highest probability for its true class (shown as misclassified)."""
    picked = []
    for c in range(n_classes):
        of_class = [k for k in calls if k["y"] == c]
        if of_class:
            correct = [k for k in of_class if k["pred"] == c]
            picked.append(max(correct or of_class, key=lambda k: k["p_true"]))
    return picked


def gradcam_panel(model: MultiTaskGliomaNet, loader, device: torch.device, class_names: Mapping[str, Sequence],
                  out_dir: str | Path, mask_key: str = "label", image_name: str = "image") -> list[Path]:
    """One paper figure per task, ``gradcam_panel_<task>.png``: a row per class (see ``panel_patients``), the axial
    slice through the tumour with the Grad-CAM of every feature map the classifier pools.

    Two passes over the loader (batch size 1): predictions for every patient, then Grad-CAM for the picked ones.
    """
    from . import plotting

    model.eval().to(device)

    def inputs(batch):
        image = batch["image"].to(device)
        tabular = {n: batch[f"tab_{n}"].to(device) for n in model.tabular_dims}
        return image, tabular, batch[mask_key].to(device) if model.mask_mode == "input" else None

    calls: dict[str, list] = {t: [] for t in model.tasks}
    for batch in tqdm(loader, desc="panel: predict", leave=False):
        image, tabular, mask = inputs(batch)
        with torch.no_grad():
            enc = model.encode(image, tabular, mask)
            logits, _ = model.fuse(model.context(enc["image_tokens"], enc["tab_tokens"]))
        for task in model.tasks:
            y = int(torch.as_tensor(batch[f"y_{task}"]).view(-1)[0])
            if y >= 0:
                p = torch.softmax(logits[task][0].float(), 0).cpu().numpy()
                calls[task].append({"patient_id": batch["patient_id"][0], "y": y, "pred": int(p.argmax()),
                                    "p_true": float(p[y]), "p_pred": float(p.max())})
    picked = {t: panel_patients(calls[t], len(class_names[t])) for t in model.tasks}

    layers = cam_layers(model)
    slices = {}
    for batch in tqdm(loader, desc="panel: Grad-CAM", leave=False):
        pid = batch["patient_id"][0]
        tasks = [t for t, rows in picked.items() if any(k["patient_id"] == pid for k in rows)]
        if not tasks:
            continue
        image, tabular, mask = inputs(batch)
        with torch.no_grad():
            logits = model.encode(image, tabular, mask)["feats"].logits
        seg = (torch.sigmoid(logits.float()) > 0.5)[0, 0].cpu().numpy()
        gt = batch["label"][0, 0].cpu().numpy() if "label" in batch else None
        z = plotting.tumour_center(gt, seg)[2]
        for task in tasks:
            slices[pid, task] = {"image": image[0, 0, :, :, z].cpu().numpy(), "seg": seg[:, :, z],
                                 "gt": None if gt is None else gt[:, :, z],
                                 "cams": {layer: grad_cam(model, image, tabular, task, layer, mask=mask)[0][:, :, z]
                                          for layer in layers}}
    return [plotting.plot_cam_panel([{**k, **slices[k["patient_id"], task]} for k in rows], class_names[task],
                                    layers, Path(out_dir) / f"gradcam_panel_{task}.png",
                                    f"{task}: Grad-CAM, the most confidently correct test patient of each class "
                                    "(a class with no correct call: its most probable patient)",
                                    image_name)
            for task, rows in picked.items() if rows]
