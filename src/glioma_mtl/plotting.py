"""Figures for the paper. Uses matplotlib's object API (no pyplot), so it works headless and in subprocesses."""
from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.metrics import auc, roc_curve

SHORT_NAMES = {
    "Astrocytoma, IDH-mutant": "Astro IDH-mut",
    "Astrocytoma, IDH-wildtype": "Astro IDH-wt",
    "Glioblastoma, IDH-wildtype": "GBM IDH-wt",
    "Oligodendroglioma, IDH-mutant, 1p/19q-codeleted": "Oligo",
}


def short(names: Sequence) -> list[str]:
    return [SHORT_NAMES.get(str(n), str(n)) for n in names]


def _fit_titles(fig: Figure) -> bool:
    """Break titles wider than their room onto more lines, instead of letting the figure edge clip them.

    An axes title is centred on its axes, so its room is twice the distance from that centre to the nearer
    figure edge; the figure title has the whole width. Returns whether a title changed.
    """
    changed = False
    width = fig.bbox.width
    for text in [ax.title for ax in fig.axes] + [getattr(fig, "_suptitle", None)]:
        if text is None or not text.get_text():
            continue
        box = text.get_window_extent()
        center = (box.x0 + box.x1) / 2
        room = 0.95 * 2 * min(center, width - center)
        if box.width > room:
            lines = text.get_text().split("\n")
            chars = int(room / (box.width / max(len(line) for line in lines)))
            text.set_text("\n".join(textwrap.fill(line, max(chars, 10)) for line in lines))
            changed = True
    return changed


def _save(fig: Figure, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    if _fit_titles(fig):
        fig.tight_layout()
    fig.savefig(path, dpi=150)
    return path


def plot_history(history: pd.DataFrame | str | Path, out: str | Path, title: str | None = None) -> Path | None:
    """Loss curves (left) and validation metrics (right) from a run's history (``utils.read_history`` or a CSV)."""
    df = history if isinstance(history, pd.DataFrame) else pd.read_csv(history)
    if df.empty or "epoch" not in df:
        return None
    losses = [c for c in ("train_loss", "train_total", "val_loss", "val_total", "val_cls_nll") if c in df]
    metrics = [c for c in df if c.startswith("val_") and c not in losses
               and any(k in c for k in ("dice", "macro_f1", "bal_acc", "rot_acc"))]
    fig = Figure(figsize=(11, 4))
    ax_l, ax_m = fig.subplots(1, 2)
    for c in losses:
        d = df[["epoch", c]].dropna()
        ax_l.plot(d["epoch"], d[c], marker="." if len(d) < 30 else None, label=c)
    for c in metrics:
        d = df[["epoch", c]].dropna()
        ax_m.plot(d["epoch"], d[c], marker="." if len(d) < 30 else None, label=c)
    ax_l.set(xlabel="epoch", ylabel="loss", title="Loss")
    ax_m.set(xlabel="epoch", ylabel="score", title="Validation metrics")
    for ax in (ax_l, ax_m):
        if ax.lines:
            ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    if title:
        fig.suptitle(title)
    return _save(fig, out)


def plot_box(data: Mapping[str, Sequence[float]], out: str | Path, ylabel: str, title: str) -> Path:
    names = list(data)
    values = [np.asarray(pd.Series(data[n]).replace([np.inf, -np.inf], np.nan).dropna()) for n in names]
    fig = Figure(figsize=(1.6 * len(names) + 2, 4.5))
    ax = fig.subplots()
    ax.boxplot(values, showmeans=True)
    ax.set_xticks(range(1, len(names) + 1), names, rotation=25, ha="right")
    for i, v in enumerate(values, start=1):
        ax.scatter(np.random.default_rng(0).normal(i, 0.05, len(v)), v, s=6, alpha=0.4, color="k")
    ax.set(ylabel=ylabel, title=title)
    ax.grid(axis="y", alpha=0.3)
    return _save(fig, out)


def plot_confusion(cm: np.ndarray, classes: Sequence, out: str | Path, title: str) -> Path:
    classes = short(classes)
    rows = cm.sum(1, keepdims=True)
    frac = np.divide(cm, rows, out=np.zeros_like(cm, dtype=float), where=rows > 0)
    fig = Figure(figsize=(1.1 * len(classes) + 2.5, 1.0 * len(classes) + 2))
    ax = fig.subplots()
    im = ax.imshow(frac, cmap="Blues", vmin=0, vmax=1)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]}\n{frac[i, j]:.0%}", ha="center", va="center",
                    color="white" if frac[i, j] > 0.6 else "black", fontsize=8)
    ax.set_xticks(range(len(classes)), classes, rotation=30, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    ax.set(xlabel="predicted", ylabel="true", title=title)
    fig.colorbar(im, ax=ax, fraction=0.046)
    return _save(fig, out)


def plot_roc(y: np.ndarray, prob: np.ndarray, classes: Sequence, out: str | Path, title: str) -> Path:
    fig = Figure(figsize=(5, 5))
    ax = fig.subplots()
    pairs = [(1, short(classes)[1])] if prob.shape[1] == 2 else list(enumerate(short(classes)))
    for k, name in pairs:
        positives = (y == k)
        if positives.all() or not positives.any():
            continue  # class absent from this split
        fpr, tpr, _ = roc_curve(positives, prob[:, k])
        ax.plot(fpr, tpr, label=f"{name} (AUC {auc(fpr, tpr):.2f}, n={positives.sum()})")
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1)
    ax.set(xlabel="false positive rate", ylabel="true positive rate", title=title, xlim=(0, 1), ylim=(0, 1.02))
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(alpha=0.3)
    return _save(fig, out)


def plot_bars(labels: Sequence, mean: Sequence[float], sd: Sequence[float] | None, out: str | Path,
              title: str, xlabel: str) -> Path:
    labels, mean = list(labels), np.asarray(mean, dtype=float)
    sd = None if sd is None else np.nan_to_num(np.asarray(sd, dtype=float))
    fig = Figure(figsize=(7, 0.35 * len(labels) + 1.5))
    ax = fig.subplots()
    ax.barh(range(len(labels)), mean, xerr=sd, color="tab:blue", alpha=0.8, capsize=3)
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    ax.invert_yaxis()
    ax.axvline(0, color="k", lw=0.8)
    ax.set(xlabel=xlabel, title=title)
    ax.grid(axis="x", alpha=0.3)
    return _save(fig, out)


def plot_heatmap(matrix: np.ndarray, rows: Sequence[str], cols: Sequence[str], out: str | Path, title: str) -> Path:
    fig = Figure(figsize=(1.0 * len(cols) + 2.5, 0.7 * len(rows) + 2))
    ax = fig.subplots()
    im = ax.imshow(matrix, cmap="viridis", vmin=0, vmax=max(float(np.nanmax(matrix)), 1e-6))
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", color="w", fontsize=8)
    ax.set_xticks(range(len(cols)), cols, rotation=30, ha="right")
    ax.set_yticks(range(len(rows)), rows)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046)
    return _save(fig, out)


def _slices(volume: np.ndarray, center: Sequence[int]) -> list[np.ndarray]:
    x, y, z = center
    return [np.rot90(volume[:, :, z]), np.rot90(volume[:, y, :]), np.rot90(volume[x, :, :])]


def tumour_center(*masks: np.ndarray | None) -> list[int]:
    """Voxel where the first non-empty mask is thickest along each axis; the volume centre if all are empty."""
    for m in masks:
        if m is not None and m.any():
            return [int(np.argmax(m.sum(axis=tuple(a for a in range(3) if a != ax)))) for ax in range(3)]
    shape = next(m.shape for m in masks if m is not None)
    return [s // 2 for s in shape]


def plot_cam(image: np.ndarray, cam: np.ndarray, gt: np.ndarray | None, pred: np.ndarray | None,
             out: str | Path, title: str) -> Path:
    """Axial / coronal / sagittal slices through the tumour: image, Grad-CAM overlay, GT (green), prediction (red)."""
    center = tumour_center(gt, cam > 0.5)
    fig = Figure(figsize=(12, 4.4))
    axes = fig.subplots(1, 3)
    for ax, name, img, c, g, p in zip(axes, ("axial", "coronal", "sagittal"), _slices(image, center),
                                      _slices(cam, center), _slices(gt, center) if gt is not None else [None] * 3,
                                      _slices(pred, center) if pred is not None else [None] * 3):
        ax.imshow(img, cmap="gray")
        ax.imshow(c, cmap="jet", alpha=0.35, vmin=0, vmax=1)
        if g is not None and g.any():
            ax.contour(g, levels=[0.5], colors="lime", linewidths=1)
        if p is not None and p.any():
            ax.contour(p, levels=[0.5], colors="red", linewidths=1, linestyles="--")
        ax.set_title(name, fontsize=9)
        ax.axis("off")
    fig.suptitle(title + "\n(green: ground truth, red dashed: predicted tumour)", fontsize=9)
    return _save(fig, out)


def plot_cam_panel(cases: Sequence[Mapping], class_names: Sequence, layers: Sequence[str], out: str | Path,
                   title: str, image_name: str = "image") -> Path:
    """Rows = patients as axial slices through the tumour ('image', 'gt', 'seg', 'cams': {layer: slice}, 'y',
    'pred', 'p_pred'); columns = the image with its masks, then the Grad-CAM of each layer."""
    names = short(class_names)
    fig = Figure(figsize=(2.4 * (len(layers) + 1) + 0.8, 2.5 * len(cases) + 0.7))
    axes = fig.subplots(len(cases), len(layers) + 1, squeeze=False)
    for r, case in enumerate(cases):
        img, seg = np.rot90(case["image"]), np.rot90(case["seg"])
        gt = np.rot90(case["gt"]) if case["gt"] is not None else None
        for c, ax in enumerate(axes[r]):
            ax.imshow(img, cmap="gray")
            if c:
                ax.imshow(np.rot90(case["cams"][layers[c - 1]]), cmap="jet", alpha=0.4, vmin=0, vmax=1)
            if gt is not None and gt.any():
                ax.contour(gt, levels=[0.5], colors="lime", linewidths=1)
            if c == 0 and seg.any():
                ax.contour(seg, levels=[0.5], colors="red", linewidths=1, linestyles="--")
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(image_name if c == 0 else f"Grad-CAM, {layers[c - 1].replace('stage', 'stage ')}",
                             fontsize=9)
        wrong = "" if case["pred"] == case["y"] else "\nmisclassified"
        axes[r, 0].set_ylabel(f"true {names[case['y']]}\npred {names[case['pred']]} ({case['p_pred']:.2f}){wrong}",
                              fontsize=8)
    fig.suptitle(title + "\n(green: ground truth, red dashed: predicted tumour)", fontsize=9)
    return _save(fig, out)


def plot_seg_cases(cases: Sequence[Mapping], out: str | Path, title: str) -> Path:
    """Rows = cases ({'case', 'image', 'gt', 'preds': {model: mask}, 'dice': {model: value}}), columns = models."""
    models = list(cases[0]["preds"])
    fig = Figure(figsize=(2.6 * (len(models) + 1), 2.7 * len(cases)))
    axes = np.atleast_2d(fig.subplots(len(cases), len(models) + 1))
    for r, case in enumerate(cases):
        z = int(np.argmax(case["gt"].sum(axis=(0, 1)))) if case["gt"].any() else case["gt"].shape[2] // 2
        img, gt = np.rot90(case["image"][:, :, z]), np.rot90(case["gt"][:, :, z])
        axes[r, 0].imshow(img, cmap="gray")
        axes[r, 0].contour(gt, levels=[0.5], colors="lime", linewidths=1)
        axes[r, 0].set_title(f"{case['case']}\nground truth", fontsize=8)
        for c, model in enumerate(models, start=1):
            axes[r, c].imshow(img, cmap="gray")
            axes[r, c].contour(gt, levels=[0.5], colors="lime", linewidths=0.8)
            pred = np.rot90(case["preds"][model][:, :, z])
            if pred.any():
                axes[r, c].contour(pred, levels=[0.5], colors="red", linewidths=1)
            dice = case.get("dice", {}).get(model)
            axes[r, c].set_title(f"{model}" + (f"\nDice {dice:.3f}" if dice is not None else ""), fontsize=8)
        for ax in axes[r]:
            ax.axis("off")
    fig.suptitle(title + " (green: ground truth, red: prediction)", fontsize=10)
    return _save(fig, out)
