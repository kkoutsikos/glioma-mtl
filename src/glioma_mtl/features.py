"""Label-free deep features for late fusion (XGBoost etc.).

Features come from the *segmentation* SwinUNETR (trained with tumour masks only, never with grade /
diagnosis labels), pooled inside the tumour mask at selected encoder stages. Because no class label was
used to produce them, they can be cross-validated over the classification cohort without fold leakage.
(The thesis took embeddings from the fine-tuned classifier, which had been trained on the same patients'
labels, so every evaluation of those features was in-sample.)
"""
from __future__ import annotations

from typing import Sequence

import pandas as pd
import torch
from tqdm import tqdm

from .models.backbone import swin_unetr_forward
from .models.classification import masked_mean


@torch.no_grad()
def extract_deep_features(model, loader, device: torch.device, stages: Sequence[int] = (2, 3, 4),
                          use_decoder: bool = True, mask_source: str = "pred", amp: bool = True) -> pd.DataFrame:
    """One row per patient; columns ``deep_s<stage>_<i>`` and ``deep_dec_<i>``.

    ``mask_source='pred'`` pools inside the model's own predicted tumour probability (deployable);
    ``'label'`` pools inside the ground-truth mask (oracle).
    """
    model.eval().to(device)
    rows = []
    for batch in tqdm(loader, desc="deep features", leave=False):
        image = batch["image"].to(device)
        with torch.autocast(device.type, enabled=amp and device.type == "cuda"):
            out = swin_unetr_forward(model, image)
        mask = torch.sigmoid(out.logits.float()) if mask_source == "pred" else batch["label"].to(device).float()
        blocks = {f"s{s}": masked_mean(out.hidden[s].float(), mask) for s in stages}
        if use_decoder:
            blocks["dec"] = masked_mean(out.decoder.float(), mask)
        for i, pid in enumerate(batch["patient_id"]):
            row = {"patient_id": pid}
            for name, vec in blocks.items():
                row.update({f"deep_{name}_{j}": float(v) for j, v in enumerate(vec[i].cpu())})
            rows.append(row)
    return pd.DataFrame(rows).set_index("patient_id")


def load_deep_features(path) -> pd.DataFrame:
    return pd.read_csv(path).set_index("patient_id")
