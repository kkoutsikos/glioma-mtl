"""Paired Wilcoxon signed-rank test between two models' per-case segmentation results.

    python scripts/compare_models.py runs/seg_segresnet/per_case_segmentation_test.csv \
        runs/seg_swin_unetr_ssl/per_case_segmentation_test.csv --metrics dice hd95
"""
import torch  # noqa: F401  (load before pandas; see src/glioma_mtl/__init__.py)
import argparse

import pandas as pd

from glioma_mtl.evaluation import paired_wilcoxon


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("a")
    parser.add_argument("b")
    parser.add_argument("--metrics", nargs="+", default=["dice", "hd95"])
    args = parser.parse_args()
    a, b = pd.read_csv(args.a), pd.read_csv(args.b)
    rows = [{"metric": m, **paired_wilcoxon(a, b, m)} for m in args.metrics]
    print(f"A = {args.a}\nB = {args.b}")
    print(pd.DataFrame(rows).to_string(index=False, float_format="%.4f"))


if __name__ == "__main__":
    main()
