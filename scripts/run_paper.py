"""Run every experiment of the paper in dependency order; safe to re-run after a crash or disconnect.

    python scripts/run_paper.py --plan configs/paper.yaml --dry-run          # list the steps
    python scripts/run_paper.py --plan configs/paper.yaml --stages splits seg
    python scripts/run_paper.py --plan configs/paper.yaml --set data.image_dir=/content/data/volumes

Each step is a subprocess call of one of the scripts in this folder. A finished step writes a marker
in ``<output_root>/.done/``, so re-running skips it; training scripts themselves resume from
``last.pt``. Use ``--redo STEP_ID`` to force a step again. ``aggregate`` and ``figures`` never get a
marker: they rerun every time they are selected, so they always reflect the finished runs. A step with a
``version`` reruns when its marker was written by an older version of that step. Its version names the
masks (cleanup and the backbone that predicted them) and, for classifiers, the backbone, so switching
``classification.backbone`` reruns everything downstream. A classifier folder trained for another version
is moved to ``<output_root>/.stale/`` instead of being resumed. Classifier evaluation also carries
``EVAL_VERSION``, so a change in how test predictions are scored re-evaluates without retraining; the
Grad-CAM panel likewise carries ``PANEL_VERSION``.

Stages: splits, ssl, seg, predict, features, cls, kfold, aggregate.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
STAGES = ["splits", "ssl", "seg", "predict", "features", "cls", "explain", "kfold", "aggregate", "figures"]
# Version of the predicted masks. Predicting and feature extraction rerun when their markers predate it.
MASKS_VERSION = "largest-component"
# Version of the classifier test evaluation. cls_eval reruns (the trained classifiers are kept) when it changes.
EVAL_VERSION = "LGG/HGG call from the grade call"
# Version of the Grad-CAM panel; gradcam_panel reruns when it changes.
PANEL_VERSION = "calls as evaluated (flip-averaged)"


@dataclass
class Step:
    id: str
    stage: str
    args: list[str]
    optional: bool = False
    needs: list[str] = field(default_factory=list)  # step ids that must have succeeded
    run_dir: Path | None = None                     # training steps: where last.pt lives
    always: bool = False                            # summary steps: rerun whenever selected, no marker
    version: str = ""                               # a marker without this version does not count as done


def _script(name: str) -> str:
    return str(HERE / name)


def is_done(marker: Path, step: Step) -> bool:
    return marker.exists() and (not step.version or f"version: {step.version}" in marker.read_text())


def set_aside_stale_run(step: Step, root: Path, dry_run: bool) -> None:
    """A run folder written for another version of its step (e.g. another backbone) must not be resumed."""
    tag = step.run_dir / ".step_version"
    if not step.run_dir.exists() or (tag.exists() and tag.read_text().strip() == step.version):
        return
    name = f"{step.run_dir.name}-{time.strftime('%Y%m%d-%H%M%S')}"
    old, i = root / ".stale" / name, 1
    while old.exists():
        old, i = root / ".stale" / f"{name}-{i}", i + 1
    print(f"      {step.run_dir.name} was trained for another version of this step: moving it to {old}")
    if not dry_run:
        old.parent.mkdir(parents=True, exist_ok=True)
        step.run_dir.rename(old)


def build_steps(plan: dict, cli_overrides: list[str]) -> tuple[list[Step], Path]:
    from glioma_mtl.config import load_config

    g = list(plan.get("overrides") or [])
    root = Path(load_config(REPO / plan["splits"]["config"], g + cli_overrides).get("output_root", "runs"))
    root_set = [f"output_root={root.as_posix()}"]

    def sets(stage_over, step_over=()):
        return ["--set", *g, *(stage_over or []), *cli_overrides, *root_set, *step_over]

    steps = [Step("splits", "splits", [_script("make_splits.py"), "--config", plan["splits"]["config"], *sets([])])]

    ssl = plan.get("ssl")
    if ssl:
        steps.append(Step("ssl", "ssl", [_script("pretrain_ssl.py"), "--config", ssl["config"],
                                         *sets(ssl.get("overrides"), ["name=ssl_swin_unetr"])], needs=["splits"],
                          run_dir=root / "ssl_swin_unetr"))
    ssl_ckpt = (root / "ssl_swin_unetr" / "best.pt").as_posix()

    seg = plan["segmentation"]
    seeds = seg.get("seeds", [42])
    for model, spec in seg["models"].items():
        for seed in seeds:
            name = f"{model}_s{seed}"
            extra = [f"seed={seed}", f"name={name}"]
            needs = ["splits"]
            if spec.get("needs_ssl"):
                extra.append(f"model.init.encoder={ssl_ckpt}")
                needs.append("ssl")
            run = (root / name).as_posix()
            steps.append(Step(f"seg_train:{name}", "seg", [_script("train_segmentation.py"), "--config", spec["config"],
                                                          *sets(seg.get("overrides"), extra)],
                              optional=spec.get("optional", False), needs=needs, run_dir=root / name))
            steps.append(Step(f"seg_eval:{name}", "seg", [_script("evaluate_segmentation.py"), "--config", spec["config"],
                                                         "--checkpoint", f"{run}/best.pt", "--stage", "segmentation",
                                                         "--split", "test", "--out-dir", run,
                                                         *sets(seg.get("overrides"), extra)],
                              optional=spec.get("optional", False), needs=[f"seg_train:{name}"]))

    cls = plan.get("classification")
    kfold = plan.get("kfold")
    if cls or kfold:
        bb_model = cls["backbone"]
        bb = f"{bb_model}_s{seeds[0]}"
        bb_spec = seg["models"][bb_model]
        bb_ckpt = (root / bb / "best.pt").as_posix()
        pred_dir = (root / "predictions" / bb).as_posix()
        masks = f"{MASKS_VERSION} from {bb}"  # anything that uses the predicted masks or features
        for part in ("train", "val", "test"):
            steps.append(Step(f"predict:{part}", "predict", [
                _script("evaluate_segmentation.py"), "--config", bb_spec["config"], "--checkpoint", bb_ckpt,
                "--stage", "classification", "--split", part, "--save-predictions", pred_dir, "--largest-component",
                "--out-dir", (root / bb).as_posix(), *sets(seg.get("overrides"), [f"name={bb}"])],
                needs=[f"seg_train:{bb}"], version=masks))
        predicted = [f"predict:{p}" for p in ("train", "val", "test")]
        fcfg = cls["feature_config"]
        deep_csv = (root / "deep_features" / "deep_features_pred.csv").as_posix()
        rad_csv = (root / "radiomics" / "radiomics_pred.csv").as_posix()
        steps.append(Step("features:deep", "features", [
            _script("extract_deep_features.py"), "--config", fcfg, "--checkpoint", bb_ckpt, "--mask-source", "pred",
            "--pred-dir", pred_dir, "--out-dir", (root / "deep_features").as_posix(), *sets([], ["name=deep_features"])],
            needs=predicted, version=masks))
        steps.append(Step("features:radiomics", "features", [
            _script("extract_radiomics.py"), "--config", fcfg, "--mask-source", "pred", "--pred-dir", pred_dir,
            "--out-dir", (root / "radiomics").as_posix(), *sets([], ["name=radiomics"])],
            optional=not (kfold or {}).get("use_radiomics"), needs=predicted, version=masks))

        for exp, spec in (cls.get("experiments") or {}).items():
            exp_seeds = spec.get("seeds", cls.get("seeds", [42]))
            # Classifier backbone: the shared one, or another segmentation model key of the same architecture as
            # the config's model.backbone. Crops are always centred on the shared backbone's predicted masks,
            # so all classifiers see the same crops.
            exp_bb = f"{spec.get('backbone', bb_model)}_s{seeds[0]}"
            exp_ckpt = (root / exp_bb / "best.pt").as_posix()
            trained = f"{masks}, backbone {exp_bb}"
            for seed in exp_seeds:
                name = f"{exp}_s{seed}"
                run = (root / name).as_posix()
                extra = [f"seed={seed}", f"name={name}", f"model.backbone_checkpoint={exp_ckpt}",
                         "data.crop_mask_key=pred_mask", f"data.pred_mask_dir={pred_dir}"]
                needs = ["predict:val", f"seg_train:{exp_bb}"]
                if spec.get("use_radiomics"):
                    extra.append(f"radiomics.csv={rad_csv}")
                    needs.append("features:radiomics")
                steps.append(Step(f"cls_train:{name}", "cls", [_script("train_classifier.py"), "--config", spec["config"],
                                                              *sets(cls.get("overrides"), extra)],
                                  optional=spec.get("optional", False), needs=needs, run_dir=root / name,
                                  version=trained))
                steps.append(Step(f"cls_eval:{name}", "cls", [_script("evaluate_classifier.py"), "--config",
                                                             f"{run}/config.yaml", "--checkpoint", f"{run}/best.pt",
                                                             "--split", "test", "--out-dir", run,
                                                             *sets([], ["data.crop_mask_key=pred_mask",
                                                                        f"data.pred_mask_dir={pred_dir}"])],
                                  optional=spec.get("optional", False), needs=[f"cls_train:{name}", "predict:test"],
                                  version=f"{trained}, eval {EVAL_VERSION}"))
            first = f"{exp}_s{exp_seeds[0]}"
            run = (root / first).as_posix()
            explain = [_script("explain_classifier.py"), "--config", f"{run}/config.yaml", "--checkpoint",
                       f"{run}/best.pt", "--split", "test", "--out-dir", f"{run}/explain",
                       *sets([], ["data.crop_mask_key=pred_mask", f"data.pred_mask_dir={pred_dir}"])]
            steps.append(Step(f"explain:{first}", "explain", explain, optional=True,
                              needs=[f"cls_train:{first}", "predict:test"], version=trained))
            # The paper's Grad-CAM figure (one patient per class); its own step, so it never reruns the above.
            steps.append(Step(f"gradcam_panel:{first}", "explain", [*explain, "--panel-only"], optional=True,
                              needs=[f"cls_train:{first}", "predict:test"], version=f"{trained}, panel {PANEL_VERSION}"))

    if kfold:
        extra = []
        needs = ["splits"]
        if kfold.get("use_deep_features"):
            extra.append(f"deep_features.csv={deep_csv}")
            needs.append("features:deep")
        if kfold.get("use_radiomics"):
            extra.append(f"radiomics.csv={rad_csv}")
            needs.append("features:radiomics")
        steps.append(Step("kfold", "kfold", [_script("kfold_cv.py"), "--config", kfold["config"],
                                             *(["--holdout"] if kfold.get("holdout") else []),
                                             "--out-dir", (root / "kfold_cv").as_posix(),
                                             *sets(kfold.get("overrides"), extra)], needs=needs,
                              version=masks if len(needs) > 1 else ""))

    steps.append(Step("aggregate", "aggregate", [_script("aggregate_results.py"), "--plan", "PLAN",
                                                 "--root", root.as_posix(), "--out", (root / "paper_tables").as_posix()],
                      always=True))
    steps.append(Step("figures", "figures", [_script("make_figures.py"), "--plan", "PLAN", "--root", root.as_posix(),
                                             "--out", (root / "paper_figures").as_posix(), "--qualitative"],
                      always=True))
    return steps, root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plan", default="configs/paper.yaml")
    parser.add_argument("--set", nargs="*", default=[], action="extend",
                        help="extra overrides applied to every step; may be repeated, later values win")
    parser.add_argument("--stages", nargs="*", default=STAGES, choices=STAGES)
    parser.add_argument("--only", nargs="*", default=None, help="run only these step ids")
    parser.add_argument("--redo", nargs="*", default=[], help="step ids to run again even if marked done")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    os.chdir(REPO)
    sys.path.insert(0, str(REPO / "src"))
    with open(args.plan, encoding="utf-8") as f:
        plan = yaml.safe_load(f)
    steps, root = build_steps(plan, args.set)
    done_dir = root / ".done"
    done_dir.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(REPO / "src"), os.environ.get("PYTHONPATH", "")])}

    status: dict[str, str] = {}
    for step in steps:
        marker = done_dir / (step.id.replace(":", "__") + ".done")
        selected = step.stage in args.stages and (args.only is None or step.id in args.only)
        # Tables and figures summarise whatever has finished so far, so a marker must never freeze them.
        if is_done(marker, step) and step.id not in args.redo and not step.always:
            status[step.id] = "done"
            continue
        if not selected:
            status[step.id] = "not selected"
            continue
        blocked = [n for n in step.needs if status.get(n) not in ("done", "ok")]
        if blocked:
            status[step.id] = f"blocked by {blocked}"
            print(f"[skip] {step.id}: waiting for {blocked}")
            continue
        cmd = [sys.executable, *[args.plan if a == "PLAN" else a for a in step.args]]
        print(f"\n[run] {step.id}\n      {' '.join(cmd[1:])}", flush=True)
        if step.run_dir and step.version:
            set_aside_stale_run(step, root, args.dry_run)
        if args.dry_run:
            status[step.id] = "ok"  # pretend, so the full dependency chain is listed
            continue
        if step.run_dir and step.version:
            step.run_dir.mkdir(parents=True, exist_ok=True)
            (step.run_dir / ".step_version").write_text(step.version + "\n")
        start = time.time()
        code = subprocess.call(cmd, env=env)
        minutes = (time.time() - start) / 60
        if code == 0:
            if not step.always:
                marker.write_text(f"{time.ctime()} ({minutes:.1f} min)\n"
                                  + (f"version: {step.version}\n" if step.version else ""))
            status[step.id] = "ok"
            print(f"[ok]  {step.id} in {minutes:.1f} min")
            # last.pt (weights + optimizer, ~750 MB for SwinUNETR) is only needed to resume an unfinished
            # run; best.pt is what later steps use. Dropping it keeps Google Drive usage to a few GB.
            last = step.run_dir / "last.pt" if step.run_dir else None
            if last and last.exists() and not plan.get("keep_last_checkpoints", False):
                last.unlink()
                print(f"      removed {last} (set keep_last_checkpoints: true to keep it)")
        else:
            status[step.id] = f"FAILED ({code})"
            # Keep going: steps that depend on this one are blocked through `needs`.
            print(f"[FAIL] {step.id} exited with {code} after {minutes:.1f} min"
                  + (" (optional)" if step.optional else ""))

    print("\nSummary:")
    for step in steps:
        print(f"  {status.get(step.id, 'not reached'):>24}  {step.id}")
    required_failed = [s.id for s in steps if not s.optional and status.get(s.id, "").startswith(("FAILED", "blocked"))]
    if required_failed:
        print(f"\nRequired steps not completed: {required_failed}")
        sys.exit(1)


if __name__ == "__main__":
    main()
