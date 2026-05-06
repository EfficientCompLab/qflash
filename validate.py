#!/usr/bin/env python3
import argparse
import json
import logging
import os
import time
from argparse import Namespace
from pathlib import Path

import torch
from timm.data import create_dataset, create_loader, resolve_data_config
from timm.models import create_model
from timm.utils import accuracy, AverageMeter, setup_default_logging

from qflash import patch_attention

MODELS = {
    "ViT-S":  "vit_small_patch16_224.augreg_in21k_ft_in1k",
    "ViT-B":  "vit_base_patch16_224.augreg2_in21k_ft_in1k",
    "DeiT-T": "deit_tiny_patch16_224.fb_in1k",
    "DeiT-S": "deit_small_patch16_224.fb_in1k",
    "DeiT-B": "deit_base_patch16_224.fb_in1k",
    "Swin-T": "swin_tiny_patch4_window7_224.ms_in1k",
    "Swin-S": "swin_small_patch4_window7_224.ms_in1k",
}

DEFAULT_DATA_DIR = os.path.expanduser("~/dataset/imagenet")

_logger = logging.getLogger("validate")


def build_parser():
    p = argparse.ArgumentParser(description="ImageNet Top-1 evaluation with QFlash attention")
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR,
                   help=f"ImageNet root (default: {DEFAULT_DATA_DIR})")
    p.add_argument("-m", "--model", default=None, metavar="NAME",
                   help=f"timm name or paper label ({', '.join(MODELS)})")
    p.add_argument("--all", action="store_true",
                   help="Sweep all paper models")
    p.add_argument("-b", "--batch-size", default=256, type=int)
    p.add_argument("-j", "--workers", default=8, type=int)
    p.add_argument("--num-samples", default=None, type=int, metavar="N",
                   help="Evenly-spaced subset (sanity check; not paper-comparable)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--log-freq", default=50, type=int)
    p.add_argument("--results-dir", default=None,
                   help="Write per-model JSON + summary.csv (sweep default: results/)")
    return p


def evaluate(args):
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    device = torch.device(args.device)
    model = create_model(args.model, pretrained=True)
    counts = patch_attention(model)
    _logger.info(f"patch_attention: QAttention={counts['attn']}, QWindowAttention={counts['wattn']}")

    param_count = sum(p.numel() for p in model.parameters())
    _logger.info(f"{args.model}  params={param_count/1e6:.2f}M")

    data_config = resolve_data_config(vars(args), model=model, use_test_size=True, verbose=True)
    model = model.to(device).eval()

    dataset = create_dataset(root=args.data_dir, name="", split="validation", download=False)

    if args.num_samples is not None and args.num_samples < len(dataset):
        from torch.utils.data import Subset
        step = max(1, len(dataset) // args.num_samples)
        indices = list(range(0, len(dataset), step))[:args.num_samples]
        dataset = Subset(dataset, indices)
        _logger.info(f"Subset: {len(dataset)} images (step={step})")

    loader = create_loader(
        dataset,
        input_size=data_config["input_size"],
        batch_size=args.batch_size,
        use_prefetcher=True,
        interpolation=data_config["interpolation"],
        mean=data_config["mean"],
        std=data_config["std"],
        num_workers=args.workers,
        crop_pct=data_config["crop_pct"],
        crop_mode=data_config.get("crop_mode"),
        device=device,
    )
    # Subset doesn't propagate transform to its inner dataset; forward it manually.
    if isinstance(dataset, torch.utils.data.Subset):
        dataset.dataset.transform = dataset.transform

    batch_time, top1, top5 = AverageMeter(), AverageMeter(), AverageMeter()

    with torch.inference_mode():
        warm = torch.randn((args.batch_size,) + tuple(data_config["input_size"]), device=device)
        model(warm)
        del warm

        end = time.time()
        for batch_idx, (input, target) in enumerate(loader):
            output = model(input)
            acc1, acc5 = accuracy(output.detach(), target, topk=(1, 5))
            bs = output.shape[0]
            top1.update(acc1.item(), bs)
            top5.update(acc5.item(), bs)
            batch_time.update(time.time() - end)
            end = time.time()

            if batch_idx % args.log_freq == 0:
                _logger.info(
                    f"[{batch_idx:>4d}/{len(loader)}]  "
                    f"{bs/batch_time.avg:>6.1f} img/s  "
                    f"Acc@1 {top1.avg:>6.3f}  Acc@5 {top5.avg:>6.3f}"
                )

    _logger.info(f" * Acc@1 {top1.avg:.3f}  Acc@5 {top5.avg:.3f}")
    return {
        "model": args.model,
        "top1": round(top1.avg, 4),
        "top5": round(top5.avg, 4),
        "param_count": round(param_count / 1e6, 2),
        "img_size": data_config["input_size"][-1],
        "crop_pct": data_config["crop_pct"],
        "interpolation": data_config["interpolation"],
    }


def _resolve(model_arg):
    if model_arg in MODELS:
        return model_arg, MODELS[model_arg]
    return model_arg, model_arg


def main():
    setup_default_logging()
    args = build_parser().parse_args()

    if args.all:
        targets = list(MODELS.items())
    elif args.model:
        targets = [_resolve(args.model)]
    else:
        raise SystemExit("--model NAME or --all required")

    results_dir = Path(args.results_dir) if args.results_dir else (Path("results") if args.all else None)
    if results_dir:
        results_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for label, name in targets:
        _logger.info(f"=== {label} ({name}) ===")
        kw = vars(args).copy()
        kw["model"] = name
        kw.pop("all", None)
        r = evaluate(Namespace(**kw))
        r["label"] = label
        rows.append(r)
        if results_dir:
            (results_dir / f"{label}.json").write_text(json.dumps(r, indent=2))

    if len(rows) == 1 and not results_dir:
        print(json.dumps(rows[0], indent=2))
        return

    print()
    print(f"{'Label':<8} {'Model':<48} {'Top-1':>8} {'Top-5':>8}")
    print("-" * 76)
    for r in rows:
        print(f"{r['label']:<8} {r['model']:<48} {r['top1']:>8.3f} {r['top5']:>8.3f}")

    if results_dir:
        csv = results_dir / "summary.csv"
        with csv.open("w") as f:
            f.write("label,model,top1,top5,param_count_M\n")
            for r in rows:
                f.write(f"{r['label']},{r['model']},{r['top1']},{r['top5']},{r['param_count']}\n")
        _logger.info(f"Summary -> {csv}")


if __name__ == "__main__":
    main()
