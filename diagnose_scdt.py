"""Diagnose trained SCDT routing without modifying the model or retraining."""
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import torch

from ultralytics import YOLO
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.nn.modules.improve.scdt import SCDT

COLLECTOR = None


def _mean(total, count):
    return total / count if count else float("nan")


def _fmt(x):
    return "nan" if math.isnan(x) else f"{x:.6f}"


class RunningStats:
    def __init__(self):
        self.r = {"fg": self._new(), "bg": self._new()}

    @staticmethod
    def _new():
        return {
            "count": 0,
            "entropy_sum": 0.0,
            "max_prob_sum": 0.0,
            "residual_ratio_sum": 0.0,
            "residual_rms_sum": 0.0,
            "retrieved_rms_sum": 0.0,
            "winner_count": [0, 0, 0, 0],
        }

    def update(self, name, mask, entropy, max_prob, residual_ratio, residual_rms, retrieved_rms, winner):
        dst = self.r[name]
        n = int(mask.sum().item())
        if n == 0:
            return
        dst["count"] += n
        dst["entropy_sum"] += float(entropy[mask].sum().item())
        dst["max_prob_sum"] += float(max_prob[mask].sum().item())
        dst["residual_ratio_sum"] += float(residual_ratio[mask].sum().item())
        dst["residual_rms_sum"] += float(residual_rms[mask].sum().item())
        dst["retrieved_rms_sum"] += float(retrieved_rms[mask].sum().item())
        sel = winner[mask]
        for i in range(4):
            dst["winner_count"][i] += int((sel == i).sum().item())

    def summary(self, name):
        x = self.r[name]
        n = x["count"]
        return {
            "count": n,
            "entropy": _mean(x["entropy_sum"], n),
            "max_prob": _mean(x["max_prob_sum"], n),
            "residual_ratio": _mean(x["residual_ratio_sum"], n),
            "residual_rms": _mean(x["residual_rms_sum"], n),
            "retrieved_rms": _mean(x["retrieved_rms_sum"], n),
            **{f"winner_{i}_frac": _mean(x["winner_count"][i], n) for i in range(4)},
        }


class SCDTDiagnosticCollector:
    def __init__(self):
        self.active = False
        self.current = {}
        self.handles = []
        self.stats = defaultdict(RunningStats)
        self.per_image_rows = []

    def register(self, yolo):
        found = []
        for name, module in yolo.model.named_modules():
            if not isinstance(module, SCDT):
                continue
            found.append(name)

            def make_hook(layer_name):
                def hook(mod, args):
                    if not self.active:
                        return
                    if len(args) != 1 or not isinstance(args[0], (list, tuple)) or len(args[0]) != 2:
                        raise RuntimeError("Unexpected SCDT hook input; expected [x_detail, x_sem].")
                    x_detail, x_sem = args[0]
                    with torch.no_grad():
                        b, _, h, w = x_sem.shape
                        groups = mod._split_subpixels(x_detail, h, w)
                        detail = groups - groups.mean(dim=1, keepdim=True)
                        query = mod.query(x_sem)
                        keys = mod.key(detail.reshape(b * 4, mod.detail_channels, h, w))
                        keys = keys.reshape(b, 4, mod.attn_channels, h, w)
                        scores = (keys * query.unsqueeze(1)).sum(dim=2) * mod.scale
                        route = torch.softmax(scores, dim=1)
                        retrieved = (detail * route.unsqueeze(2)).sum(dim=1)
                        residual = mod.out_proj(retrieved)

                        route = route.float()
                        x_sem_f = x_sem.float()
                        residual_f = residual.float()
                        retrieved_f = retrieved.float()
                        eps = 1e-12

                        p = route.clamp_min(eps)
                        entropy = -(p * p.log()).sum(dim=1) / math.log(4.0)
                        max_prob, winner = route.max(dim=1)
                        semantic_rms = torch.sqrt(x_sem_f.pow(2).mean(dim=1) + eps)
                        residual_rms = torch.sqrt(residual_f.pow(2).mean(dim=1) + eps)
                        residual_ratio = residual_rms / (semantic_rms + eps)
                        retrieved_rms = torch.sqrt(retrieved_f.pow(2).mean(dim=1) + eps)

                        self.current[layer_name] = {
                            "entropy": entropy,
                            "max_prob": max_prob,
                            "winner": winner,
                            "residual_ratio": residual_ratio,
                            "residual_rms": residual_rms,
                            "retrieved_rms": retrieved_rms,
                            "shape": (b, h, w),
                        }
                return hook

            self.handles.append(module.register_forward_pre_hook(make_hook(name)))

        if not found:
            raise RuntimeError("No SCDT module found in checkpoint.")
        print("Found SCDT layer(s):")
        for n in found:
            print("  -", n)

    def begin_batch(self):
        self.current = {}
        self.active = True

    def end_batch(self):
        self.active = False

    @staticmethod
    def build_gt_mask(batch, batch_size, h, w, device):
        mask = torch.zeros((batch_size, h, w), dtype=torch.bool, device=device)
        bboxes = batch["bboxes"]
        batch_idx = batch["batch_idx"].view(-1)
        for bi in range(batch_size):
            boxes = bboxes[batch_idx == bi]
            if boxes.numel() == 0:
                continue
            xc, yc = boxes[:, 0] * w, boxes[:, 1] * h
            bw, bh = boxes[:, 2] * w, boxes[:, 3] * h
            x1 = torch.floor(xc - bw / 2).long().clamp(0, w - 1)
            y1 = torch.floor(yc - bh / 2).long().clamp(0, h - 1)
            x2 = torch.ceil(xc + bw / 2).long().clamp(1, w)
            y2 = torch.ceil(yc + bh / 2).long().clamp(1, h)
            x2 = torch.maximum(x2, x1 + 1).clamp(max=w)
            y2 = torch.maximum(y2, y1 + 1).clamp(max=h)
            for j in range(boxes.shape[0]):
                mask[bi, y1[j]:y2[j], x1[j]:x2[j]] = True
        return mask

    def consume_batch(self, batch):
        if not self.current:
            raise RuntimeError("SCDT hook produced no diagnostic data for this batch.")
        files = batch.get("im_file", None)
        for layer, d in self.current.items():
            entropy = d["entropy"]
            max_prob = d["max_prob"]
            winner = d["winner"]
            residual_ratio = d["residual_ratio"]
            residual_rms = d["residual_rms"]
            retrieved_rms = d["retrieved_rms"]
            b, h, w = d["shape"]
            fg = self.build_gt_mask(batch, b, h, w, entropy.device)
            bg = ~fg

            for region, mask in (("fg", fg), ("bg", bg)):
                self.stats[layer].update(region, mask, entropy, max_prob, residual_ratio, residual_rms, retrieved_rms, winner)

            for bi in range(b):
                image = str(files[bi]) if files is not None else f"image_{bi}"
                for region, mask in (("fg", fg[bi]), ("bg", bg[bi])):
                    n = int(mask.sum().item())
                    if not n:
                        continue
                    sel = winner[bi][mask]
                    row = {
                        "layer": layer,
                        "image": image,
                        "region": region,
                        "cells": n,
                        "entropy": float(entropy[bi][mask].mean().item()),
                        "max_prob": float(max_prob[bi][mask].mean().item()),
                        "residual_ratio": float(residual_ratio[bi][mask].mean().item()),
                        "residual_rms": float(residual_rms[bi][mask].mean().item()),
                        "retrieved_rms": float(retrieved_rms[bi][mask].mean().item()),
                    }
                    for k in range(4):
                        row[f"winner_{k}_frac"] = float((sel == k).float().mean().item())
                    self.per_image_rows.append(row)

    def close(self):
        self.active = False
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def write_outputs(self, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        fields = [
            "layer", "region", "cells", "entropy", "max_prob", "residual_ratio",
            "residual_rms", "retrieved_rms", "winner_0_frac", "winner_1_frac",
            "winner_2_frac", "winner_3_frac"
        ]
        rows = []
        for layer, running in self.stats.items():
            for region in ("fg", "bg"):
                s = running.summary(region)
                rows.append({"layer": layer, "region": region, "cells": s["count"], **{k: s[k] for k in fields[3:]}})

        summary = output_dir / "summary.csv"
        with summary.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)

        per_image = output_dir / "per_image.csv"
        if self.per_image_rows:
            with per_image.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(self.per_image_rows[0].keys()))
                w.writeheader()
                w.writerows(self.per_image_rows)

        report = output_dir / "diagnostic_report.txt"
        lines = ["SCDT ROUTING DIAGNOSTIC", "=" * 72, ""]
        for layer, running in self.stats.items():
            fg = running.summary("fg")
            bg = running.summary("bg")
            entropy_gap = bg["entropy"] - fg["entropy"]
            max_gap = fg["max_prob"] - bg["max_prob"]
            bg_fg_ratio = bg["residual_ratio"] / fg["residual_ratio"] if fg["residual_ratio"] > 0 else float("nan")

            lines += [
                f"Layer: {layer}",
                "-" * 72,
                f"FG: entropy={_fmt(fg['entropy'])}, max_prob={_fmt(fg['max_prob'])}, residual_ratio={_fmt(fg['residual_ratio'])}, residual_rms={_fmt(fg['residual_rms'])}, retrieved_rms={_fmt(fg['retrieved_rms'])}",
                f"BG: entropy={_fmt(bg['entropy'])}, max_prob={_fmt(bg['max_prob'])}, residual_ratio={_fmt(bg['residual_ratio'])}, residual_rms={_fmt(bg['residual_rms'])}, retrieved_rms={_fmt(bg['retrieved_rms'])}",
                f"BG entropy - FG entropy: {_fmt(entropy_gap)}",
                f"FG max_prob - BG max_prob: {_fmt(max_gap)}",
                f"BG residual_ratio / FG residual_ratio: {_fmt(bg_fg_ratio)}",
                "FG winners: " + ", ".join(f"{i}:{_fmt(fg[f'winner_{i}_frac'])}" for i in range(4)),
                "BG winners: " + ", ".join(f"{i}:{_fmt(bg[f'winner_{i}_frac'])}" for i in range(4)),
                "",
                "Interpretation hints (diagnostic only; NOT model thresholds):",
            ]
            if fg["entropy"] > 0.90 and fg["max_prob"] < 0.40:
                lines.append("- Foreground routing is close to uniform; the selector may not have learned meaningful sub-pixel preference.")
            if not math.isnan(bg_fg_ratio) and bg_fg_ratio >= 0.80:
                lines.append("- Background residual injection is close to foreground in relative strength; background detail leakage is likely.")
            if max_gap < 0.03:
                lines.append("- Foreground/background routing concentration is very similar; target selectivity is weak.")
            if abs(entropy_gap) < 0.03:
                lines.append("- Foreground/background entropy is very similar; routing is weakly object-dependent.")
            lines.append("")

        report.write_text("\n".join(lines), encoding="utf-8")
        return summary, per_image, report


class RoutingDiagnosticValidator(DetectionValidator):
    def preprocess(self, batch):
        batch = super().preprocess(batch)
        if COLLECTOR is None:
            raise RuntimeError("Diagnostic collector not initialized.")
        COLLECTOR.begin_batch()
        return batch

    def update_metrics(self, preds, batch):
        try:
            COLLECTOR.consume_batch(batch)
        finally:
            COLLECTOR.end_batch()
        return super().update_metrics(preds, batch)


def parse_args():
    p = argparse.ArgumentParser(description="SCDT routing diagnostic")
    p.add_argument("--model", default="outputs/no-pretrained/2160-yolo26-scdt-exp1-e200-b16-s42/weights/best.pt")
    p.add_argument("--data", default="./datasets/tt100k_aug_2160/TT100K.yaml")
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--device", default="0")
    p.add_argument("--output", default="outputs/diagnostics/scdt-routing-test")
    return p.parse_args()


def main():
    global COLLECTOR
    args = parse_args()
    if not Path(args.model).exists():
        raise FileNotFoundError(args.model)
    if not Path(args.data).exists():
        raise FileNotFoundError(args.data)

    yolo = YOLO(args.model)
    COLLECTOR = SCDTDiagnosticCollector()
    COLLECTOR.register(yolo)
    try:
        yolo.val(
            validator=RoutingDiagnosticValidator,
            data=args.data,
            split=args.split,
            imgsz=args.imgsz,
            batch=args.batch,
            workers=args.workers,
            device=args.device,
            plots=False,
            save_json=False,
            save_txt=False,
            verbose=False,
        )
        summary, per_image, report = COLLECTOR.write_outputs(args.output)
    finally:
        COLLECTOR.close()

    print("\nDiagnostic finished")
    print("summary   :", summary)
    print("per-image :", per_image)
    print("report    :", report)
    print("\n" + report.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
