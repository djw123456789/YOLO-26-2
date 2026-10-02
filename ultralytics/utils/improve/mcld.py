from __future__ import annotations

from typing import Any

import torch

from ultralytics.utils.tal import make_anchors


class MCLDLoss:
    """
    Multi-hypothesis Consensus Localization Distillation (MCLD)
    with a coordinate-wise Teacher-Advantage Gate.

    Distillation is active only when the detached O2M consensus is
    closer to GT than the detached O2O student for that coordinate.

    No new learnable parameter, lambda, size threshold, uncertainty
    threshold, or inference branch is introduced.
    """

    def __init__(self, eps: float = 1e-9):
        self.eps = float(eps)

    @staticmethod
    def _as_xyxy_scale(gt_box: torch.Tensor, eps: float) -> torch.Tensor:
        """Return [w, h, w, h] for xyxy error normalization."""
        w = (gt_box[2] - gt_box[0]).clamp_min(eps)
        h = (gt_box[3] - gt_box[1]).clamp_min(eps)
        return torch.stack((w, h, w, h))

    def _assignment_meta(
        self,
        criterion: Any,
        preds: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        pred_distri = preds["boxes"].permute(0, 2, 1).contiguous()
        pred_scores = preds["scores"].permute(0, 2, 1).contiguous()

        anchor_points, stride_tensor = make_anchors(
            preds["feats"],
            criterion.stride,
            0.5,
        )

        batch_size = pred_scores.shape[0]
        dtype = pred_scores.dtype

        imgsz = (
            torch.tensor(
                preds["feats"][0].shape[2:],
                device=criterion.device,
                dtype=dtype,
            )
            * criterion.stride[0]
        )

        targets = torch.cat(
            (
                batch["batch_idx"].view(-1, 1),
                batch["cls"].view(-1, 1),
                batch["bboxes"],
            ),
            dim=1,
        )

        targets = criterion.preprocess(
            targets.to(criterion.device),
            batch_size,
            scale_tensor=imgsz[[1, 0, 1, 0]],
        )

        gt_labels, gt_bboxes = targets.split((1, 4), dim=2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        pred_bboxes = criterion.bbox_decode(
            anchor_points,
            pred_distri,
        )
        pred_bboxes_px = pred_bboxes * stride_tensor

        with torch.no_grad():
            (
                _,
                _,
                target_scores,
                fg_mask,
                target_gt_idx,
            ) = criterion.assigner(
                pred_scores.detach().sigmoid(),
                pred_bboxes_px.detach().type(gt_bboxes.dtype),
                anchor_points * stride_tensor,
                gt_labels,
                gt_bboxes,
                mask_gt,
            )

        return {
            "pred_bboxes_px": pred_bboxes_px,
            "target_scores": target_scores.detach(),
            "fg_mask": fg_mask.detach(),
            "target_gt_idx": target_gt_idx.detach(),
            "gt_bboxes": gt_bboxes.detach(),
        }

    def __call__(
        self,
        one2many_criterion: Any,
        one2one_criterion: Any,
        one2many_preds: dict[str, torch.Tensor],
        one2one_preds: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        teacher = self._assignment_meta(
            one2many_criterion,
            one2many_preds,
            batch,
        )
        student = self._assignment_meta(
            one2one_criterion,
            one2one_preds,
            batch,
        )

        zero = one2one_preds["boxes"].sum() * 0.0

        teacher_fg = teacher["fg_mask"]
        student_fg = student["fg_mask"]

        if not bool(teacher_fg.any()) or not bool(student_fg.any()):
            return zero

        gt_losses = []

        for b in range(teacher_fg.shape[0]):
            t_fg = teacher_fg[b]
            s_fg = student_fg[b]

            if not bool(t_fg.any()) or not bool(s_fg.any()):
                continue

            gt_ids = torch.unique(
                teacher["target_gt_idx"][b, t_fg]
            )

            for gt_id_tensor in gt_ids:
                gt_id = int(gt_id_tensor.item())

                t_mask = t_fg & (
                    teacher["target_gt_idx"][b] == gt_id
                )
                t_idx = torch.nonzero(
                    t_mask,
                    as_tuple=False,
                ).squeeze(1)

                if t_idx.numel() < 2:
                    continue

                s_mask = s_fg & (
                    student["target_gt_idx"][b] == gt_id
                )
                s_idx = torch.nonzero(
                    s_mask,
                    as_tuple=False,
                ).squeeze(1)

                if s_idx.numel() == 0:
                    continue

                # -----------------------------
                # O2M teacher consensus
                # -----------------------------
                teacher_boxes = (
                    teacher["pred_bboxes_px"][b, t_idx]
                    .detach()
                    .float()
                )

                quality = (
                    teacher["target_scores"][b, t_idx]
                    .sum(dim=-1)
                    .detach()
                    .float()
                    .clamp_min(0.0)
                )

                quality_sum = quality.sum()
                if float(quality_sum.item()) <= self.eps:
                    continue

                weights = quality / quality_sum.clamp_min(self.eps)

                consensus = (
                    weights[:, None] * teacher_boxes
                ).sum(dim=0)

                centered = teacher_boxes - consensus[None, :]

                dispersion = torch.sqrt(
                    (
                        weights[:, None]
                        * centered.square()
                    ).sum(dim=0)
                    + self.eps
                )

                gt_box = (
                    teacher["gt_bboxes"][b, gt_id]
                    .detach()
                    .float()
                )

                coord_scale = self._as_xyxy_scale(
                    gt_box,
                    self.eps,
                )

                normalized_dispersion = dispersion / coord_scale

                dispersion_reliability = 1.0 / (
                    1.0 + normalized_dispersion
                )

                group_quality = quality.max().clamp(
                    min=0.0,
                    max=1.0,
                )

                teacher_reliability = (
                    dispersion_reliability
                    * group_quality
                )

                # -----------------------------
                # O2O student
                # -----------------------------
                student_boxes = (
                    student["pred_bboxes_px"][b, s_idx]
                    .float()
                )

                # -----------------------------
                # Teacher-Advantage Gate
                # -----------------------------
                # Gate uses detached values only, so the student cannot
                # "game" the gate through gradients.
                teacher_error = (
                    (consensus - gt_box).abs()
                    / coord_scale
                ).detach()

                student_error_for_gate = (
                    (
                        student_boxes.detach()
                        - gt_box[None, :]
                    ).abs()
                    / coord_scale[None, :]
                )

                teacher_advantage = (
                    (
                        student_error_for_gate
                        - teacher_error[None, :]
                    ).clamp_min(0.0)
                    / (
                        student_error_for_gate
                        + teacher_error[None, :]
                        + self.eps
                    )
                ).detach()

                final_reliability = (
                    teacher_reliability[None, :]
                    * teacher_advantage
                )

                # -----------------------------
                # Scale-normalized distillation
                # -----------------------------
                normalized_distill_error = (
                    (
                        student_boxes
                        - consensus[None, :]
                    ).abs()
                    / coord_scale[None, :]
                )

                per_student_loss = (
                    normalized_distill_error
                    * final_reliability
                ).mean(dim=-1)

                gt_losses.append(
                    per_student_loss.mean()
                )

        if not gt_losses:
            return zero

        return torch.stack(gt_losses).mean()
