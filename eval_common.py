"""
Shared evaluation primitives so every model (prototype FSS, MedSAM2, ...) is
scored by the *identical* code and lands in the same per-organ + MEAN table.

Kept deliberately tiny and dependency-light (only torch/numpy) so importing it
does not drag in the FSS model stack.
"""
import numpy as np
import torch


class Scores:
    """Accumulates per-patient 3D Dice and IoU (same as Q-Net utils.Scores)."""

    def __init__(self):
        self.patient_dice: list[float] = []
        self.patient_iou:  list[float] = []

    def record(self, pred: torch.Tensor, label: torch.Tensor):
        tp = ((label == 1) & (pred == 1)).sum().float()
        fp = ((label == 0) & (pred == 1)).sum().float()
        fn = ((label == 1) & (pred == 0)).sum().float()
        dice = (2 * tp / (2 * tp + fp + fn + 1e-8)).item()
        iou  = (tp / (tp + fp + fn + 1e-8)).item()
        self.patient_dice.append(dice)
        self.patient_iou.append(iou)


def aggregate_and_print(class_dice: dict[str, float],
                        class_iou: dict[str, float]) -> dict:
    """Per-class results + MEAN (mean of per-class means). Same format/semantics
    as test.py final block so the two scripts produce comparable tables."""
    print('\n===== Final results =====')
    results: dict[str, dict] = {}
    for name in class_dice:
        results[name] = {'dice': class_dice[name], 'iou': class_iou[name]}
        print(f'  {name}: Dice={class_dice[name]:.4f}  IoU={class_iou[name]:.4f}')
    if class_dice:
        mean_d = float(np.mean(list(class_dice.values())))
        mean_i = float(np.mean(list(class_iou.values())))
        results['MEAN'] = {'dice': mean_d, 'iou': mean_i}
        print(f'  MEAN:  Dice={mean_d:.4f}  IoU={mean_i:.4f}')
    return results
