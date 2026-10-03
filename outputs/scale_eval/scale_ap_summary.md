# Scale-aware AP evaluation

- Dataset: `D:\vscode\workspace\recurrencePaper\YOLO-26-2\datasets\tt100k_aug_2160\TT100K.yaml`
- Input size: `640 × 640`
- Definition: detector-input-space COCO-style area thresholds
  - Small: area < 32²
  - Medium: 32² ≤ area < 96²
  - Large: area ≥ 96²

| Model | Group | GT | Classes | AP50 | AP75 | mAP50-95 | ΔmAP vs Baseline |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | All | 5474 | 45 | 0.7492 | 0.6505 | 0.5561 | +0.0000 |
| Baseline | Small | 5162 | 45 | 0.7507 | 0.6447 | 0.5506 | +0.0000 |
| Baseline | Medium | 312 | 42 | 0.8237 | 0.7945 | 0.7231 | +0.0000 |
| Baseline | Large | 0 | 0 | - | - | - | - |
| WGFS | All | 5474 | 45 | 0.7622 | 0.6750 | 0.5731 | +0.0170 |
| WGFS | Small | 5162 | 45 | 0.7671 | 0.6732 | 0.5714 | +0.0208 |
| WGFS | Medium | 312 | 42 | 0.7997 | 0.7978 | 0.7048 | -0.0183 |
| WGFS | Large | 0 | 0 | - | - | - | - |
| LCMA | All | 5474 | 45 | 0.7581 | 0.6695 | 0.5666 | +0.0105 |
| LCMA | Small | 5162 | 45 | 0.7596 | 0.6643 | 0.5617 | +0.0111 |
| LCMA | Medium | 312 | 42 | 0.8295 | 0.8278 | 0.7346 | +0.0115 |
| LCMA | Large | 0 | 0 | - | - | - | - |
| MCLD | All | 5474 | 45 | 0.7573 | 0.6716 | 0.5704 | +0.0143 |
| MCLD | Small | 5162 | 45 | 0.7607 | 0.6698 | 0.5678 | +0.0172 |
| MCLD | Medium | 312 | 42 | 0.7992 | 0.7856 | 0.7012 | -0.0219 |
| MCLD | Large | 0 | 0 | - | - | - | - |
| Final | All | 5474 | 45 | 0.7697 | 0.6850 | 0.5773 | +0.0212 |
| Final | Small | 5162 | 45 | 0.7704 | 0.6794 | 0.5722 | +0.0216 |
| Final | Medium | 312 | 42 | 0.8308 | 0.8278 | 0.7293 | +0.0061 |
| Final | Large | 0 | 0 | - | - | - | - |
