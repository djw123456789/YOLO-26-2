# Scale-aware AP evaluation

- Dataset: `D:\vscode\workspace\recurrencePaper\YOLO-26-2\datasets\tt100k_aug_2160\TT100K.yaml`
- Input size: `640 × 640`
- Equivalent side: `s = sqrt(box area)` in detector-input space
- COCO-style reference partition:
  - Small: `s < 32 px`
  - Medium: `32 <= s < 96 px`
  - Large: `s >= 96 px`
- Stride-aware traffic-sign partition (P3 stride = 8):
  - Ultra-Tiny: `s < 8 px`
  - Tiny: `8 <= s < 16 px`
  - Small: `16 <= s < 32 px`
  - Non-small: `s >= 32 px`
- The 8/16/32 thresholds are fixed from detector resolution and must not be tuned from AP results.

| Model | Group | GT | Classes | AP50 | AP75 | mAP50-95 | ΔmAP vs Baseline |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | All | 5474 | 45 | 0.7492 | 0.6505 | 0.5561 | +0.0000 |
| Baseline | COCO-Small(<32) | 5162 | 45 | 0.7507 | 0.6447 | 0.5506 | +0.0000 |
| Baseline | COCO-Medium(32-96) | 312 | 42 | 0.8237 | 0.7945 | 0.7231 | +0.0000 |
| Baseline | COCO-Large(>=96) | 0 | 0 | - | - | - | - |
| Baseline | Ultra-Tiny(<8) | 1606 | 45 | 0.5954 | 0.3929 | 0.3608 | +0.0000 |
| Baseline | Tiny(8-16) | 2277 | 45 | 0.7934 | 0.7133 | 0.5844 | +0.0000 |
| Baseline | Small(16-32) | 1279 | 45 | 0.8149 | 0.7933 | 0.6823 | +0.0000 |
| Baseline | Non-small(>=32) | 312 | 42 | 0.8237 | 0.7945 | 0.7231 | +0.0000 |
| WGFS | All | 5474 | 45 | 0.7622 | 0.6750 | 0.5731 | +0.0170 |
| WGFS | COCO-Small(<32) | 5162 | 45 | 0.7671 | 0.6732 | 0.5714 | +0.0208 |
| WGFS | COCO-Medium(32-96) | 312 | 42 | 0.7997 | 0.7978 | 0.7048 | -0.0183 |
| WGFS | COCO-Large(>=96) | 0 | 0 | - | - | - | - |
| WGFS | Ultra-Tiny(<8) | 1606 | 45 | 0.5979 | 0.4307 | 0.3784 | +0.0176 |
| WGFS | Tiny(8-16) | 2277 | 45 | 0.8147 | 0.7329 | 0.6005 | +0.0160 |
| WGFS | Small(16-32) | 1279 | 45 | 0.8332 | 0.8088 | 0.7045 | +0.0223 |
| WGFS | Non-small(>=32) | 312 | 42 | 0.7997 | 0.7978 | 0.7048 | -0.0183 |
| LCMA | All | 5474 | 45 | 0.7581 | 0.6695 | 0.5666 | +0.0105 |
| LCMA | COCO-Small(<32) | 5162 | 45 | 0.7596 | 0.6643 | 0.5617 | +0.0111 |
| LCMA | COCO-Medium(32-96) | 312 | 42 | 0.8295 | 0.8278 | 0.7346 | +0.0115 |
| LCMA | COCO-Large(>=96) | 0 | 0 | - | - | - | - |
| LCMA | Ultra-Tiny(<8) | 1606 | 45 | 0.5845 | 0.3997 | 0.3575 | -0.0032 |
| LCMA | Tiny(8-16) | 2277 | 45 | 0.8107 | 0.7333 | 0.6003 | +0.0159 |
| LCMA | Small(16-32) | 1279 | 45 | 0.8331 | 0.8061 | 0.6938 | +0.0116 |
| LCMA | Non-small(>=32) | 312 | 42 | 0.8295 | 0.8278 | 0.7346 | +0.0115 |
| MCLD | All | 5474 | 45 | 0.7573 | 0.6716 | 0.5704 | +0.0143 |
| MCLD | COCO-Small(<32) | 5162 | 45 | 0.7607 | 0.6698 | 0.5678 | +0.0172 |
| MCLD | COCO-Medium(32-96) | 312 | 42 | 0.7992 | 0.7856 | 0.7012 | -0.0219 |
| MCLD | COCO-Large(>=96) | 0 | 0 | - | - | - | - |
| MCLD | Ultra-Tiny(<8) | 1606 | 45 | 0.6086 | 0.4520 | 0.3917 | +0.0309 |
| MCLD | Tiny(8-16) | 2277 | 45 | 0.8035 | 0.7182 | 0.5944 | +0.0100 |
| MCLD | Small(16-32) | 1279 | 45 | 0.8332 | 0.8095 | 0.6999 | +0.0176 |
| MCLD | Non-small(>=32) | 312 | 42 | 0.7992 | 0.7856 | 0.7012 | -0.0219 |
| Final | All | 5474 | 45 | 0.7697 | 0.6850 | 0.5773 | +0.0212 |
| Final | COCO-Small(<32) | 5162 | 45 | 0.7704 | 0.6794 | 0.5722 | +0.0216 |
| Final | COCO-Medium(32-96) | 312 | 42 | 0.8308 | 0.8278 | 0.7293 | +0.0061 |
| Final | COCO-Large(>=96) | 0 | 0 | - | - | - | - |
| Final | Ultra-Tiny(<8) | 1606 | 45 | 0.6248 | 0.4260 | 0.3883 | +0.0275 |
| Final | Tiny(8-16) | 2277 | 45 | 0.7966 | 0.7354 | 0.5904 | +0.0060 |
| Final | Small(16-32) | 1279 | 45 | 0.8434 | 0.8154 | 0.7067 | +0.0245 |
| Final | Non-small(>=32) | 312 | 42 | 0.8308 | 0.8278 | 0.7293 | +0.0061 |
