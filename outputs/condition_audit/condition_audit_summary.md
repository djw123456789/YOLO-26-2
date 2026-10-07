# Condition audit

- Dataset: `D:\vscode\workspace\recurrencePaper\YOLO-26-2\datasets\tt100k_aug_2160\TT100K.yaml`
- Test images: **2118**

## Keyword/path evidence

| Group | Images |
|---|---:|
| night_or_lowlight | 0 |
| rain | 0 |
| fog | 0 |
| snow | 0 |
| blur | 0 |
| low_contrast | 0 |
| no_keyword_hint | 2118 |

## Descriptive image statistics (NOT condition labels)

- Brightness P10/P50/P90: [113.442, 137.216, 155.386]
- Contrast P10/P50/P90: [38.682, 51.069, 66.909]
- Sharpness P10/P50/P90: [1214.933, 2083.884, 4094.161]

Important: brightness/contrast/sharpness values are descriptive diagnostics only.
Do not call them ground-truth weather labels unless the dataset provides a reproducible mapping.
