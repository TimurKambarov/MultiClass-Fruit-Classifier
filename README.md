# Fruit Quality Classification

A deep learning project for automated 3-class fruit quality grading, developed as part of the Applied Data Science & Artificial Intelligence programme at Breda University of Applied Sciences (Block 1C, 2025–2026).

## Project Overview

This project trains a CNN-based image classifier to grade fresh produce into three quality classes:

| Class | Description | Destination |
|---|---|---|
| **1st Class** | Perfect produce, no visible damage | Premium retail market |
| **2nd Class** | Minor skin damage, still edible | Processing / discount market |
| **Rotten** | Mouldy or heavily damaged | Compost / disposal |

The classifier supports three fruit types: **banana**, **apple**, and **orange** — resulting in 9 total classes.

## Motivation

Small-to-medium AGF (Aardappelen, Groente & Fruit) distributors in the Netherlands face high manual sorting costs and are priced out of commercial automation systems (€100,000–€500,000). This project explores an affordable deep learning alternative using a Raspberry Pi + server-based architecture.

## Dataset

- **Base dataset:** [Fruits Fresh and Rotten for Classification](https://www.kaggle.com/datasets/sriramr/fruits-fresh-and-rotten-for-classification) (Kaggle) — provides Class 1 and Rotten images
- **Class 2 images:** AI-generated using detailed prompts to simulate slightly damaged produce
- **Total images used:** 3,960 (440 per class, balanced)
- **Resolution:** 128×128 pixels
- **Train / Validation / Test split:** 80 / 10 / 10

The final dataset is not included in this repository due to size. You can download it from [here](https://edubuas-my.sharepoint.com/:f:/g/personal/250596_buas_nl/IgAQLwVEvTTCRrglUbUN6-sfAbJj8TnTCkQLOkFKe8_VV-c?e=aNbuCi).

Once downloaded, place it so the structure matches:

```
raw_dataset/
├── collecting_data.py      # downloads base Kaggle dataset via kagglehub
└── dataset/
    ├── 1st_class_apple/
    ├── 1st_class_banana/
    ├── 1st_class_oranges/
    ├── 2nd_class_apple/
    ├── 2nd_class_banana/
    ├── 2nd_class_orange/
    ├── rotten_apple/
    ├── rotten_banana/
    └── rotten_orange/
```

The Class 1 and Rotten images can alternatively be downloaded by running `collecting_data.py`, which uses `kagglehub` to fetch the base Kaggle dataset automatically.

## Notebooks

| File | Description |
|---|---|
| `main_notebook.ipynb` | Four CNN iterations (Basic CNN → Data Augmentation → Transfer Learning → Model Improvement) |
| `mlp_baseline.ipynb` | Multilayer Perceptron baseline model |
| `xai_notebook.ipynb` | Explainable AI analysis |

## Model Iterations

| Iteration | Approach | Test Accuracy |
|---|---|---|
| Random Guess | — | 11.1% |
| MLP Baseline | 2 hidden layers (512 → 256) | 79.8% |
| Basic CNN | 3 conv layers + dropout + early stopping | 95.45% |
| Data Augmentation | Random flip, rotation, zoom | 96.46% |
| Transfer Learning | Pretrained MobileNet | 99.24% |
| **Model Improvement** | **MobileNet + smaller dense layers + ReduceLROnPlateau** | **100%** |

## Key Findings

- More parameters are not always better — the final model is roughly half the size of the transfer learning iteration while achieving higher accuracy
- Pretrained models (MobileNet) dramatically reduce training time and improve performance
- Callback functions (EarlyStopping, ReduceLROnPlateau) are effective when tuned carefully
- The main error pattern in earlier iterations was confusion between round-shaped fruits (orange vs. apple), resolved by transfer learning

## Requirements

See `requirements.txt` for the full Python environment.

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

## Acknowledgements

- Dataset: Kalluri, S. R. (2018). *Fruits fresh and rotten for classification*. Kaggle.
- Pretrained model: MobileNet via [Keras Applications](https://keras.io/api/applications/mobilenet/)