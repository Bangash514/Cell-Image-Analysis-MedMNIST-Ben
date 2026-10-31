# Blood Cell Image Analysis: Classical Vision vs Transfer Learning

A small, reproducible learning project on **BloodMNIST** (public microscope images of blood cells, 8 classes).
It compares a classical image-processing pipeline with a deep-learning transfer-learning model, using the same train/val/test split.

> **Status:** code is complete. The result numbers below are filled in from my own runs (see *Results*). Nothing here is clinical software, and no patient, hospital or employer data is used.

## What it does
| Step | Script | Content |
|---|---|---|
| 1 | `src/01_explore_preprocess.py` | class balance, example images, CLAHE + Gaussian + Otsu preprocessing demo |
| 2 | `src/02_classical_features.py` | Otsu segmentation, shape/colour/texture (LBP) features, Random Forest |
| 3 | `src/03_transfer_learning.py` | MobileNetV2 (ImageNet) with augmentation, frozen then fine-tuned (TensorFlow/Keras) |

Metrics: accuracy and macro-F1 on the official test split, plus confusion matrices. Macro-F1 is reported because classes are imbalanced.

## Run
```bash
pip install -r requirements.txt
pip install tensorflow          # only needed for step 3
cd src
python 01_explore_preprocess.py
python 02_classical_features.py
python 03_transfer_learning.py  # free GPU on Google Colab / Kaggle recommended
```
Figures and metric files are written to `results/`. The dataset downloads automatically on first run.

## Results
Fill these in after running (copy from `results/metrics_*.txt`):

| Model | Test accuracy | Macro-F1 |
|---|---|---|
| Classical features + Random Forest | _TODO_ | _TODO_ |
| MobileNetV2 transfer learning | _TODO_ | _TODO_ |

Figures: `results/class_balance.png`, `results/examples.png`, `results/preprocessing_demo.png`, `results/confusion_classical.png`, `results/confusion_transfer.png`, `results/training_curve.png`.

## Notes / limitations
- Images are 64x64, so segmentation features are coarse; this is a learning benchmark, not a diagnostic tool.
- The official split is used; no test data is used for tuning.
- Seeds are fixed, but GPU runs can vary slightly.

## Data and citations
- Dataset: BloodMNIST from MedMNIST v2, https://medmnist.com (check the dataset licence, CC BY 4.0, and cite properly).
- Yang J. et al., *MedMNIST v2: A large-scale lightweight benchmark for 2D and 3D biomedical image classification*, Scientific Data, 2023.
- Original blood-cell data: Acevedo A. et al., *A dataset of microscopic peripheral blood cell images for development of automatic recognition systems*, Data in Brief, 2020.
- Libraries: scikit-image, scikit-learn, TensorFlow/Keras, MedMNIST.

## Author
BEN, Software Engineer (GitHub: Bangash514)
