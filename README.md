# Cell-Image-Analysis-MedMNIST-Ben

**Blood cell image classification: classical image processing vs deep transfer learning**

A small, reproducible biomedical image analysis project on **BloodMNIST**, a public set of microscope images of blood cells (8 classes, 64x64 RGB). It compares a classical pipeline (segmentation + handcrafted features + Random Forest) with a fine-tuned MobileNetV2 on the same official train/val/test split.

> Learning and portfolio project on a public dataset. It is not clinical software. No patient, hospital or employer data is used.

## Pipeline
| Step | Script | What it does |
|---|---|---|
| 1 | `src/01_explore_preprocess.py` | Class balance, example images per class, CLAHE + Gaussian + Otsu preprocessing demo |
| 2 | `src/02_classical_features.py` | Otsu segmentation, shape / colour / LBP texture features, Random Forest (balanced class weights) |
| 3 | `src/03_transfer_learning.py` | MobileNetV2 (ImageNet weights) with augmentation: frozen head, then partial fine-tuning (TensorFlow/Keras) |

Metrics are test accuracy and **macro-F1** (the classes are imbalanced), plus confusion matrices.

## Quick start
```bash
git clone https://github.com/Bangash514/Cell-Image-Analysis-MedMNIST-Ben.git
cd Cell-Image-Analysis-MedMNIST-Ben
pip install -r requirements.txt
pip install tensorflow            # only for step 3
cd src
python 01_explore_preprocess.py
python 02_classical_features.py
python 03_transfer_learning.py    # GPU recommended (Google Colab / Kaggle)
```
The dataset downloads automatically on the first run. Outputs are written to `results/`.

## Results
| Model | Test accuracy | Macro-F1 |
|---|---|---|
| Classical features + Random Forest | TODO | TODO |
| MobileNetV2 transfer learning | TODO | TODO |

Figures in `results/`: `class_balance.png`, `examples.png`, `preprocessing_demo.png`, `confusion_classical.png`, `confusion_transfer.png`, `training_curve.png`.

Observations:** TODO (one or two sentences on what you saw, e.g. which classes are confused most).

## Limitations
- 64x64 images limit the detail available to segmentation and texture features.
- The official split is used and test data is not used for tuning.
- Seeds are fixed, but GPU runs can vary slightly.

## Data and references
- Dataset: BloodMNIST from MedMNIST v2, https://medmnist.com (CC BY 4.0; please cite the papers below).
- Yang J. et al., *MedMNIST v2: A large-scale lightweight benchmark for 2D and 3D biomedical image classification*, Scientific Data, 2023.
- Acevedo A. et al., *A dataset of microscopic peripheral blood cell images for development of automatic recognition systems*, Data in Brief, 2020.
- Libraries: scikit-image, scikit-learn, TensorFlow/Keras, MedMNIST.

## Author
BEN, Software Engineer. GitHub: [Bangash514](https://github.com/Bangash514)
