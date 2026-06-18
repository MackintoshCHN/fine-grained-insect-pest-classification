# Fine-Grained Insect Pest Classification

This project builds a fine-grained image classification pipeline for insect pest recognition using the IP102 dataset. It compares several CNN-based baselines with a dual-stream vision transformer model that combines Swin Transformer and BEiT representations for 102-class insect pest classification.

The project covers dataset preparation, exploratory checks, model training, test-set evaluation, per-class analysis, confusion matrices, and Grad-CAM visualisation.

## Project Overview

Fine-grained insect pest classification is challenging because many pest categories have visually similar body shapes, colours, and textures. The IP102 dataset also contains a long-tailed class distribution, which makes model evaluation more difficult than standard balanced image classification tasks.

This project investigates the task by comparing:

* ResNet50
* DenseNet121
* EfficientNet-B0
* MobileNetV2
* DualStream Swin-BEiT model

The final evaluation uses accuracy, precision, recall, F1-score, confusion matrices, per-class accuracy, and Grad-CAM visualisation.

## Repository Structure

```text
fine-grained-insect-pest-classification/
├── README.md
├── .gitignore
├── requirements.txt
├── classes.txt
├── insect_pest_classification.ipynb
├── figures/
│   ├── accuracy_comparison.png
│   ├── model_comparison_all_metrics_annotated.png
│   ├── line_comparison_metrics.png
│   ├── gradcam_comparison.png
│   ├── ResNet50_confusion_matrix.png
│   ├── DenseNet121_confusion_matrix.png
│   ├── EfficientNet_B0_confusion_matrix.png
│   ├── MobileNetV2_confusion_matrix.png
│   ├── DualStream_Swin_BEiT_confusion_matrix.png
│   ├── ResNet50_top10_per_class_accuracy.png
│   ├── DenseNet121_top10_per_class_accuracy.png
│   ├── EfficientNet_B0_top10_per_class_accuracy.png
│   ├── MobileNetV2_top10_per_class_accuracy.png
│   └── DualStream_Swin_BEiT_top10_per_class_accuracy.png
└── reports/
    ├── ResNet50_report.txt
    ├── DenseNet121_report.txt
    ├── EfficientNet_B0_report.txt
    ├── MobileNetV2_report.txt
    └── DualStream_Swin_BEiT_report.txt
```

## Dataset

This project uses the IP102 insect pest recognition dataset.

Original dataset repository:

https://github.com/xpwu95/IP102

The raw dataset is not included in this repository due to file size. To reproduce the notebook, download the required files from the original IP102 resources.

Required files:

```text
classes.txt
ip102_v1.1.tar
resnet50_0.497.pkl
```

The dataset archive `ip102_v1.1.tar` contains:

```text
images/
train.txt
val.txt
test.txt
```

Dataset archive:

https://drive.google.com/drive/folders/1svFSy2Da3cVMvekBwe13mzyx38XZ9xWo?usp=sharing

Pretrained ResNet50 model:

https://drive.google.com/drive/folders/1MTfO3zq6BkJfhzQqxoNblQuQealdSs9U?usp=sharing

## Google Colab Setup

The notebook is designed to run in Google Colab with Google Drive mounted.

After running the first setup cell, the notebook creates the following project directory:

```python
project_drive_dir = "/content/drive/MyDrive/insect_pest_classification"
```

Place the required files in this directory:

```text
/content/drive/MyDrive/insect_pest_classification/
├── classes.txt
├── ip102_v1.1.tar
└── resnet50_0.497.pkl
```

After extraction, the expected dataset structure is:

```text
/content/drive/MyDrive/insect_pest_classification/
├── classes.txt
├── resnet50_0.497.pkl
└── ip102_v1.1/
    ├── images/
    ├── train.txt
    ├── val.txt
    └── test.txt
```

The notebook then loads the class labels, extracts the dataset if needed, prepares the train/validation/test splits, trains the models, and evaluates them on the test set.

## Installation

Install the required Python packages with:

```bash
pip install -r requirements.txt
```

Main dependencies include:

* PyTorch
* torchvision
* timm
* scikit-learn
* pandas
* NumPy
* Matplotlib
* Seaborn
* Grad-CAM

## Methodology

The project follows this workflow:

1. Set up the Google Drive project directory.
2. Load class labels and IP102 train/validation/test split files.
3. Prepare image datasets and data loaders.
4. Train and evaluate CNN baseline models.
5. Build and evaluate a dual-stream Swin-BEiT transformer model.
6. Compare model performance on the IP102 test set.
7. Analyse class-level performance using confusion matrices and top-class accuracy.
8. Visualise model attention using Grad-CAM.

## Models

### CNN Baselines

The project evaluates four CNN-based baseline models:

* ResNet50
* DenseNet121
* EfficientNet-B0
* MobileNetV2

These models provide a baseline comparison across different convolutional architectures with different parameter-efficiency and feature-extraction characteristics.

### Dual-Stream Swin-BEiT Model

The dual-stream model combines representations from Swin Transformer and BEiT. The goal is to compare transformer-based feature extraction against conventional CNN baselines on a fine-grained insect pest classification task.

## Evaluation Metrics

The models are evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion matrix
* Per-class accuracy
* Grad-CAM visualisation

The classification reports are included in the `reports/` folder.

## Results

### Overall Test-Set Performance

| Model                | Accuracy | Weighted Precision | Weighted Recall | Weighted F1 |
| -------------------- | -------: | -----------------: | --------------: | ----------: |
| ResNet50             |   0.7252 |             0.7270 |          0.7252 |      0.7247 |
| DenseNet121          |   0.7170 |             0.7146 |          0.7170 |      0.7136 |
| EfficientNet-B0      |   0.6983 |             0.6956 |          0.6983 |      0.6955 |
| MobileNetV2          |   0.6755 |             0.6695 |          0.6755 |      0.6674 |
| DualStream Swin-BEiT |   0.7520 |             0.7506 |          0.7520 |      0.7498 |

The DualStream Swin-BEiT model achieved the strongest overall test-set performance among the evaluated models.

### Accuracy Comparison

![Accuracy Comparison](figures/accuracy_comparison.png)

### Multi-Metric Model Comparison

![Model Comparison](figures/model_comparison_all_metrics_annotated.png)

![Line Comparison](figures/line_comparison_metrics.png)

## Confusion Matrix Analysis

The confusion matrices show how each model performs across selected insect pest classes and where class confusion occurs.

### ResNet50

![ResNet50 Confusion Matrix](figures/ResNet50_confusion_matrix.png)

### DenseNet121

![DenseNet121 Confusion Matrix](figures/DenseNet121_confusion_matrix.png)

### EfficientNet-B0

![EfficientNet-B0 Confusion Matrix](figures/EfficientNet_B0_confusion_matrix.png)

### MobileNetV2

![MobileNetV2 Confusion Matrix](figures/MobileNetV2_confusion_matrix.png)

### DualStream Swin-BEiT

![DualStream Swin-BEiT Confusion Matrix](figures/DualStream_Swin_BEiT_confusion_matrix.png)

## Per-Class Accuracy Analysis

The project also compares the top-performing classes for each model.

### ResNet50

![ResNet50 Top 10 Class Accuracy](figures/ResNet50_top10_per_class_accuracy.png)

### DenseNet121

![DenseNet121 Top 10 Class Accuracy](figures/DenseNet121_top10_per_class_accuracy.png)

### EfficientNet-B0

![EfficientNet-B0 Top 10 Class Accuracy](figures/EfficientNet_B0_top10_per_class_accuracy.png)

### MobileNetV2

![MobileNetV2 Top 10 Class Accuracy](figures/MobileNetV2_top10_per_class_accuracy.png)

### DualStream Swin-BEiT

![DualStream Swin-BEiT Top 10 Class Accuracy](figures/DualStream_Swin_BEiT_top10_per_class_accuracy.png)

## Grad-CAM Visualisation

Grad-CAM is used to inspect which image regions different models focus on when making predictions.

![Grad-CAM Comparison](figures/gradcam_comparison.png)

The visualisation helps compare whether each model focuses on insect body regions, background context, or less relevant image areas.

## Key Findings

* The DualStream Swin-BEiT model produced the best overall test-set accuracy and weighted F1-score.
* CNN baselines remained competitive, especially ResNet50 and DenseNet121.
* MobileNetV2 showed lower overall performance but remains useful as a lightweight baseline.
* Per-class accuracy varied noticeably across models, reflecting the fine-grained and imbalanced nature of the dataset.
* Grad-CAM visualisation provided an interpretable way to compare model attention patterns.

## How to Reproduce

1. Clone this repository.

```bash
git clone https://github.com/MackintoshCHN/fine-grained-insect-pest-classification.git
cd fine-grained-insect-pest-classification
```

2. Install dependencies.

```bash
pip install -r requirements.txt
```

3. Download the IP102 dataset archive and pretrained ResNet50 file from the original project resources.

4. Upload the following files to Google Drive:

```text
/content/drive/MyDrive/insect_pest_classification/
├── classes.txt
├── ip102_v1.1.tar
└── resnet50_0.497.pkl
```

5. Open and run:

```text
insect_pest_classification.ipynb
```

6. Run the notebook cells in order.

## Files Not Included

The following files are intentionally excluded from this repository:

```text
ip102_v1.1.tar
resnet50_0.497.pkl
*.pth
insect_pest_classification_outputs.zip
images/
```

These files are excluded because they are large dataset, pretrained-weight, checkpoint, or generated-output files.

## Limitations

* The dataset has a long-tailed class distribution, which affects minority-class performance.
* The project focuses on image-level classification rather than object detection or instance segmentation.
* The pretrained weights and raw dataset must be downloaded separately.
* The notebook is optimised for Google Colab rather than a fully packaged command-line training pipeline.

## Future Work

Potential extensions include:

* Applying class-balanced sampling or loss reweighting.
* Testing additional transformer backbones.
* Adding more systematic hyperparameter tuning.
* Converting the notebook workflow into reusable Python scripts.
* Exploring detection-based pest recognition for images containing multiple insects.

## Tech Stack

* Python
* PyTorch
* torchvision
* timm
* scikit-learn
* pandas
* NumPy
* Matplotlib
* Seaborn
* Grad-CAM
* Google Colab
