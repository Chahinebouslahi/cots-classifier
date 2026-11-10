# Crown-of-Thorns Starfish Classifier

Binary image classifier that tells **crown-of-thorns starfish (COTS)** apart from other reef imagery, built with transfer learning (ResNet18, PyTorch).

Crown-of-thorns starfish outbreaks are a major cause of coral loss on the Great Barrier Reef, and automated detection from underwater footage can support reef monitoring. This project is a first step: classifying pre-cropped images as COTS or not COTS.

## Dataset

[COTS v NotCOTS Cropped Crown of Thorns Dataset](https://www.kaggle.com/datasets/alexteboul/binary-cropped-crown-of-thorns-dataset) by Alex Teboul on Kaggle.

- 23,796 images, cropped from the bounding boxes of the TensorFlow "Help Protect the Great Barrier Reef" competition data
- Two balanced classes: `cots_crops` (11,898 starfish) and `notcots_crops` (11,898 other objects)

The images are **not included** in this repository (see [Getting started](#getting-started)).

## Project structure

```
cots-classifier/
├── data/                # not tracked by Git
│   ├── cots_crops/
│   └── notcots_crops/
├── cots.ipynb           # full pipeline: data, training, evaluation
├── requirements.txt
└── README.md
```

## Getting started

### 1. Clone and set up the environment

```bash
git clone https://github.com/Chahinebouslahi/cots-classifier.git
cd cots-classifier
python -m venv .venv

# Windows (Git Bash)
source .venv/Scripts/activate
# macOS / Linux
# source .venv/bin/activate

pip install -r requirements.txt
```

`pip install` gives you the CPU build of PyTorch by default. For GPU support, use the install command from [pytorch.org](https://pytorch.org/get-started/locally/) that matches your CUDA version.

### 2. Download the data

Download the dataset from the [Kaggle page](https://www.kaggle.com/datasets/alexteboul/binary-cropped-crown-of-thorns-dataset) and unzip it so that the two class folders sit directly in `data/`:

```
data/cots_crops/
data/notcots_crops/
```

### 3. Run the notebook

Open `cots.ipynb` (JupyterLab or VS Code), select the `.venv` kernel, and run the cells in order. In the notebook, set:

```python
DATA_DIR = "data"
NUM_WORKERS = 0   # safer on Windows
```

The first run downloads the pre-trained ResNet18 weights (about 45 MB), so an internet connection is needed. On CPU, expect a few minutes per epoch.

## Method

1. Load the images with `torchvision.datasets.ImageFolder`, resized to 224×224 and normalized with ImageNet statistics.
2. Split into train / validation / test (70 / 15 / 15).
3. Fine-tune a ResNet18 pre-trained on ImageNet, with its last layer replaced by a 2-class output (Adam, learning rate 1e-4, cross-entropy loss, 3 epochs).
4. Evaluate on the test set with a classification report and a confusion matrix, then inspect misclassified images.

The main metric of interest is the **recall of the COTS class**: missing a real starfish is worse than raising a false alarm.

## Results

Evaluated on a held-out test set of 3,570 images (1,778 COTS, 1,792 not COTS), used only for the final evaluation. After 3 epochs on CPU, training accuracy was 99.6% and validation accuracy 99.1%.

| Metric (test set) | Value |
|---|---|
| Accuracy | 99.3% |
| COTS precision | 99.5% |
| COTS recall | 99.1% |
| COTS F1-score | 99.3% |

Confusion matrix:

| | Predicted COTS | Predicted not COTS |
|---|---|---|
| **True COTS** | 1,762 | 16 |
| **True not COTS** | 8 | 1,784 |

Only 24 of the 3,570 test images are misclassified (16 missed starfish, 8 false alarms). These scores come from a random split and are probably optimistic (see [Limitations](#limitations)).

## Limitations

- **Possible data leakage.** The crops come from video frames, so near-identical images can land in both the training and the test set with a random split. Scores from the current split are probably optimistic, and a 99% score on a balanced two-class task is a reason to check for this before drawing conclusions.
- **Crops only.** The model classifies images that are already cropped around an object. It does not locate starfish in a full frame.
- **Baseline.** No data augmentation, hyperparameter tuning or calibration yet.
- **Single run.** Results come from one training run with one random seed, without confidence intervals.

## Roadmap

- [x] Run the baseline and fill in the results table
- [ ] Split by video or sequence to avoid leakage, and compare scores
- [ ] Add data augmentation
- [ ] Add Grad-CAM to check what the model looks at
- [ ] Compare other architectures (EfficientNet, ConvNeXt)
- [ ] Tune the decision threshold for higher COTS recall
- [ ] Export the model (ONNX) and wrap it in a small demo

## Credits

- Dataset: [Alex Teboul on Kaggle](https://www.kaggle.com/datasets/alexteboul/binary-cropped-crown-of-thorns-dataset), derived from the TensorFlow "Help Protect the Great Barrier Reef" competition data. See the dataset page for its license and terms of use.
- Model: ResNet18 from `torchvision`, pre-trained on ImageNet.

## Author

Chahine Bouslahi — [GitHub](https://github.com/Chahinebouslahi)