Crown-of-Thorns Starfish Classifier

Binary image classifier that tells crown-of-thorns starfish (COTS) apart from other reef imagery, built with transfer learning (ResNet18, PyTorch).

Crown-of-thorns starfish outbreaks are a major cause of coral loss on the Great Barrier Reef, and automated detection from underwater footage can support reef monitoring. This project is a first step: classifying pre-cropped images as COTS or not COTS.

Dataset

COTS v NotCOTS Cropped Crown of Thorns Dataset by Alex Teboul on Kaggle.

11,898 images cropped from the bounding boxes of the TensorFlow "Help Protect the Great Barrier Reef" competition data
Two classes: cots_crops (starfish) and notcots_crops (everything else)

The images are not included in this repository (see Getting started).

Project structure
cots-classifier/
├── data/                # not tracked by Git
│   ├── cots_crops/
│   └── notcots_crops/
├── cots.ipynb           # full pipeline: data, training, evaluation
├── requirements.txt
└── README.md
Getting started
1. Clone and set up the environment
bash
git clone https://github.com/Chahinebouslahi/cots-classifier.git
cd cots-classifier
python -m venv .venv

# Windows (Git Bash)
source .venv/Scripts/activate


pip install -r requirements.txt

pip install gives you the CPU build of PyTorch by default. For GPU support, use the install command from pytorch.org that matches your CUDA version.

2. Download the data

Download the dataset from the Kaggle page and unzip it so that the two class folders sit directly in data/:

data/cots_crops/
data/notcots_crops/
3. Run the notebook

Open cots.ipynb (JupyterLab or VS Code), select the .venv kernel, and run the cells in order. In the notebook, set:

python
DATA_DIR = "data"
NUM_WORKERS = 0   # safer on Windows

The first run downloads the pre-trained ResNet18 weights (about 45 MB), so an internet connection is needed. On CPU, expect a few minutes per epoch.

Method
Load the images with torchvision.datasets.ImageFolder, resized to 224×224 and normalized with ImageNet statistics.
Split into train / validation / test (70 / 15 / 15).
Fine-tune a ResNet18 pre-trained on ImageNet, with its last layer replaced by a 2-class output (Adam, learning rate 1e-4, cross-entropy loss, 3 epochs).
Evaluate on the test set with a classification report and a confusion matrix, then inspect misclassified images.

The main metric of interest is the recall of the COTS class: missing a real starfish is worse than raising a false alarm.

Results

To be filled in after the first full run.

Metric (test set)	Value
Accuracy	TBD
COTS precision	TBD
COTS recall	TBD
COTS F1	TBD
Limitations
Possible data leakage. The crops come from video frames, so near-identical images can land in both the training and the test set with a random split. Scores from the current split are probably optimistic.
Crops only. The model classifies images that are already cropped around an object. It does not locate starfish in a full frame.
Baseline. No data augmentation, hyperparameter tuning or calibration yet.

Author

Chahine Bouslahi — GitHub