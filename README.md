# Pupil-DLC

![](giffs/pupil_tracking_ellipse.gif)





**Pupil-DLC** is a command-line tool built around [DeepLabCut](https://www.deeplabcut.org/) for pupil tracking and ellipse fitting. It supports both Individual Model (IM) and General Model (GM) workflows, includes automated video analysis, and computes eye diameter from tracked points.

---

## 🚀 Features

- CLI-based workflow for fast setup
- Supports both pre-trained (GM) and new training (IM) modes
- Automatically fits ellipses and computes eye diameter
- YAML patching to streamline IM training

---

## 🛠️ Requirements

- **Anaconda** or **Miniconda** installed  
  (Download: [Anaconda](https://www.anaconda.com/download) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html))

- **Windows 64-bit** or **Linux/macOS** with NVIDIA GPU (optional but recommended)

---

## ⚙️ Installation

### 1. Clone the repository

In the terminal (Git Bash or Anaconda terminal in Windows):
```
git clone https://github.com/Parsa2018/Pupil-DLC.git

cd Pupil-DLC
```

### 2. Create and activate the Conda environment

```
conda env create -f environment.yaml

conda activate pupil-dlc
```

### 3. Install the package locally

```
pip install -e .
```

You should see the ASCII logo and be prompted for input.
```
    ____              _ __      ____  __    ______
   / __ \__  ______  (_) /     / __ \/ /   / ____/
  / /_/ / / / / __ \/ / /_____/ / / / /   / /
 / ____/ /_/ / /_/ / / /_____/ /_/ / /___/ /___
/_/    \__,_/ .___/_/_/     /_____/_____/\____/
           /_/
```

🎮 Usage Instructions
Run the CLI:

bash:

```
pupil-dlc
```

You'll be guided through:

Choosing mode (IM or GM)

Providing experiment name

Selecting a video file

(Optional for IM) Labeling, training, and evaluation

Output: ellipse-fitted CSV with eye diameter


📦 Dependencies

Python 3.8

DeepLabCut 2.2.3

TensorFlow <2.11
