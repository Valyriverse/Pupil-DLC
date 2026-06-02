# Pupil-DLC

<p align="center">
  <img src="giffs/pupil_tracking_ellipse.gif" />
</p>




**Pupil-DLC** is a command-line tool built around [DeepLabCut](https://www.deeplabcut.org/) for pupil tracking and ellipse fitting. It supports both Individual Model (IM) and General Model (GM) workflows, includes automated video analysis, and computes eye diameter from tracked points.


Compiled annotated frames to train our model is publicly [available](https://doi.org/10.6084/m9.figshare.31282714).

There is a video [Tutorial](https://youtu.be/G_frXJzCL6I) as well.

There is a jupyter Notebook demo under Test folder as well.

If you encounter any problems or bugs, please reach out [here](https://github.com/Valyriverse/Pupil-DLC/issues).

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

Choosing mode (IM or GM or RT)

Providing experiment name

Selecting a video file

(Optional for IM) Labeling, training, and evaluation

Output: ellipse-fitted CSV with eye diameter


📦 Dependencies

Python 3.8

DeepLabCut 2.2.3

TensorFlow <2.11

### Citations
If you use our corresponding pupil data (cite paper 1), or our tool please cite the following:
```
1. Seyfourian, P., Marks, L.C., Claar, L.D., Nahas, Y., Keating, M., Koch, C. and Rembado, I., 2026. Pupil-DLC: an open-source deep learning pipeline for scalable, markerless tracking of pupil dynamics across conscious and unconscious states. bioRxiv, pp.2026-01.

2. Mathis, A., Mamidanna, P., Cury, K.M., Abe, T., Murthy, V.N., Mathis, M.W. and Bethge, M., 2018. DeepLabCut: markerless pose estimation of user-defined body parts with deep learning. Nature neuroscience, 21(9), pp.1281-1289.
```

## License

This project is licensed under the GNU General Public License v3.0.

This means you may use, modify, and redistribute this software, but any redistributed modified version must also remain open source under the same license.

See the LICENSE file for the full license text.
