# PLANT-DISEASE-DETECTION-USING-EFFNET_V2_L
Okay, while Markdown itself doesn't support complex animations like videos directly within the file, we can make the GitHub README.md more visually engaging using elements like:

Animated GIFs: You'd need to create this separately (e.g., showing a demo of the classification, training progress, or just a relevant animation) and embed it.

Badges/Shields: These provide visually appealing information (like build status, code version, license).

Emojis: Add visual cues and personality.

Well-structured Layout: Using headings, code blocks, and lists effectively.

Here's an enhanced README.md incorporating these ideas. You'll need to create the animated GIF yourself if you want one.

# 🌿 Plant Disease Classification with EfficientNetV2-L & PyTorch 🚀

[![Python Version](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![PyTorch Version](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) <!-- Update license if different -->

This repository provides a Python script for training an EfficientNetV2-L model to classify plant diseases using PyTorch. It leverages transfer learning on the "New Plant Diseases Dataset".

---

<!-- 🖼️ **Optional: Insert an Animated GIF Here!** 🖼️ -->
<!-- You can create a GIF showing:
     - Examples of input images and predicted classes.
     - A screen recording of the training progress (loss decreasing).
     - A cool plant/AI related animation.
     Replace the line below with your GIF embedding code:
     ![Model Demo](link_to_your_demo.gif)
-->

---

## ✨ Features

*   **🧠 Transfer Learning:** Fine-tunes a pre-trained EfficientNetV2-L (`torchvision`).
*   **🔥 PyTorch Powered:** Built entirely using the PyTorch framework.
*   **⚡ GPU Accelerated:** Automatically utilizes CUDA GPU if available.
*   **💾 Easy Dataset Handling:** Uses `torchvision.datasets.ImageFolder`.
*   **💡 Mixed-Precision:** Option for faster training and lower memory usage via `torch.cuda.amp`.
*   **⚙️ Memory Optimization:** Configures PyTorch CUDA allocator for potentially reduced fragmentation.
*   **📉 AdamW Optimizer:** Utilizes the AdamW optimization algorithm.
*   **💾 Checkpointing:** Saves model state after each epoch (named `loss epoch.pth`).
*   **👯 Multi-GPU Ready:** Basic multi-GPU support using `nn.DataParallel`.

---

## 💾 Dataset

*   **Required:** "New Plant Diseases Dataset (Augmented)". Find versions online (e.g., Kaggle).
*   **Expected Structure:** Standard `ImageFolder` format:
    ```
    <dataset_root_path>/
    └── train/
        ├── Class_1/
        │   ├── image_1.jpg
        │   └── ...
        ├── Class_2/
        │   ├── image_3.jpg
        │   └── ...
        └── ...
    ```
*   **❗ Configuration:** **You MUST update** the `train` variable in the script to your dataset's path.

---

## ⚙️ Requirements

*   Python 3.x
*   PyTorch (`torch`)
*   Torchvision (`torchvision`)
*   NVIDIA GPU + CUDA (Highly Recommended!)

Install primary libraries using pip:
```bash
pip install torch torchvision torchaudio


(See official PyTorch website for specific CUDA versions).

▶️ Usage Steps

Clone:

git clone <your-repository-url>
cd <repository-directory>
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END

(Replace placeholders)

Dataset: Download and place the dataset. Ensure the structure matches the description above.

Modify Script: Open train_plant_disease.py (or your script name) and edit the train path:

# --- MODIFY THIS LINE ---
train = "/path/to/your/dataset/New Plant Diseases Dataset(Augmented)/train"
# ------------------------
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Python
IGNORE_WHEN_COPYING_END

(Replace with your actual path)

Run Training:

python train_plant_disease.py
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END

(Use your script's filename)

📈 Training progress (loss per epoch) will be printed to the console. Model checkpoints (.pth files) are saved locally.

🛠️ Configuration Parameters

Adjust these within the script as needed:

train: Path to training data (MUST change).

num_classes: Auto-detected from dataset folders.

lr: Learning Rate (default: 0.0001).

betas: AdamW betas (default: (0.5, 0.999)).

batch_size: Batch Size (default: 32). Adjust based on GPU VRAM.

num_workers: DataLoader workers (default: 2).

num_epochs: Training epochs (default: 15).

📊 Output

Console: Epoch number and average training loss.

Epoch [1/15], Loss: X.XXXX
Epoch [2/15], Loss: Y.YYYY
...
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
IGNORE_WHEN_COPYING_END

Files: Model state dictionaries saved as {avg_loss} {epoch}.pth after each epoch.

📜 License

[Specify Your License Here - e.g., MIT]
(Replace with your chosen license or remove)

**To use this README:**

1.  **Copy the entire text block** above.
2.  Create a file named `README.md` in your project's root directory.
3.  **Paste** the text into the file.
4.  **Crucially:**
    *   Replace placeholder values like `<your-repository-url>`, `<repository-directory>`, `/path/to/your/dataset/...`, and the License information.
    *   **Create an animated GIF** (if desired) using screen recording tools (like Kap, ScreenToGif) or image editing software. Upload it somewhere accessible (like GitHub itself, Imgur) and replace the placeholder comment/link `![Model Demo](link_to_your_demo.gif)` with the correct Markdown image link.
    *   Update the script name (`train_plant_disease.py`) if yours is different.
5.  Save the `README.md` file.
6.  Commit and push to GitHub. It will render automatically on your repository page.
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
IGNORE_WHEN_COPYING_END
