Here’s a clean, professional `README.md` for your project:

---

# 🔍 SIFT-Guided RISE Saliency Visualization

This project implements a hybrid **SIFT-guided RISE (Randomized Input Sampling for Explanation)** method to generate saliency maps for image classification models. It combines classical computer vision (SIFT keypoints) with modern explainability techniques to produce more structured and meaningful visual explanations.

---

## 📌 Overview

Traditional RISE generates random masks across the entire image. In this implementation, we improve efficiency and interpretability by:

* Detecting **important regions** using SIFT keypoints
* Generating **Gaussian-based masks** centered on these keypoints
* Applying **RISE-style perturbations**
* Computing a **saliency map** that highlights influential regions for model predictions

---

## ⚙️ Features

* ✅ SIFT-based keypoint detection
* ✅ Guided mask generation (focused perturbations)
* ✅ Batch-based RISE saliency computation
* ✅ Visualization of:

  * Original image
  * Keypoints
  * Sample mask
  * Saliency heatmap
  * Overlay
  * Score distribution
* ✅ Works with:

  * Pretrained PyTorch models
  * Dummy model (fallback)

---

## 📂 Project Structure

```
.
├── main.py                 # Main script (this code)
├── heroine.jpg            # Input image (required)
├── your_model.pth         # Optional trained model
├── sift_rise_result.jpg   # Output visualization
└── README.md              # Documentation
```

---

## 🧠 Methodology

### 1. SIFT Keypoint Detection

Detects salient regions in the image using OpenCV's SIFT:

```python
kp = sift.detect(gray, None)
```

---

### 2. Mask Generation

* Select random subsets of keypoints
* Generate Gaussian blobs around them
* Apply downsampling + upsampling (RISE-style)

---

### 3. Saliency Computation

For each mask:

1. Apply mask to image
2. Pass through model
3. Collect prediction scores

Final saliency:

[
Saliency = \frac{1}{N} \sum_{i=1}^{N} (mask_i \cdot score_i)
]

---

### 4. Visualization

Outputs a 2×3 grid showing:

* Original image
* Keypoints
* Sample mask
* Heatmap
* Overlay
* Score histogram

---

## 🚀 Installation

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd <repo-folder>
```

### 2. Install dependencies

```bash
pip install numpy opencv-python torch torchvision matplotlib
```

---

## ▶️ Usage

1. Place your image in the project directory:

```
heroine.jpg
```

2. (Optional) Add your trained model:

```
your_model.pth
```

3. Run the script:

```bash
python main.py
```

---

## 📊 Output

After execution:

* 📁 `sift_rise_result.jpg` → Visualization output
* 📈 Console logs:

  * Number of keypoints
  * Mask details
  * Mean score
  * Peak saliency value

---

## 🧪 Notes

* If no model is provided, a **dummy CNN** is used (random weights)
* Saliency maps from the dummy model are **not meaningful**, only for testing
* For real results, use a trained classifier

---

## ⚡ Customization

You can tweak:

| Parameter      | Description                    |
| -------------- | ------------------------------ |
| `N`            | Number of masks (default: 800) |
| `grid_size`    | Resolution for mask sampling   |
| `batch_size`   | GPU/CPU performance tuning     |
| `target_class` | Explain a specific class       |

---

## 📉 Limitations

* Depends on quality of SIFT keypoints
* Computationally expensive for large `N`
* Randomness introduces slight variance
* Requires trained model for meaningful explanations

---

## 🔮 Future Improvements

* Replace SIFT with:

  * ORB / SuperPoint
* GPU optimization
* Integration with Grad-CAM comparison
* Real-time saliency visualization

---

## 📜 License

This project is open-source and available under the MIT License.

---

## 👨‍💻 Author

Developed as part of an exploration into **Explainable AI (XAI)** combining classical and deep learning techniques.

---

