# Deep Learning & Neural Networks in Python
### Chapter 10 Applied Exercises — *An Introduction to Statistical Learning (ISLR2)*

[![Author](https://img.shields.io/badge/Author-Shad%20Ali%20Shah-blue)](https://github.com/shadalishah)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?logo=linkedin)](https://www.linkedin.com/in/shad-ali-shah-6439ab339/)
[![Framework](https://img.shields.io/badge/Framework-PyTorch%20%7C%20PyTorch--Lightning-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![Language](https://img.shields.io/badge/Language-Python-3776AB?logo=python)](https://www.python.org/)
[![Topic](https://img.shields.io/badge/Topic-Deep%20Learning-blueviolet)]()

---

## 🚀 Project Overview

This project tackles **8 deep learning challenges** using PyTorch and PyTorch Lightning — covering everything from implementing gradient descent from scratch to deploying a pre-trained ResNet50 image classifier and building RNN-based financial forecasting models.

Every model was **designed, trained, and evaluated end-to-end**: architecture choices are justified, results are benchmarked against baselines, and key engineering decisions (dropout rates, learning rates, sequence design) are documented.

> Built with: `PyTorch` · `PyTorch Lightning` · `scikit-learn` · `statsmodels` · `ResNet50` · `ISLP`

---

## 🛠️ Technical Skills Demonstrated

| Area | Technologies & Techniques |
|---|---|
| **Neural Network Design** | Feedforward networks, ReLU activations, Sigmoid output, custom `nn.Module` classes |
| **Regularization** | Dropout (0.3–0.5), overfitting analysis across model configurations |
| **Optimization** | Gradient descent (manual), RMSprop, Adam, learning rate tuning |
| **Classification** | Binary classification with BCE loss; multi-class with softmax probabilities |
| **Regression** | R² evaluation for continuous targets; linear and non-linear AR models |
| **Sequence Modelling** | RNN with lag-5 windows, flattened AR sequences, day-of-week feature injection |
| **Transfer Learning** | ResNet50 (ImageNet pre-trained) inference with top-5 probability output |
| **Model Benchmarking** | Neural networks vs. logistic regression, linear vs. non-linear AR vs. RNN |
| **Feature Engineering** | Lag features, one-hot encoding, StandardScaler, month/day dummies |
| **Deep Learning Stack** | `torch`, `torch.nn`, `pytorch_lightning`, `torchvision`, `torchmetrics`, `torchinfo` |

---

## 📦 What Was Built

---

### ⚙️ Exercise 6 — Gradient Descent Implemented from Scratch

**Problem:** Minimize R(β) = sin(β) + β/10 using gradient descent with two different starting points.

**What I built:**
- A reusable `gradient_descent()` function with configurable learning rate, iterations, objective and derivative
- Gradient trajectory plots overlaid on the loss curve showing step-by-step convergence

**Key results:**
- β₀ = 2.3 → converged to a **local minimum** in the right region of the curve
- β₀ = 1.4 → converged to a **different local minimum** — demonstrating how initialization determines which solution is found
- Derivative R'(β) = cos(β) + 1/10, analytically derived and verified in code

**Why it matters:** Understanding gradient descent at this level — not just calling an optimizer — demonstrates the mathematical foundation underpinning every neural network training loop.

---

### 🧠 Exercise 7 — Neural Network for Credit Default Prediction

**Problem:** Predict whether a bank customer will default on their credit (Default dataset, 10,000 observations).

**Architecture:**
```
Input(3) → Linear(3→10) → ReLU → Dropout(0.4) → Linear(10→1) → Sigmoid
```

**Training:** PyTorch Lightning · 50 epochs · batch size 32 · BCE loss

**Results vs. baseline:**

| Model | Approach | Test Performance |
|---|---|---|
| Neural Network | 1 hidden layer, 10 units, dropout 0.4 | Comparable accuracy |
| Logistic Regression | statsmodels GLM, Binomial family | ~96.5% accuracy |

**Key finding:** The neural network matched logistic regression — an important result showing that for low-dimensional, linearly separable problems, simpler models are equally effective and cheaper to deploy.

---

### 🖼️ Exercise 8 — Animal Image Classification with ResNet50 (Transfer Learning)

**Problem:** Classify animal images using a pre-trained CNN without training from scratch.

**What I built:**
- End-to-end ResNet50 inference pipeline using `torchvision`
- ImageNet class index loaded from JSON; top-5 class probabilities reported per image

**Stack:** `resnet50(weights=ResNet50_Weights.DEFAULT)` → `.eval()` mode → softmax → top-5 ranking

**Why it matters:** Transfer learning is the standard industry approach for computer vision. Loading, preprocessing, running inference, and interpreting a pre-trained CNN is a core production ML engineering skill.

---

### 📈 Exercise 9 — NYSE Trading Volume Forecasting with Month Effects

**Problem:** Build a lag-5 AR model to forecast log NYSE trading volume, then test whether a 12-level month indicator improves performance.

**Features:** 5 lags of `DJ_return`, `log_volume`, `log_volatility` (15 total) + optional month dummies

| Model | Test R² Change |
|---|---|
| Linear AR(5) baseline | — |
| Linear AR(5) + month dummies | +0.4% |

**Key finding:** Monthly seasonality adds only **0.4%** to R² — the 5-day lag structure already captures the relevant dynamics, making month dummies redundant for this task.

---

### 🔁 Exercise 10 — Linear AR via RNN Sequence Flattening

**Problem:** Replicate a linear AR(5) model by flattening RNN-style input sequences and compare to the standard implementation.

**Architecture:**
```
Input(5×3) → Flatten → Linear(15→20) → Dropout(0.5) → Linear(20→1)
```

**Training:** RMSprop (lr=0.001) · 20 epochs · PyTorch Lightning

| Approach | Test R² |
|---|---|
| Standard Linear AR(5) (textbook, p.461) | 0.4129 |
| Flattened sequence linear AR | **0.4051** |

**Key finding:** Both approaches yield virtually identical R² — confirming that sequence flattening is a valid and more modular alternative that integrates naturally into deep learning pipelines.

---

### 🌊 Exercise 11 — Non-Linear AR via ReLU Sequence Model

**Problem:** Add a ReLU non-linearity to the flattened sequence model and test whether non-linear patterns in NYSE volume exist.

**Architecture:**
```
Input(5×3) → Flatten → Linear(15→32) → ReLU → Dropout(0.5) → Linear(32→1)
```

| Model | Test R² |
|---|---|
| Linear AR (Exercise 10) | 0.4051 |
| Non-Linear AR | **0.422** |

**Key finding:** A single ReLU layer improved R² by **~1.7 percentage points**, confirming mild non-linear dynamics in NYSE volume that purely linear models miss.

---

### 📅 Exercise 12 — RNN with Day-of-Week Signal for NYSE Forecasting

**Problem:** Inject `day_of_week` as an additional input into the RNN and measure the effect on predictive performance.

**Architecture:** Custom `nn.RNN` (12 hidden units) → Dropout(0.1) → Dense(1)

**Engineering challenge:** Aligning one-hot day-of-week dummies with the 5-lag RNN input tensor required careful index alignment and reshape logic — implemented without data leakage.

**Training:** Adam (lr=0.0001) · 200 epochs · batch size 128

**Why it matters:** Injecting structured time signals (weekday, holiday indicators) into sequence models is standard practice in production financial forecasting systems.

---

### 🎬 Exercise 13 — IMDb Sentiment Classification: Architecture Search

**Problem:** Compare four neural network configurations for binary sentiment classification (positive/negative review).

**Configurations tested:**

| Model | Units/Layer | Dropout | Test Accuracy |
|---|---|---|---|
| 2-layer NN | 32 | ✅ 30% | **87%** |
| 2-layer NN | 32 | ❌ None | 85% |
| 2-layer NN | 64 | ✅ 30% | 86% |
| 2-layer NN | 64 | ❌ None | 86% |

**Training:** RMSprop · 30 epochs · batch size 512 · PyTorch Lightning

**Key findings:**
- **Dropout consistently outperforms no-dropout** — 87% vs. 85% at 32 units confirms regularization's role on high-dimensional text data
- **More units ≠ better** — 32-unit dropout (87%) outperformed 64-unit dropout (86%), suggesting the simpler architecture is better matched to the task
- **Optimal config: 32 units + 30% dropout** — faster to train, simpler to deploy, highest accuracy

---

## 📁 Repository Structure

```
📦 ISLR2-Chapter10-DeepLearning/
│
├── 📓 ISLR2_chapter_10.ipynb   # Full PyTorch notebook — all 8 exercises
├── 🌐 Chapter_10.html          # Web-viewable rendered version
└── 📋 README.md                # This file
```

---

## ▶️ Setup & Run

**Step 1 — Install dependencies:**
```bash
pip install torch torchvision torchmetrics pytorch-lightning torchinfo
pip install numpy pandas matplotlib scikit-learn statsmodels islp jupyter
```

**Step 2 — Clone and launch:**
```bash
git clone https://github.com/shadalishah/ISLR2-Chapter10-DeepLearning.git
cd ISLR2-Chapter10-DeepLearning
jupyter notebook ISLR2_chapter_10.ipynb
```

> **Note:** Exercise 8 requires `torchvision` and internet access for ResNet50 pre-trained weights. Exercises 9–12 use the NYSE dataset from the ISLP library.

---

## 🙏 Acknowledgements

Special thanks to **[Karim Aboussel Ham](https://github.com/KarimABOUSSELHAM)** whose repository [ISLP-applied-solutions](https://github.com/KarimABOUSSELHAM/ISLP-applied-solutions) provided helpful code examples and guidance throughout these exercises.

---

## 👤 About the Author

**Shad Ali Shah**
MPhil Economics Student — School of Economics, Quaid-i-Azam University, Islamabad
Passionate about the intersection of **Economics, Data Science, and Machine Learning**

🔗 [LinkedIn](https://www.linkedin.com/in/shad-ali-shah-6439ab339/) &nbsp;|&nbsp; 🐙 [GitHub](https://github.com/shadalishah)

---

## 📚 Reference

> James, G., Witten, D., Hastie, T., Tibshirani, R., & Taylor, J. (2023).
> *An Introduction to Statistical Learning with Applications in Python*. Springer.
> [https://www.statlearning.com](https://www.statlearning.com)
