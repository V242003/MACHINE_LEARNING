# 📈 Multiple Linear Regression using Gradient Descent — From Scratch

A from-scratch implementation of **Multiple Linear Regression** trained with **Gradient Descent**, built on the Advertising dataset.

The goal isn't just to predict sales — it's to understand *how* Gradient Descent actually learns model parameters, instead of treating a pre-built ML algorithm as a black box. To confirm the implementation is correct, results are benchmarked directly against **`scikit-learn`'s built-in `LinearRegression`**, trained on the same data.

---

## 📌 Overview

This project predicts **Sales** from two advertising channels:

- 📺 TV Advertising Spend
- 📻 Radio Advertising Spend

**Model:**

```
ŷ = w₁·X₁ + w₂·X₂ + b
```

| Symbol | Meaning |
|---|---|
| X₁ | TV advertising spend |
| X₂ | Radio advertising spend |
| w₁ | Learned weight for TV |
| w₂ | Learned weight for Radio |
| b | Bias / intercept |
| ŷ | Predicted Sales |

---

## 🎯 Objectives

- [x] Understand Multiple Linear Regression
- [x] Apply feature scaling (standardization)
- [x] Implement a cost function from scratch
- [x] Derive and compute partial derivatives
- [x] Implement Gradient Descent from scratch
- [x] Visualize convergence and the regression hyperplane in 3D
- [x] Train `scikit-learn`'s built-in `LinearRegression` on the same dataset
- [x] Compare its coefficients and predictions against the from-scratch Gradient Descent model

---

## 📊 Dataset

This project uses the **Advertising Dataset** — advertising spend across media channels alongside resulting product sales.

| Feature | Description |
|---|---|
| TV | Advertising expenditure on TV |
| Radio | Advertising expenditure on Radio |
| Newspaper | Advertising expenditure on Newspaper |
| Sales | Product sales (target variable) |

> Only **TV** and **Radio** are used for training, so the result can be visualized as a 3D plane.

**Source:** [Kaggle — Advertising Dataset](https://www.kaggle.com/datasets/ashydv/advertising-dataset)

**Why TV and Radio?** Two features let the whole regression be plotted in 3D — TV on the X-axis, Radio on the Y-axis, Sales on the Z-axis — so you can literally *see* the model as a plane fitting a cloud of points.

---

## ⚙️ Methodology

```
Dataset
  → Data Exploration
  → Select TV + Radio
  → Feature Scaling
  → Initialize w₁, w₂, b
  → Predict
  → Compute Error
  → Compute Cost
  → Compute Gradients
  → Update Parameters
  → Repeat for N iterations
  → Final Regression Hyperplane
```

---

## 📏 Feature Scaling

Inputs are standardized before training:

```
X_scaled = (X - μ) / σ
```

where **μ** is the feature mean and **σ** is the feature standard deviation. This keeps TV and Radio on comparable scales, which helps Gradient Descent converge faster and more reliably.

---

## 💰 Cost Function

Mean squared error (halved for a cleaner gradient):

```
J(w₁, w₂, b) = (1 / 2n) · Σ (ŷᵢ - yᵢ)²
```

Gradient Descent's job is to find the (w₁, w₂, b) that minimize this.

---

## 🔽 Gradient Descent

**Partial derivatives:**

```
∂J/∂w₁ = (1/n) · Σ (ŷ - y)·X₁
∂J/∂w₂ = (1/n) · Σ (ŷ - y)·X₂
∂J/∂b  = (1/n) · Σ (ŷ - y)
```

**Parameter updates** (learning rate α):

```
w₁ = w₁ - α · ∂J/∂w₁
w₂ = w₂ - α · ∂J/∂w₂
b  = b  - α · ∂J/∂b
```

---

## 🏗️ Implementation

Gradient Descent is implemented from scratch as a small Python class:

```python
class GradientDescent:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        ...

    def predict(self, X):
        """ŷ = w₁X₁ + w₂X₂ + b"""
        ...

    def cost_function(self, y_true, y_pred):
        """Mean squared error"""
        ...

    def fit(self, X, y):
        """Runs the full training loop: predict → error → cost →
        gradients → parameter update → repeat"""
        ...
```

| Method | Purpose |
|---|---|
| `predict()` | Computes ŷ for given inputs and current parameters |
| `cost_function()` | Evaluates current model cost (MSE) |
| `fit()` | Runs the complete Gradient Descent training loop |

---

## 📉 Cost vs. Iterations

Cost is logged at every iteration during training. A steadily decreasing cost curve confirms the model is converging toward an optimal solution rather than diverging or stalling.

---

## 📊 Visualizations

**1. Original Data (3D Scatter)**
Raw observations plotted with TV on X, Radio on Y, and Sales on Z — showing the underlying relationship before any model is fit.

**2. Regression Hyperplane**
Once trained, the learned parameters define a plane in the same 3D space, showing the model's predicted Sales across the full TV/Radio range.

**3. Cost vs. Iterations**
A line plot of cost per training iteration, used to visually confirm convergence.

---

## 🔬 Gradient Descent (from scratch) vs. Scikit-learn

This is the core validation step of the project: the custom Gradient Descent model is trained side-by-side with `sklearn.linear_model.LinearRegression` on the **same dataset**, and the two are compared directly.

**Two things are compared:**

1. **Learned coefficients** (w₁ for TV, w₂ for Radio, and the bias/intercept b)
2. **Predicted Sales values** on the same input data

**Why they aren't identical out of the box:**
The from-scratch model is trained on **standardized (scaled)** features, while `sklearn`'s `LinearRegression` is fit directly on the **original, unscaled** features. So the two sets of raw weights aren't on the same scale and can't be compared as-is.

To fix this, the Gradient Descent weights are **converted back to the original feature scale** (unscaled) before comparison — after which both models should describe the same line/plane.

**Example comparison:**

| | w₁ (TV) | w₂ (Radio) | b (Intercept) |
|---|---|---|---|
| `sklearn` LinearRegression | ~0.0545 | ~0.1010 | ~4.63 |
| Gradient Descent (rescaled) | ~0.0545 | ~0.1010 | ~4.63 |

*(Illustrative — actual values depend on your train/test split and hyperparameters; the notebook prints the real numbers.)*

```
Linear Regression (sklearn)  ──┐
                                ├──▶ Predictions & coefficients compared
Gradient Descent (scratch)   ──┘        → difference ≈ negligible
```

**Result:** after rescaling, the coefficients and predictions from both approaches match almost exactly — confirming the from-scratch Gradient Descent implementation converges to the same optimum as `sklearn`'s closed-form solution.

---

## 🛠️ Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Kaggle Dataset
- Google Colab

---

## 📂 Project Structure

```
Multiple-Linear-Regression-Gradient-Descent/
│
├── Multiple_Linear_Regression_GD.ipynb
├── README.md
└── images/
    ├── original_data.png
    ├── regression_plane.png
    └── cost_vs_iterations.png
```

---

## 🚀 How to Run

**1. Clone the repository**

```bash
git clone https://github.com/<your-username>/Multiple-Linear-Regression-Gradient-Descent.git
cd Multiple-Linear-Regression-Gradient-Descent
```

**2. Install dependencies**

```bash
pip install numpy pandas matplotlib scikit-learn kagglehub
```

**3. Open the notebook**

Open `Multiple_Linear_Regression_GD.ipynb` in Google Colab or Jupyter Notebook.

**4. Run the cells sequentially**

1. Load the dataset
2. Explore the data
3. Visualize the data in 3D
4. Scale the features
5. Implement Gradient Descent
6. Train the model
7. Plot the cost curve
8. Visualize the final regression hyperplane
9. Compare the result with `sklearn`'s Linear Regression

---

## 💡 Key Learnings

- Linear Regression can be solved iteratively via Gradient Descent, not just in closed form.
- Feature scaling materially affects Gradient Descent's convergence speed.
- Gradient Descent doesn't "draw" a line or plane directly — it iteratively updates parameters to minimize cost, and the plane emerges from those parameters.
- A correct from-scratch implementation converges to essentially the same solution as `sklearn`'s Linear Regression.
- Visualizing cost curves and regression planes makes the optimization process far more intuitive.

---

## 🔮 Future Improvements

- [ ] Implement Batch, Stochastic, and Mini-Batch Gradient Descent
- [ ] Animate the regression plane's movement across training iterations
- [ ] Experiment with different learning rates and convergence behavior
- [ ] Add a train/test split for proper evaluation
- [ ] Report R², MAE, and RMSE
- [ ] Extend the model to include Newspaper advertising
- [ ] Vectorize Gradient Descent using matrix operations

---

## 👩‍💻 Author

**Varsha Rathour**
M.Tech, Artificial Intelligence
Dr. B.R. Ambedkar National Institute of Technology, Jalandhar

---

⭐ If you found this project useful, consider giving the repository a star!
