# Overfitting vs Ridge Regression

A hands-on notebook that shows **overfitting** in polynomial regression, explains **why** it happens (bias-variance tradeoff), and demonstrates how **regularization — specifically Ridge Regression** — fixes it.

---

## 1. The Bias-Variance Tradeoff

Every model's prediction error can be broken down into three parts:

```
Total Error = Bias² + Variance + Irreducible Noise
```

- **Bias** — error from a model being *too simple* to capture the real pattern in the data.
  A straight line trying to fit a curvy sine wave has high bias — it systematically misses the shape no matter how much data you give it. This is **underfitting**.

- **Variance** — error from a model being *too sensitive* to the specific training data it saw.
  A high-degree polynomial can bend and twist to pass through every single training point — including the noise. Give it a slightly different training set and the fitted curve changes drastically. This is **overfitting**.

- **Irreducible noise** — randomness in the data itself that no model can remove.

The tradeoff:

| Model complexity | Bias | Variance | Result |
|---|---|---|---|
| Too simple (e.g. straight line) | High | Low | Underfitting |
| Too complex (e.g. degree-15 polynomial) | Low | High | Overfitting |
| Just right | Balanced | Balanced | Good generalization |

The goal isn't to eliminate bias or variance completely — it's to find the sweet spot that minimizes their sum, so the model generalizes well to **unseen (test) data**, not just the training set.

This notebook makes that tradeoff visible:
- **Linear Regression (degree 1)** → high bias, low variance → underfits
- **Polynomial Regression (degree 15)** → low bias, very high variance → overfits
- **Ridge Regression (degree 15 + regularization)** → controlled variance → better balance

---

## 2. Regularization

Regularization is a technique to **reduce variance** by discouraging a model from becoming too complex, without changing the model's structure (it still uses degree-15 polynomial features).

It works by adding a **penalty term** to the loss function that the model minimizes:

```
Loss = Prediction Error  +  Penalty on large coefficients
```

Normally, ordinary least squares regression only tries to minimize prediction error on the training set. Given enough flexibility (like a degree-15 polynomial), it will happily assign huge, wild coefficients to fit every noisy point exactly — that's what causes the extreme swings you see in the overfit curve.

Regularization pushes back against that by penalizing large coefficients, forcing the model to keep them small and "smoother" unless the data really justifies a big value.

Two common types:
- **Ridge (L2 regularization)** — penalizes the *sum of squared* coefficients. Shrinks coefficients smoothly toward zero but rarely to exactly zero.
- **Lasso (L1 regularization)** — penalizes the *sum of absolute* coefficients. Can shrink some coefficients all the way to zero (useful for feature selection).

This notebook uses **Ridge**.

---

## 3. Ridge Regression

Ridge Regression modifies the ordinary least squares loss function by adding an L2 penalty:

```
Loss = Σ(yᵢ - ŷᵢ)²  +  alpha · Σ(coefⱼ²)
```

- The first term is the usual squared error (how well the model fits the training data).
- The second term penalizes large coefficients, scaled by a hyperparameter **alpha**.

**Effect of alpha:**

| alpha value | Effect |
|---|---|
| `alpha = 0` | No penalty — identical to plain Linear/Polynomial Regression (overfits) |
| small `alpha` (e.g. 0.01) | Light penalty — reduces overfitting while still capturing the pattern |
| large `alpha` (e.g. 10, 100) | Heavy penalty — coefficients shrink a lot, model becomes too simple → underfitting |

So `alpha` directly controls where you sit on the bias-variance tradeoff:
- ↓ alpha → ↓ bias, ↑ variance (toward overfitting)
- ↑ alpha → ↑ bias, ↓ variance (toward underfitting)

The right alpha is usually found via cross-validation (`RidgeCV` in scikit-learn), though this notebook just demonstrates the effect with a fixed value.

---

## 4. What's in the notebook

1. **Dataset** — `overfit_dataset.csv`: 30 noisy points sampled from a sine curve (`x` in [0, 1], `y = sin(2πx) + noise`). Small and noisy on purpose — that's what makes overfitting easy to trigger and see.

2. **Train/test split** — 80/20 split using `train_test_split`.

3. **Linear Regression (baseline)**
   A plain straight-line fit. High bias, underfits the sine pattern — test R² ≈ 0.60.

4. **Polynomial Regression, degree 15 (overfit)**
   Features expanded with `PolynomialFeatures(degree=15)`, fit with plain `LinearRegression`.
   Matches training points almost perfectly (low bias) but swings wildly between them and at the edges (high variance) — test R² ≈ **-1.04** (worse than just predicting the mean!).

5. **Ridge Regression (the fix)**
   Same degree-15 polynomial features, fit with `Ridge(alpha=0.01)` instead of plain linear regression.
   The L2 penalty shrinks the oversized coefficients from the degree-15 fit, smoothing the curve and dramatically improving generalization — test R² ≈ **0.85**.

6. **Visualizations**
   - Dataset scatter plot
   - Train vs. test scatter with fitted line (linear model)
   - Train vs. test scatter with fitted curve (overfit polynomial model)
   - Side-by-side comparison: Linear Regression / Polynomial (Overfit) / Ridge (Controlled)

---

## 5. Results summary

| Model | Bias | Variance | Test R² |
|---|---|---|---|
| Linear Regression | High | Low | ~0.60 |
| Polynomial (degree 15, no regularization) | Low | Very High | ~-1.04 |
| Ridge (degree 15, alpha=0.01) | Low-Medium | Controlled | ~0.85 |

**Takeaway:** Same model complexity (degree-15 features) — the only difference is the L2 penalty. Regularization doesn't reduce the model's capacity to represent complex functions; it just discourages it from *abusing* that capacity to memorize noise.

---

## Requirements

```
pandas
numpy
matplotlib
scikit-learn
```

## How to run

Open the notebook in Jupyter or Google Colab and run all cells in order. Make sure `overfit_dataset.csv` is in the same directory (or update the file path in the `pd.read_csv(...)` cell).
