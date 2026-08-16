# Mini-Batch Gradient Descent from Scratch — Diabetes Dataset

This project implements **Mini-Batch Gradient Descent** from scratch (using only NumPy and Pandas) to solve a linear regression problem on the classic **Diabetes** dataset from `sklearn.datasets`. It walks through exploratory data analysis, manual feature scaling, a step-by-step derivation of the gradient descent update rule, and finally wraps the algorithm into a reusable Python class.

## Dataset

The [Diabetes dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset) contains 442 patient records with 10 baseline physiological features (age, sex, BMI, blood pressure, and six blood serum measurements), and a continuous target representing disease progression one year after baseline.

- **Samples:** 442
- **Features:** 10 (all mean-centered and scaled by `sklearn`)
- **Target:** Quantitative measure of disease progression

## Project Workflow

### 1. Exploratory Data Analysis (EDA)
- Loaded the dataset into a Pandas DataFrame and inspected shape, types, summary statistics, and missing values.
- Visualized the distribution of the target variable with a histogram.
- Plotted histograms for all features.
- Created scatter plots of each feature against the target to visually inspect relationships.
- Generated a correlation matrix (heatmap) and ranked features by their correlation with the target.

### 2. Train/Test Split & Feature Scaling
- Split the data into training (80%) and test (20%) sets using `train_test_split`.
- Manually standardized features (zero mean, unit variance) using the **mean and standard deviation computed only from the training set**, then applied the same transformation to the test set to avoid data leakage.

### 3. Gradient Descent — Built from First Principles
The notebook builds up mini-batch gradient descent step by step before wrapping it in a class:

1. **Initialization** — weights (`w`) initialized to zeros, bias (`b`) initialized to `0.0`.
2. **Shuffling** — training data is randomly shuffled each epoch.
3. **Batching** — data is split into mini-batches (default `batch_size=32`).
4. **Forward pass** — predictions computed as `y_pred = X_batch @ w + b`.
5. **Loss** — Mean Squared Error (MSE) computed per batch.
6. **Gradients** — gradients of the loss with respect to `w` and `b` derived and computed manually:
   - `dw = (2/m) * X_batch.T @ (y_pred - y_batch)`
   - `db = (2/m) * sum(y_pred - y_batch)`
7. **Parameter update** — weights and bias updated using the gradient descent rule:
   - `w = w - learning_rate * dw`
   - `b = b - learning_rate * db`
8. **Epoch loop** — the above steps repeat for a configured number of epochs (default `1000`), tracking a weighted average loss per epoch.

This manual walkthrough is first done as a single pass/single batch example (for intuition), then as a full training loop, and finally refactored into a class.

### 4. `MiniBatchGradientDescent` Class

A clean, reusable implementation of the algorithm as a scikit-learn-style estimator:

```python
model = MiniBatchGradientDescent(
    learning_rate=0.005,
    batch_size=32,
    epochs=1000
)

model.fit(X_train_scaled, y_train)
y_test_pred = model.predict(X_test_scaled)

mse = model.mse(y_test, y_test_pred)
r2 = model.r2_score(y_test, y_test_pred)
```

**Class methods:**

| Method | Description |
|---|---|
| `fit(X, y)` | Trains the model using mini-batch gradient descent, storing loss history per epoch |
| `predict(X)` | Returns predictions for new data: `X @ w + b` |
| `mse(y_true, y_pred)` | Computes Mean Squared Error |
| `r2_score(y_true, y_pred)` | Computes the R² (coefficient of determination) |

### 5. Evaluation & Visualization
- **Loss curve** — MSE loss plotted against epoch number to visualize convergence.
- **Test set MSE** and **R² score** reported.
- **Actual vs. Predicted** scatter plot with a diagonal reference line for perfect predictions.
- **Residual plot** — predicted values vs. residuals, with a horizontal reference line at zero, used to check for patterns indicating model bias.

## Requirements

```
numpy
pandas
matplotlib
scikit-learn
```

Install with:

```bash
pip install numpy pandas matplotlib scikit-learn
```

## Usage

1. Run the script/notebook top to bottom. It will:
   - Load and explore the dataset
   - Split and scale the data
   - Walk through a manual single-batch gradient step (for learning purposes)
   - Train a full mini-batch gradient descent model over 1000 epochs
   - Refit using the `MiniBatchGradientDescent` class
   - Print final Test MSE and R² score
   - Display all EDA and evaluation plots

## Key Learning Objectives

- Understanding how mini-batch gradient descent works under the hood, without relying on `sklearn`'s built-in optimizers
- Manual derivation and implementation of gradients for a linear regression loss (MSE)
- Correct practice for train/test feature scaling (avoiding data leakage)
- Structuring a from-scratch ML algorithm as a reusable, object-oriented class
- Standard model evaluation techniques: loss curves, MSE, R², actual-vs-predicted plots, and residual plots

## Notes

- This implementation is for **educational purposes** — it demonstrates the mechanics of gradient descent rather than being optimized for production use.
- No regularization (L1/L2) is applied; the model is a plain linear regression trained via mini-batch SGD.
- Original notebook auto-generated from Google Colab.
