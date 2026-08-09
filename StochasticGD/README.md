# SGD vs Linear Regression — From Scratch

This project compares scikit-learn's `LinearRegression` against a **Stochastic Gradient Descent (SGD) regressor built from scratch** on a synthetic 2-feature dataset, with visualizations at every stage.

## Dataset

`sgd_dataset.csv` — 2000 synthetic samples generated as:

```
target = 3.5 * feature_1 - 2.1 * feature_2 + 7.0 + noise
```

| Column     | Description                          |
|------------|---------------------------------------|
| feature_1  | Random value, range [-10, 10]         |
| feature_2  | Random value, range [-10, 10]         |
| target     | Linear combination of both features + Gaussian noise (std=4) |

**Ground truth parameters:** `w1 = 3.5`, `w2 = -2.1`, `b = 7.0`

## What the notebook does

1. **Load & split data** — reads the CSV, splits into train/test (80/20, `random_state=42`).
2. **Baseline model** — fits scikit-learn's `LinearRegression`, reports coefficients, intercept, MSE, and R².
3. **Data visualization**
   - Raw scatter plots of each feature vs. target ("before" fit)
   - Combined scatter of both features vs. target, overlaid with independent best-fit lines (`np.polyfit`)
   - Predicted vs. actual scatter for the sklearn model ("after" fit), with a red dashed perfect-fit reference line
4. **SGD from scratch** — a custom `SGD` class implementing single-sample stochastic gradient descent:
   - Initializes weights/bias at zero
   - On each iteration, samples one random training row, computes the prediction error, and updates weights/bias via the gradient of squared error
   - Tracks `loss_history`, `weights_history`, and `bias_history` across iterations for visualization
5. **SGD evaluation** — predicts on the test set and plots predicted vs. actual, same style as the sklearn comparison.

## Requirements

```
numpy
pandas
matplotlib
scikit-learn
```

## Usage

```bash
pip install numpy pandas matplotlib scikit-learn
jupyter notebook Untitled5.ipynb
```

Make sure `sgd_dataset.csv` is in the same directory as the notebook (or update the `pd.read_csv` path).

## Notes / things to tune

- **Learning rate & iterations**: the custom SGD uses `learning_rate=0.001`, `n_iterations=3000`. Since features range up to ±10, a learning rate that's too high can cause the weights to diverge — lower it further if you see loss exploding instead of shrinking.
- **Convergence check**: after fitting, compare `sgd.weights` / `sgd.bias` against the true generating parameters (`3.5, -2.1, 7.0`) and against `model.coef_` / `model.intercept_` from the sklearn baseline — they should land close to each other.
- **Suggested extra plots** (not yet in the notebook, easy to add):
  - Loss curve (`sgd.loss_history`, ideally with a rolling average to smooth the per-sample noise)
  - Weight convergence over iterations (`sgd.weights_history`, `sgd.bias_history`) with dashed reference lines at the true values

## Files

- `Untitled5.ipynb` — the notebook (also exported as `untitled5.py`)
- `sgd_dataset.csv` — the synthetic dataset

---

## How SGD Works (Explanation)

### What does regression actually do?

We have `feature_1` and `feature_2`, and want to predict `target`. The formula:

```
prediction = w1*feature_1 + w2*feature_2 + b
```

`w1`, `w2` (**weights**) and `b` (**bias**) are the 3 numbers the model needs to learn. They start at **0**:

```python
self.weights = np.zeros(X.shape[1])
self.bias = 0
```

### How do we find the right weights? → Gradient Descent

Picture yourself standing on a mountain (bad weights = high up the slope), and the valley below has the lowest possible **error/loss**. You want to walk downhill, one step (iteration) at a time.

**Loss** measures how far the prediction is from the actual target:

```
error = actual_target - prediction
loss  = error²     # squared so negative and positive errors both count as "bad"
```

At each step, we nudge each weight up or down in whichever direction reduces the loss — that direction is called the **gradient**.

### What does "Stochastic" mean?

- **Regular Gradient Descent**: looks at the *entire* dataset (all ~1600 rows) before taking a single step. Accurate, but slow, since every step requires processing the whole dataset.
- **Stochastic Gradient Descent (SGD)**: picks just **one random row**, computes the error from that single point, and updates the weights immediately. Much faster per step, but "noisier" — each step doesn't always move in the perfect direction.

That's why an SGD loss curve doesn't drop in a smooth straight line — it **oscillates** up and down while trending downward overall.

### How this looks in code

```python
idx = random.randint(0, n - 1)          # pick one random sample
y_hat = np.dot(X.iloc[idx], self.weights) + self.bias   # prediction

error = y.iloc[idx] - y_hat
bias_gradient = -2 * error
weights_gradient = -2 * error * X.iloc[idx]

self.bias -= self.learning_rate * bias_gradient
self.weights -= self.learning_rate * weights_gradient
```

- **`learning_rate`** controls how big each step is. Too large → weights can diverge (blow up). Too small → learning takes forever.
- **`n_iterations`** = how many times this update process repeats. More iterations generally means better learning (up to a point).

### What happens at the end?

After 3000 iterations, `sgd.weights` and `sgd.bias` gradually converge close to the **true generating values**:

```
true w1 = 3.5, true w2 = -2.1, true b = 7.0
```

To visualize this convergence:
- **Loss curve** (`sgd.loss_history`) — shows the loss decreasing over time (noisily, but trending down)
- **Weight convergence plot** (`sgd.weights_history`, `sgd.bias_history`) — shows each weight moving toward its true value, iteration by iteration
