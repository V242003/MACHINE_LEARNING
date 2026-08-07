# 📈 Multiple Linear Regression — From Scratch vs Scikit-Learn

A hands-on implementation of **Multiple Linear Regression (MLR)**, built two ways:

1. **Scikit-Learn** — the standard, production-ready approach
2. **From Scratch (NumPy)** — deriving and coding the **Normal Equation** manually

The goal isn't just to build a working model, but to fully understand the **mathematics behind it** — from the cost function to the closed-form solution for the regression coefficients.

---

## 🎯 Objectives

- Understand how Multiple Linear Regression extends Simple Linear Regression to multiple features
- Derive the Normal Equation from first principles
- Implement MLR from scratch using only NumPy
- Compare the from-scratch results against Scikit-Learn's implementation
- Build intuition for how regression coefficients are actually computed, not just how to call `.fit()`

---

## 📚 Topics Covered

- Simple vs. Multiple Linear Regression
- Mathematical formulation and matrix representation
- Prediction equation and error vector
- Sum of Squared Errors (SSE) / cost function
- Normal Equation derivation
- Computing coefficients manually
- Predicting with learned parameters
- Benchmarking against Scikit-Learn

---

## 📖 Mathematical Background

**Simple Linear Regression** (one feature):

$$y = mx + b$$

**Multiple Linear Regression** (multiple features — e.g. predicting *Salary* from *CGPA*, *IQ*, and *Gender*):

$$y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_n x_n$$

| Term | Meaning |
|------|---------|
| $\beta_0$ | Intercept (bias) |
| $\beta_1 \dots \beta_n$ | Feature coefficients |
| $x_1 \dots x_n$ | Input features |

Each coefficient $\beta_i$ represents how much the output changes for a one-unit change in $x_i$, holding all other features constant.

### Matrix Representation

$$\hat{Y} = X\beta$$

- $X$ — feature matrix (with a bias column of 1s)
- $\beta$ — coefficient vector
- $\hat{Y}$ — predicted output

### Error Term

$$e = Y - \hat{Y}$$

$$E = e^{T}e = (Y - X\beta)^{T}(Y - X\beta)$$

### The Normal Equation

Minimizing the cost function with respect to $\beta$ (setting the derivative to zero) gives:

$$X^{T}X\beta = X^{T}Y$$

$$\boxed{\beta = (X^{T}X)^{-1}X^{T}Y}$$

This closed-form solution is the **Normal Equation** — it directly computes the optimal coefficients without iterative optimization.

---

## 🛠 Implementation

### 1. Scikit-Learn Approach
- Import `LinearRegression`
- Fit the model on training data
- Generate predictions
- Inspect `.coef_` and `.intercept_`

### 2. From-Scratch Approach (NumPy)
- Build the feature matrix and prepend a bias column
- Compute $(X^{T}X)^{-1}$
- Compute $X^{T}Y$
- Solve for $\beta$
- Predict using $\hat{Y} = X\beta$

---

## 🔍 Scikit-Learn vs. From Scratch

| | Scikit-Learn | From Scratch |
|---|--------------|---------------|
| **Implementation** | Ready-made, library-based | Manual, matrix-based |
| **Effort** | One-line model fitting | Explicit linear algebra |
| **Internals** | Abstracted / optimized | Fully transparent |
| **Best for** | Fast development | Understanding the math |

---

## ▶️ Getting Started

```bash
# Clone the repository
git clone <repo-url>
cd MultipleLinearRegression

# Install dependencies
pip install numpy pandas matplotlib scikit-learn

# Launch the notebook
jupyter notebook MultipleLinearRegression.ipynb
```

---

## 📂 Project Structure

```
├── MultipleLinearRegression.ipynb   # Main notebook: theory + implementation
└── README.md                        # Project documentation
```

---

## 🧠 Learning Outcomes

By the end of this project, you should be able to:

- Explain why MLR fits a hyperplane rather than a line
- Represent regression problems in matrix form
- Derive the Normal Equation from the cost function
- Compute regression coefficients manually with NumPy
- Validate a from-scratch implementation against Scikit-Learn

---

## 🚀 Tech Stack

`Python` · `NumPy` · `Pandas` · `Matplotlib` · `Scikit-Learn`

---

## 🙏 Acknowledgement

The mathematical derivation in this notebook follows **[CampusX](https://www.youtube.com/watch?v=VmZWXzxmNrE&list=PLKnIA16_Rmvbr7zKYQuBfsVkjoLcJgxHH&index=55)** — huge thanks for making the underlying math so accessible.

---

## 📺 Reference

[CampusX — Multiple Linear Regression Playlist](https://www.youtube.com/watch?v=VmZWXzxmNrE&list=PLKnIA16_Rmvbr7zKYQuBfsVkjoLcJgxHH&index=55)
