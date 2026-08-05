# Linear Regression From Scratch

## Overview

This project demonstrates how to implement **Simple Linear Regression
from scratch using Python** without using machine learning libraries
like Scikit-Learn. The model calculates the slope (`m`) and intercept
(`b`) using mathematical formulas and visualizes the best-fit regression
line.

## Features

-   Create a noisy dataset
-   Visualize data using a scatter plot
-   Calculate mean of X and Y
-   Compute slope (m)
-   Compute intercept (b)
-   Plot the best-fit regression line
-   No Scikit-Learn used

## Technologies Used

-   Python
-   NumPy
-   Matplotlib

## Mathematical Formula

Slope:

``` text
m = Σ((x - x̄)(y - ȳ)) / Σ((x - x̄)²)
```

Intercept:

``` text
b = ȳ - m × x̄
```

Regression Equation:

``` text
y = mx + b
```

## Project Structure

``` text
Linear-Regression-From-Scratch/
│── Linear_Regression.ipynb
│── README.md
│── requirements.txt
└── images/
    └── regression_output.png
```

## Installation

``` bash
pip install numpy matplotlib
```

## Run

Open the Jupyter notebook and execute all cells sequentially.

## Output

The notebook produces: - Scatter plot of the dataset - Calculated values
of m and b - Best-fit regression line - Final regression equation

## Future Improvements

-   Multiple Linear Regression
-   Gradient Descent implementation
-   R² Score calculation
-   Mean Squared Error (MSE)

## Author

Varsha Rathour
