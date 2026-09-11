# Sketched Logistic Regression

A Numerical Analysis project exploring **second-order optimization methods for Logistic Regression** using Hessian-vector products, Conjugate Gradient, preconditioning, and randomized sketching.

## 🚀 Overview

This project compares different optimization techniques for regularized Logistic Regression:

* Gradient Descent
* Exact Newton Method
* Newton-CG
* Preconditioned Newton-CG (PCG)
* Sketched Newton

The experiments are performed mainly on the **UCI Spambase dataset**, with additional synthetic experiments to study conditioning and scalability.

## 🧠 Key Idea

Instead of explicitly computing and inverting the Hessian matrix, Newton-CG uses **Hessian-vector products** and Conjugate Gradient to solve the Newton system more efficiently.

Randomized sketching is also explored to approximate the Hessian and study the trade-off between **computational cost and optimization accuracy**.

## 📊 Dataset

**Spambase Dataset**

* 4,601 samples
* 57 features
* Binary classification
* Labels: Spam / Non-Spam
* Features are standardized before training.

## 🔬 Experiments

The project evaluates:

* Loss vs. iterations
* Loss vs. runtime
* Training accuracy
* Sketch-size trade-offs
* Hessian approximation quality
* Conditioning effects
* CG vs. PCG performance

On Spambase, Exact Newton reaches a low loss in very few iterations, while Newton-CG closely follows it. Sketching shows a trade-off where larger sketches generally provide a better approximation to Newton's method.

## 🛠️ Technologies

* Python
* NumPy
* Matplotlib
* Jupyter Notebook
* Numerical Linear Algebra
* Logistic Regression

## 📁 Project Structure

```text
Sketched-Logistic-Regression/
│
├── code.py
├── Z_project.py
├── project_analysis.ipynb
├── spambase/
├── *.png
└── context.md
```

## ▶️ Run

```bash
pip install numpy matplotlib jupyter
jupyter notebook project_analysis.ipynb
```

Run the notebook cells sequentially to reproduce the experiments and plots.

## 📌 Conclusion

The project demonstrates how **Newton-CG, preconditioning, and sketching** can reduce the computational challenges associated with second-order optimization. It also highlights an important trade-off between convergence speed, runtime, and Hessian approximation quality.

