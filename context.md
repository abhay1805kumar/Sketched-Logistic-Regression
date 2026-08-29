# ME542 Numerical Analysis Project Context

## 1. Project Goal

This project is a Numerical Analysis term project about improving second-order optimization methods for logistic regression using ideas from numerical linear algebra.

The main goal is to show how Newton-type methods can be made more practical by:

- avoiding explicit Hessian inversion
- using Hessian-vector products
- using Conjugate Gradient (CG)
- using preconditioning
- using randomized sketching for approximate Hessians

The final comparison is meant to focus on:

- Gradient Descent
- Exact Newton
- Newton-CG
- Preconditioned Newton-CG
- Sketched Newton-CG

The main desired result is that Newton-type methods reach lower training loss in fewer iterations than Gradient Descent. For runtime, interpretation depends on dataset size and feature dimension.

## 2. Reference Paper

The paper referred to during the project is:

- OverSketched Newton: https://arxiv.org/abs/1903.08857
- PDF: https://arxiv.org/pdf/1903.08857

Important clarification:

This project is **not** implementing the full OverSketched Newton system from the paper. The paper includes distributed/serverless ideas, coded computation, and AWS Lambda experiments. Our project only uses the numerical optimization idea of Hessian sketching in a single-machine setting.

Relevant paper idea:

For regularized logistic regression, the Hessian can be written using a square-root weighted data matrix. If

\[
H=\frac1n X^T D X+\lambda I,
\]

then with

\[
A=D^{1/2}X,
\]

a sketched Hessian has the form

\[
\tilde H=\frac1n A^T S^TSA+\lambda I.
\]

This means the sketch should be applied to the square-root Hessian factor, not directly as \(X^TS^TSDX\) unless the sketch operator is defined differently.

The sketched Hessian-vector product used in the code should represent:

\[
\tilde H v=\frac1n X^T\left(\sqrt D\,S^TS(\sqrt D(Xv))\right)+\lambda v.
\]

## 3. Mathematical Setup

We use binary logistic regression with labels:

\[
y_i \in \{-1,1\}.
\]

The regularized objective is:

\[
L(w)=\frac1n\sum_{i=1}^n \log(1+\exp(-y_i x_i^T w))+\frac{\lambda}{2}\|w\|_2^2.
\]

The gradient is:

\[
g(w)=-\frac1n X^T\left(y(1-\sigma(yXw))\right)+\lambda w.
\]

The Hessian is:

\[
H(w)=\frac1n X^T D X+\lambda I,
\]

where

\[
D_{ii}=\sigma(y_i x_i^T w)(1-\sigma(y_i x_i^T w)).
\]

The Hessian-vector product is:

\[
Hv=\frac1n X^T(D(Xv))+\lambda v.
\]

The project deliberately avoids computing \(H^{-1}\). Newton-type methods solve:

\[
Hp=g
\]

and update:

\[
w_{k+1}=w_k-\alpha p.
\]

## 4. Dataset Decision

We searched for a dataset that matches the project goal:

- binary classification
- downloadable
- numeric or easy preprocessing
- suitable for logistic regression
- preferably mildly ill-conditioned
- easy to use for clean plots

Recommended dataset:

- Spambase from UCI
- UCI page: https://archive.ics.uci.edu/dataset/94/spambase
- Direct zip: https://archive.ics.uci.edu/static/public/94/spambase.zip

Spambase details:

- 4601 samples
- 57 continuous features
- last column is class label
- labels are `0` for non-spam and `1` for spam
- no missing values
- file is `spambase.data`, but it is comma-separated and can be loaded like a CSV

The loader converts:

\[
0,1 \rightarrow -1,1.
\]

Feature columns are standardized.

Important limitation:

Spambase is useful for a clean implementation and GD-vs-Newton comparison, but it is small in feature dimension (`d=57`). Because the Hessian is only \(57\times57\), exact Newton is cheap. Therefore, Spambase is **not ideal** for showing runtime advantages of Newton-CG or PCG over exact Newton.

Expected Spambase behavior:

- Exact Newton reaches low loss in very few iterations.
- Newton-CG tracks exact Newton closely.
- Newton-PCG also tracks exact Newton closely.
- Exact Newton may be faster than Newton-CG and PCG because forming/solving a \(57\times57\) Hessian is cheap.
- PCG may not look dramatically better than CG on Spambase.

This is not a failure. It should be explained in the report.

Report interpretation:

“On Spambase, exact Newton is fastest or very competitive because the feature dimension is small. Newton-CG and PCG are more scalable methods and become more useful when Hessian formation or exact linear solves become expensive.”

## 5. Current Code Structure

Current important files:

- `code.py`
- `Z_project.py`
- `project_analysis.ipynb`
- `spambase/spambase.data`
- `spambase/spambase.DOCUMENTATION`
- `spambase/spambase.names`

### `code.py`

This file is the shared logistic regression and Gradient Descent file.

It contains:

- `sigmoid(z)`
- `loss(X,y,w,lam=1e-4)`
- `gradient(X,y,w,lam=1e-4)`
- `hessian_vector_product(X,y,w,v,lam=1e-4)`
- `hessian_diagonal(X,y,w,lam=1e-4)`
- `gradient_descent(X,y,w,lr=0.1,iters=100,lam=1e-4)`
- `load_spambase(path="spambase/spambase.data")`

Important corrections made:

- fixed `import nuumpy as np` typo
- fixed `Hv/hv` return typo
- made all formulas consistent with labels in `{-1,1}`
- added regularization
- added stable sigmoid clipping
- used `np.logaddexp` for stable logistic loss
- standardized Spambase features

### `Z_project.py`

This file contains second-order methods and the runner.

It contains:

- `hessian(X,y,w,lam=1e-4)`
- `line_search(...)`
- `newton_method(...)`
- `conjugate_gradient(...)`
- `newton_cg(...)`
- `diagonal_preconditioner(...)`
- `preconditioned_conjugate_gradient(...)`
- `newton_pcg(...)`
- `gaussian_sketch(n,m)`
- `sketched_hessian_vector_product(...)`
- `sketched_conjugate_gradient(...)`
- `sketched_newton(...)`
- `run_all()`

Important implementation detail:

Since the file is named `code.py`, it collides with Python's built-in `code` module in Jupyter. To avoid this, `Z_project.py` explicitly imports the local `code.py` using `importlib.util.spec_from_file_location`.

### `project_analysis.ipynb`

This notebook is the main analysis notebook.

It covers:

1. Real-data comparison on Spambase
2. Loss vs iteration and loss vs runtime
3. Final summary of loss, runtime, and training accuracy
4. Line search effect
5. Sketch size tradeoff
6. Hessian step quality
7. Synthetic conditioning study
8. Preconditioning study
9. Compact report notes

The notebook was rebuilt after some old cell contents persisted. If plots look unchanged, restart the kernel and run all cells.

Expected notebook signs that the corrected version is loaded:

- A plot titled `Newton family only`
- A final summary split into three panels: final loss, runtime, training accuracy
- Synthetic plot titles like `target=..., hess=...`
- A printed note saying line-search/full Newton overlap on Spambase is expected

## 6. Problems Faced and Fixes

### Problem 1: Dataset format

Spambase is `.data`, not `.csv`.

Conclusion:

The `.data` file is comma-separated and can be loaded with:

```python
np.loadtxt("spambase/spambase.data",delimiter=",")
```

The last column is the label.

### Problem 2: `code.py` import collision in Jupyter

Error:

```text
ImportError: cannot import name 'load_spambase' from 'code'
```

Cause:

Python imported its built-in `code` module instead of local `code.py`.

Fix:

Use `importlib.util.spec_from_file_location` in `Z_project.py` and the notebook to explicitly load the local file.

### Problem 3: Some plots looked like curves were missing

Cause:

Newton, Newton-CG, and Newton-PCG curves are almost identical on Spambase. They overlap visually.

Fix:

Notebook adds:

- different markers
- different line styles
- a `Newton family only` plot
- printed loss arrays

Conclusion:

Overlapping Newton-family curves on Spambase are expected and indicate Newton-CG is accurately approximating exact Newton.

### Problem 4: Exact Newton faster than Newton-CG/PCG

Cause:

Spambase has only 57 features, so the exact Hessian solve is cheap.

Conclusion:

This is expected. Newton-CG and PCG are mainly valuable for larger feature dimension or when Hessian formation/solves are expensive.

### Problem 5: PCG not clearly better than CG on Spambase

Cause:

The problem is not large or ill-conditioned enough for diagonal preconditioning to show a dramatic runtime advantage.

Conclusion:

Use the synthetic conditioning study to show the value of preconditioning. On synthetic ill-conditioned problems, PCG should reduce average inner CG iterations more clearly.

### Problem 6: Synthetic conditioning initially did not show useful separation

Cause:

The earlier synthetic dataset standardized features after building a controlled spectrum. This weakened or hid the intended conditioning effect.

Fix:

The notebook was changed to preserve more of the spectrum and print the measured Hessian condition number.

Expected result:

- GD degrades more as conditioning worsens
- CG iterations tend to grow
- PCG should help reduce average inner CG iterations

## 7. Expected Runtime Order

Theoretical large-scale ideal:

\[
\text{Newton-CG / PCG} < \text{Exact Newton}
\]

because Newton-CG avoids explicit Hessian formation and exact dense solves.

For Spambase:

\[
\text{Exact Newton} \le \text{Newton-CG} \approx \text{PCG}
\]

because `d=57` makes exact Newton cheap.

PCG ideally uses fewer iterations than CG, but it also has preconditioner overhead. On small datasets it may not always be faster.

## 8. Current Numerical Results Observed

One run of `Z_project.py` produced results approximately like:

- Gradient Descent final loss: around `0.2987`
- Exact Newton final loss: around `0.2300`
- Newton-CG final loss: around `0.2300`
- Newton-PCG final loss: around `0.2300`
- Sketched Newton with small sketch can be worse
- Sketched Newton with larger sketch improves and gets closer to Newton

Interpretation:

- GD is slower and reaches higher loss in the same number of iterations.
- Newton reaches the low-loss region quickly.
- Newton-CG approximates exact Newton well.
- PCG is mathematically correct but not dramatically better on Spambase.
- Sketching shows a tradeoff: too small a sketch can hurt, larger sketches better approximate Newton.

## 9. Important Report Claims

Safe claims:

- Newton reaches low loss in fewer iterations than GD.
- Newton-CG avoids explicit Hessian inversion.
- Hessian-vector products reduce the need to form/invert the Hessian.
- PCG can reduce inner CG iterations, especially under worse conditioning.
- Sketching trades accuracy for computational savings.
- On small feature dimension, exact Newton can be faster than Newton-CG.
- Conditioning is a key reason for differences in first-order and second-order convergence.

Claims to avoid:

- Do not claim full OverSketched Newton was implemented.
- Do not claim distributed/serverless speedups.
- Do not claim Newton-CG or PCG is always faster than exact Newton.
- Do not claim sketching always improves convergence.

## 10. Suggested Final Project Story

Recommended narrative:

1. Start with logistic regression and explain the convex optimization objective.
2. Show Gradient Descent is cheap per step but slow.
3. Show exact Newton has faster convergence but normally expensive Hessian operations.
4. Introduce Newton-CG as a matrix-free method using Hessian-vector products.
5. Introduce diagonal preconditioning as a numerical linear algebra improvement.
6. Introduce sketched Newton as an approximate Hessian method inspired by the paper.
7. Use Spambase for real-data validation.
8. Explain why exact Newton is still fast on Spambase due to small `d=57`.
9. Use synthetic conditioning experiments to show where CG/PCG matter more.
10. Discuss tradeoffs between iteration count, runtime, conditioning, and approximation quality.

## 11. Next Best Steps

The next model or teammate should:

1. Open `project_analysis.ipynb`.
2. Restart the kernel.
3. Run all cells from top.
4. Check that the new notebook version is loaded by confirming:
   - `Newton family only` plot appears
   - final summary has 3 separate panels
   - synthetic plots have `target=..., hess=...` in titles
5. Save final plots for the report.
6. Add explanation text in the report using the conclusions above.
7. If more time is available, add a larger synthetic dataset or higher-dimensional synthetic problem to better demonstrate Newton-CG/PCG scalability.

## 12. Coding Style Constraint

The professor warned against obvious copy-pasted AI code. Therefore the code was intentionally kept:

- simple
- compact
- modular
- readable
- with short function names
- with small comments only
- without classes or over-engineered abstractions

Preferred naming style:

- `loss`
- `gradient`
- `hessian`
- `hessian_vector_product`
- `gradient_descent`
- `newton_method`
- `newton_cg`
- `newton_pcg`
- `sketched_newton`
- `gaussian_sketch`
- `diagonal_preconditioner`

Avoid overly cryptic names like `hvg`, `pcg1`, `shvp`.

## 13. Final Takeaway

The project is currently in a good state:

- algorithms are implemented
- Spambase is integrated
- notebook analysis exists
- major issues have been identified and explained

The key conceptual point is:

Spambase is excellent for showing Newton beating GD in iteration count, but it is too small in feature dimension to clearly demonstrate the runtime benefit of Newton-CG and PCG over exact Newton. That benefit should be argued theoretically and demonstrated with the synthetic conditioning/scalability study.
