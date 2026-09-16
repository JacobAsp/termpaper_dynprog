# Termpaper Dynamic Programming

# Job Search Model

This folder contains the code used to solve, simulate, and estimate the structural job-search model developed for the term paper.

The project studies unemployment search behavior in a dynamic programming framework. Individuals choose consumption and search effort while unemployed, taking into account unemployment benefits, the value of employment, savings/assets, and search costs.

The model is estimated by matching simulated job-finding hazard rates to empirical hazard rates before and after a reform using the **Simulated Method of Moments (SMM)**.

The three main files are:

* `Model_components.py` — contains the economic model and the functions used to solve, simulate, and estimate it.
* `Estimation.ipynb` — estimates the structural parameters of the different model specifications.
* `Model prediction.ipynb` —  Produces model predictions and figures.

---

## Project structure

```text
Job Search/
│
├── Model_components.py
├── Model prediction.ipynb
├── Estimation.ipynb
│
├── base_moments_Hungary.xlsx
│
├── estimation_figures/
├── estimation_tables/
│
└── ...
```

The remaining notebooks and copies in the folder mainly contain earlier versions, auxiliary analyses, or code used during development.

---

## 1. `Model_components.py`

`Model_components.py` contains the main model code.

This file defines the primitives of the structural model and the functions needed to solve it.

### Preferences

Baseline consumption utility is logarithmic. The reference-dependent version of the model additionally allows utility to depend on consumption relative to a reference level,

$$
u(c_t,r_t)
=
\log(c_t)
+
\begin{cases}
\eta(\log c_t-\log r_t), & c_t>r_t,\\
\eta\lambda(\log c_t-\log r_t), & c_t\leq r_t.
\end{cases}
$$

where:

* \(c_t\) is consumption,
* \(r_t\) is the reference level,
* \(\eta\) determines the importance of gain-loss utility,
* \(\lambda\) determines the degree of loss aversion.

Setting the reference-dependent parameters to zero gives the standard model.

### Search costs

Search effort \(s_t\) is costly according to

$$
C(s_t)
=
\frac{k s_t^{1+\gamma}}{1+\gamma},
$$

where \(k\) determines the level of search costs and \(\gamma\) their curvature.

Optimal search effort is determined by the difference between the continuation value of employment and unemployment.

### Dynamic programming problem

The model solves for:

* the value of employment,
* the value of unemployment,
* optimal consumption,
* optimal asset holdings,
* optimal search effort,
* unemployment survival probabilities,
* job-finding hazard rates.

The solution combines backward induction with interpolation over an asset grid.

A forward simulation subsequently follows the optimal policy functions through time in order to determine the evolution of assets, consumption, and search effort for individuals entering unemployment with different initial asset levels.

### Assets

The model can be run either:

1. **Hand-to-mouth**, where individuals enter unemployment without savings and cannot use assets to smooth consumption; or
2. **With assets**, where individuals can save and dissave during unemployment.

For the asset model, initial asset holdings are represented by a discrete grid and weighted using an assumed asset distribution.

### Heterogeneous search-cost types

The model also allows for unobserved heterogeneity in search costs.

The current implementation supports:

* one search-cost type,
* two search-cost types,
* three search-cost types.

For example, in the two-type specification,

$$
k \in \{k_1,k_2\},
$$

with population shares

$$
(q_1,1-q_1).
$$

The aggregate unemployment hazard is constructed from the survival probabilities of the different types.

### Pre- and post-reform environments

The model is solved separately under the institutional settings prevailing before and after the reform.

This allows the same structural parameters to generate separate predicted hazard-rate paths for the pre- and post-reform benefit systems.

---

## 2. `Estimation.ipynb`

`Estimation.ipynb` contains the main structural estimation.

The notebook:

1. imports the model functions from `Model_components.py`,
2. defines the institutional parameters of the pre- and post-reform environments,
3. loads the empirical moments,
4. defines parameter starting values and parameter bounds,
5. solves the model for candidate parameter vectors,
6. constructs simulated hazard-rate moments,
7. compares simulated and empirical moments,
8. minimizes the SMM objective function,
9. stores the estimated parameters and goodness-of-fit measures,
10. produces estimation tables and diagnostic figures.

### Simulated Method of Moments

Let

$$
m^{data}
$$

denote the empirical hazard-rate moments and

$$
m(\theta)
$$

the corresponding moments predicted by the structural model for parameter vector \(\theta\).

Parameters are estimated by minimizing

$$
\hat{\theta}
=
\arg\min_{\theta}
\left[
m^{data}-m(\theta)
\right]'
W
\left[
m^{data}-m(\theta)
\right],
$$

where \(W\) is the weighting matrix.

The empirical moments contain hazard rates from both the pre-reform and post-reform periods.

The code estimates several specifications, including:

* standard model with one type,
* standard model with two types,
* standard model with three types,
* reference-dependent model.

Different numerical optimization algorithms can be used to examine the robustness of the estimates to the optimization procedure.

The current estimation exercises primarily compare **L-BFGS-B** and **LS-TRF**.

---

## 3. `Model prediction.ipynb`

`Model prediction.ipynb` is used to inspect the economic behavior implied by the model.

Rather than estimating parameters, this notebook takes a specified parameter vector and examines the model predictions.

It is used to visualize objects such as:

* predicted job-finding hazard rates,
* empirical versus predicted hazard rates,
* search effort over the unemployment spell,
* consumption paths,
* asset paths,
* employment and unemployment value functions,
* differences between initial asset levels,
* differences between the pre- and post-reform environments,
* differences across model specifications.

This notebook is therefore useful for understanding the mechanisms generating the estimated hazard-rate profiles.

---

## Empirical moments

The empirical moments used in estimation are stored in

```text
base_moments_Hungary.xlsx
```

The file contains job-finding hazard-rate moments for the pre- and post-reform periods together with their estimated uncertainty.

`Model_components.py` loads these moments and constructs the SMM target vector and weighting matrix.

Because the path to the data file is currently specified relative to the working directory,

```python
./base_moments_Hungary.xlsx
```

the notebooks should normally be run with `Job Search/` as the working directory.

---

## Parameters

The main structural parameters include:

| Parameter   | Interpretation                          |
| ----------- | --------------------------------------- |
| \(\delta\)  | Discount factor                         |
| \(\gamma\)  | Curvature of search costs               |
| \(k\)       | Scale of search costs                   |
| \(\eta\)    | Strength of reference-dependent utility |
| \(\lambda\) | Loss-aversion parameter                 |
| \(N\)       | Length of the reference window          |

For models with heterogeneous search costs, additional parameters include

$$
k_1,\;k_2,\;k_3
$$

and population shares

$$
q_1,\;q_2,\;q_3.
$$

The exact set of estimated parameters depends on the model specification.

---

## Requirements

The project is written in Python and uses Jupyter notebooks.

The main dependencies are:

```text
numpy
pandas
scipy
matplotlib
optimagic
openpyxl
jupyter
```

They can be installed with, for example,

```bash
pip install numpy pandas scipy matplotlib optimagic openpyxl jupyter
```

---

## Running the project

Clone the repository:

```bash
git clone https://github.com/JacobAsp/termpaper_dynprog.git
```

Move to the job-search folder:

```bash
cd termpaper_dynprog/"Job Search"
```

Start Jupyter:

```bash
jupyter notebook
```

A natural workflow is then:

```text
Model_components.py
        ↓
Model prediction.ipynb
        ↓
Estimation.ipynb
    
```

### Estimating the model

Open

```text
Estimation.ipynb
```

and run the notebook from top to bottom.

The notebook imports the model, loads the empirical moments, defines the model specifications, and runs the SMM estimation.

### Inspecting model predictions

To inspect model and how it works, open

```text
Model prediction.ipynb
```

and insert the desired parameter estimates.

Running the notebook generates the implied policy functions, hazard rates, and other model predictions.

---

## Output

Estimation output is stored primarily in:

```text
estimation_figures/
estimation_tables/
```

These folders contain figures and tables used to compare parameter estimates, goodness of fit, and predicted versus empirical hazard rates across specifications and optimization algorithms.

---

## Model specifications

The code is designed so that several versions of the job-search model can be compared within the same framework.

In particular, the project considers differences along three dimensions:

### Standard vs. reference-dependent preferences

The standard model assumes utility depends only on consumption.

The reference-dependent model additionally allows individuals to evaluate consumption relative to a reference level determined by previous income.

### Homogeneous vs. heterogeneous search costs

The baseline model has a single search-cost parameter.

The multi-type specifications allow different groups of workers to have different search-cost parameters, generating heterogeneity in job-search behavior.

### Hand-to-mouth vs. assets

The hand-to-mouth version abstracts from savings.

The asset version allows individuals to smooth consumption during unemployment by adjusting their asset holdings.

---

## Interpretation

The purpose of the model is not only to reproduce the average level of job-finding rates but also to investigate whether the structural model can reproduce the **shape and timing of search responses to changes in unemployment benefits**.

The comparison of different specifications therefore provides information about which mechanisms are important for explaining observed search behavior, including:

* dynamic incentives from future benefit changes,
* consumption smoothing through assets,
* heterogeneity in search costs,
* reference dependence and loss aversion.

---

## Authors

Code developed as part of a term paper in dynamic programming 
Jacob Asp Nissen
Albert Luis Føns Olsen

Repository:

`JacobAsp/termpaper_dynprog`
