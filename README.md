# UAI 2026: Online Bootstrap Inference for the Trend of Nonstationary Time Series

This repository accompanies the paper and contains the code to reproduce all experiments.

## Requirements

- Python **3.10–3.12**
- Core deps resolved via `pyproject.toml`

## Dependencies

Core dependencies required to run the experiments:

- `joblib>=1.5.2`
- `numpy>=1.21.2`
- `pandas>=1.4.3`
- `matplotlib>=3.10.5`
- `scipy>=1.16.1`
- `tqdm>=4.64.1`

## Installation (local, editable)

> Use this when unpacking the submission archive locally.

```bash
# from the unpacked project root
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Notebooks and Reproducibility

The experiment setup can be found in `notebooks/coveragestudy.ipynb`. Results with random seeds and parameters are completely reproducible. `notebooks/plotting.ipynb` was used to create the coverage plots. The time series plots are generated in `notebooks/timeseries.ipynb`.

**Note:** running the notebooks requires a Jupyter frontend and an IPython kernel. Apart from the core runtime dependencies listed above, the only extra packages needed to run the notebooks are:

- `ipykernel` (required to run notebooks from this virtual environment)
- a Jupyter frontend (either `jupyterlab` or `notebook`)

### Quick setup (after activating the project venv)
```bash
# install the package (editable) and the notebook extras
pip install -e .
pip install ipykernel jupyterlab      # or: pip install ipykernel notebook
```

## Runtime

All experiments can be run on a local machine. For the results in this repository we used a MacBook Pro with an **M1 Pro** chip (8 cores). The experiment scripts support parallel execution (`parallel=True`, `n_jobs=-1` by default) and will use available cores. `n_jobs` may be set to a smaller number if CPU use needs to be limited.
