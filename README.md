# AISTATS Submission — Code & Reproducibility

This repository contains code to reproduce the experiments for the AISTATS paper submission:

> **Title:** Online bootstrap inference for the level of nonstationary time series
> **Venue:** AISTATS 2026

---

# README

This repository accompanies the paper submission.

## Requirements

- Python **3.10–3.12**
- OS: Linux/macOS/Windows
- Core deps resolved via `pyproject.toml`

## Installation (local, editable)

> Use this when unpacking the submission archive locally.

```bash
# from the unpacked project root
python -m venv .venv
source .venv/bin/activate    # on Windows: .venv\Scripts\activate
pip install -e .
