# Loan Approval Prediction (SVC) — Deployed with Docker + FastAPI + Render

## ⚠️ Project Purpose: Learning, Not Production
This is a **learning project**, not a production-ready loan approval system.
The main goal was to practice the full ML lifecycle end to end — training a
classifier, handling class imbalance, and deploying it as a live API using
**Docker**, **FastAPI**, and **Render**. The code comments (in Arabic) reflect
that learning process, written by a beginner working through each step and
documenting their understanding along the way.

## Overview
A binary classifier (SVC) that predicts whether a bank loan application is
likely to be approved or rejected, based on the applicant's demographic and
financial information. The trained pipeline is deployed as a REST API.

## Dataset & Its Limitations
The dataset (`Loan_Approval.csv`) is small and imbalanced:
- **480 usable rows** after dropping missing values
- **~69% approved / ~31% rejected**
- The test split ends up with only **~28 rejected samples**, which is a very
  small sample to evaluate a model on

This matters a lot for interpreting the results below.

## Handling Class Imbalance
Two versions of the model were trained and compared:
- `svc_loan_approval.ipynb` — trained without any class balancing
- `svc_by_balancing.ipynb` — trained with manual class-weight balancing

**The unbalanced model was chosen as the final one**, since balancing did not
produce a meaningfully better result on this dataset. Both notebooks are kept
in this repo intentionally, to document that comparison — not because the
unbalanced one is definitively "correct", but because it performed
comparably while being simpler.

## Results (on the unbalanced model)
- **Accuracy:** 82%
- **Precision (Rejected):** 100%
- **Recall (Approved):** 100%
- Best hyperparameters (via GridSearchCV): linear kernel, C = 0.1

These numbers look strong on paper, but **they are misleading if read at
face value** — see the note below.

## ⚠️ Important Note: Why the High Scores Don't Mean "The Model Is Always Right"
If you try the deployed app and change small financial details (income, loan
amount) without changing `Credit_History`, the prediction usually **won't
change**. This is not a bug. It's because:

1. **`Credit_History` dominates the data almost entirely.** Out of applicants
   with good credit history (1.0), ~79% were approved. Out of applicants with
   bad credit history (0.0), only ~10% were approved. The model picked up on
   this very strong pattern and leans on it heavily — much more than income
   or loan amount.
2. **The rejected class has very few samples** (~28 in the test set), so a
   "100% precision" score is easy to reach with so little data to be wrong
   on, and does not generalize the way it would with a larger, more balanced
   dataset.

In short: the model is technically "correct" on the metrics it was measured
with, but it is **not robust or nuanced enough for real-world use**, and its
predictions should be read as an educational demonstration, not financial
advice for any real applicant.

## Deployment Stack
- **Model:** scikit-learn `Pipeline` (imputing + scaling + one-hot encoding +
  SVC), saved with `joblib`
- **API:** FastAPI (`main.py`), exposing a `/predict` endpoint
- **Containerization:** Docker
- **Hosting:** Render (free tier, Docker-based Web Service)

## Files
- `svc_loan_approval.ipynb` — main notebook (final model, no class balancing)
- `svc_by_balancing.ipynb` — experiment notebook testing class-weight balancing
- `Loan_Approval.csv` — dataset
- `main.py` — FastAPI app serving predictions
- `Dockerfile` — container definition
- `models/Loan_Approval.pkl` — trained pipeline (preprocessing + SVC)
- `requirements.txt` — required libraries

## How to Run Locally
```bash
pip install -r requirements.txt
uvicorn main:app --reload
```
Then open `http://127.0.0.1:8000/docs` to test the `/predict` endpoint.

## How to Run with Docker
```bash
docker build -t loan-api .
docker run -p 8000:8000 loan-api
```

## Live Demo
[https://loan-approval-api-9lzh.onrender.com]

## What I Learned
- Building a full scikit-learn `Pipeline` (imputation, scaling, encoding,
  model) as a single deployable object
- Comparing balanced vs. unbalanced training on an imbalanced dataset
- Reading classification metrics critically instead of trusting a single
  high number (precision/recall/F1 in context of a small test set)
- Packaging a trained model with `joblib` and serving it through a FastAPI
  endpoint
- Containerizing an ML API with Docker and deploying it on Render

## Tools
Python, Pandas, Scikit-learn, FastAPI, Docker, Render

## Author
Laith Hussein
https://www.linkedin.com/in/laith-hussein-abaub/
