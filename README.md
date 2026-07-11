<div align="center">

# AttritionIQ — Employee Attrition Risk Assessment

### Explainable Machine Learning Decision-Support System for HR

[![Live App](https://img.shields.io/badge/Live%20App-Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://explainable-hr-attrition-risk-system.streamlit.app/)
![Model](https://img.shields.io/badge/Model-Random%20Forest-green?style=flat-square)
![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square)
![Status](https://img.shields.io/badge/Status-Research%20Edition-orange?style=flat-square)

</div>

---

## The problem

Every time a company loses an employee, it costs **50–200% of their annual salary** to replace them. HR teams often find out too late — after resignation, not before.

Most attrition tools give a binary answer: *will leave* or *won't leave*.
That's not useful. HR needs to know **who is at risk, how much risk, why, and what to do about it.**

---

## What this system does differently

This is not a single classifier — it's a full decision-support suite covering the lifecycle of an attrition-risk workflow:

- **Individual risk scoring** — probability (0–100%), not just yes/no, with a confidence level
- **What-if salary simulation** — test a salary change and see the risk shift live
- **Risk trajectory analysis** — project how an employee's risk is trending over time, with an early-warning alert level
- **Causal intervention engine** — ranked, specific HR actions (not generic advice) with predicted risk reduction, cost, and timeline
- **Employee risk segmentation** — cluster an uploaded workforce into risk profiles for targeted strategy, with per-cluster recommendations
- **Fairness & bias audit** — check false positive/negative rate parity and precision parity across any demographic column, with automated bias flags (80% rule)
- **Explainability analysis** — global feature importance and counterfactual explanations ("what change would lower this employee's risk?")
- **Batch CSV support** — score, segment, and audit entire departments at once, with robust handling of messy real-world data (currency symbols, inconsistent Yes/No casing, missing values)

---

## Example output

```
Employee: John D.
Attrition Risk: 78%  ⚠️ High

Top risk factors:
↑ Overtime hours        (+23% risk contribution)
↑ Years since promotion (+18% risk contribution)
↓ Job satisfaction      (+15% risk contribution)

What-If: +10% salary increase → Risk drops to 61%
```

---

## Features

| Page | What it enables |
|---|---|
| 👤 Individual Prediction | Probability risk score + confidence + what-if salary simulation |
| 📈 Risk Trajectory | Simulated quarter-over-quarter risk trend with early-warning alerts |
| 💡 Interventions | Ranked HR actions with predicted risk reduction, cost, and timeline |
| 👥 Employee Segmentation | Cluster an uploaded workforce by risk profile, with per-cluster action plans |
| ⚖️ Fairness & Bias Audit | FPR/FNR/precision parity across any demographic column, with bias detection |
| 🔍 Explainability | Global feature importance + counterfactual "what would reduce this risk" analysis |
| 📊 Model Monitoring | Framework for calibration, prediction-drift, and data-quality checks *(reference implementation in `src/`, not yet live-wired to production data)* |

---

## Model

- **Algorithm:** Random Forest Classifier
- **Core features:** `Age`, `MonthlyIncome`, `YearsAtCompany`, `OverTime`
- **Output:** Attrition probability (0.0 – 1.0)
- **Explainability:** Feature importance + counterfactual explanations (SHAP integration available in `src/explainability.py`, in progress for full UI wiring)
- **Deployment:** Lightweight inference pipeline — training and deployment separated for fast inference

> **Note on accuracy:** This dataset (IBM HR Attrition–style data) is imbalanced (~16% attrition). Accuracy alone is not a meaningful metric here — evaluate with precision/recall/AUC on held-out data before relying on this for real HR decisions.

---

## Stack

`Python` `scikit-learn` `Streamlit` `Pandas` `NumPy` `Joblib` `Matplotlib`

---

## Run locally

```bash
git clone https://github.com/AkashMs24/Employee-Attrition-Risk-Assessment-Using-Explainable-Machine-Learning.git
cd Employee-Attrition-Risk-Assessment-Using-Explainable-Machine-Learning
pip install -r requirements.txt
streamlit run app.py
```

### Batch upload format

For Employee Segmentation and Fairness Audit, upload a CSV containing at minimum:

| Column | Type | Notes |
|---|---|---|
| `Age` | numeric | |
| `MonthlyIncome` | numeric | commas/currency symbols are auto-cleaned |
| `YearsAtCompany` | numeric | |
| `OverTime` | Yes/No | also accepts True/False, 1/0 |
| `Attrition` | Yes/No | optional — required only for Fairness Audit metrics |

Rows with unrecoverable missing/invalid values in these columns are automatically excluded, with a warning showing how many rows were dropped.

---

## Design decisions worth noting

- **Probability over binary** — a 78% risk score is actionable; "will leave" is not
- **Simulation over prediction** — what-if analysis turns insight into intervention
- **Fairness as a first-class feature** — bias auditing isn't an afterthought, it's a dedicated page
- **Defensive data handling** — batch uploads are validated and cleaned rather than trusted blindly, so malformed real-world HR exports don't crash the app
- **Separated pipelines** — model trained offline, deployed artifact is lightweight for fast inference

---

## Research

This project was developed as part of ongoing academic work and received acceptance at **ICDIA 2026**. Features have continued to expand post-submission — see commit history for the version corresponding to the original submission.

---

## Related projects

- [Decision Intelligence System](https://github.com/AkashMs24/Decisioniq-ai-business-intelligence) — ML + LLM business intelligence platform
- [Fraud Detection System](https://github.com/AkashMs24/Cost-Sensitive-Real-Time-Fraud-Detection-Decision-System) — XGBoost + SHAP + FastAPI
- [FarmVoice AI](https://github.com/AkashMs24/FarmVoice-AI) — NLP + SHAP crop advisory

---

<div align="center">

Built by **Akash M S** · Presidency University, Bengaluru
[LinkedIn](https://www.linkedin.com/in/akash-m-s-414a21297) · [GitHub](https://github.com/AkashMs24) · ms29akash@gmail.com

</div>
