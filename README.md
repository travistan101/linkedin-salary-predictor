# LinkedIn Salary Prediction

Predicting advertised salaries from ~124k LinkedIn job postings, built as a LangGraph agent
pipeline in PySpark. A tuned XGBoost model reaches **test R² 0.71** (RMSE 0.2974, MAE 0.2141).

**NUS BT4221 — Advanced Analytics with Big Data Technologies · Group 13**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/travistan101/linkedin-salary-prediction/blob/main/notebooks/02_Model_Training.ipynb)

---

## Problem

Salary data on job postings is sparse and inconsistently expressed — min/max ranges, differing
pay periods, and a large share of nulls. The task: predict a normalised salary (modelled as
`target_log_salary`) from posting, company, and benefits attributes.

## Data

Four joined source tables:

| Table | Rows | Columns |
|---|---|---|
| `postings` | 123,849 | 31 |
| `benefits` | 67,943 | 3 |
| `companies` | 24,473 | 10 |
| `employee_counts` | 35,787 | 4 |

Joins are performed defensively to avoid row duplication; null rates are profiled per column
before feature selection.

## Results

Three models trained on identical splits and evaluated with Spark MLlib's `RegressionEvaluator`.

**Baselines** (validation set):

| Model | RMSE | MAE | R² | Overfit gap (RMSE) |
|---|---|---|---|---|
| **XGBoost** | **0.3993** | **0.2373** | **0.5444** | 0.1639 |
| Linear Regression | 0.4230 | 0.2696 | 0.4887 | 0.1220 |
| Random Forest | 0.4799 | 0.3333 | 0.3418 | 0.0518 |

XGBoost was selected for tuning on lowest validation RMSE.

**Final tuned XGBoost:**

| Split | RMSE | MAE | R² |
|---|---|---|---|
| Validation | 0.3912 | 0.2282 | 0.5626 |
| Test | 0.2974 | 0.2141 | 0.7146 |

Agent-driven tuning improved validation RMSE from 0.3993 to 0.3912 (R² 0.5444 → 0.5626). The
higher test R² reflects an easier test split, not an additional modelling gain — validation is
the fair basis for comparing tuning runs.

## Architecture

The pipeline runs as a LangGraph `StateGraph` over a shared `MLAgentState` (a `TypedDict`
carrying datasets, fitted models, metrics, and tuning history), so each stage is inspectable
and re-runnable:

```
prepare splits ──> train baselines ──> rank & select ──> agent tuning loop ──> final selection
                                            │                    │
                                   lowest val RMSE      revert on regression;
                                                        stop when Δ < 0.002 over
                                                        3 runs (min 5 runs)
```

- **Baseline training** — fits Linear Regression, Random Forest, and Spark XGBoost, recording
  train/validation/test metrics plus train-validation gaps
- **Ranking node** — orders models by validation RMSE, then MAE, then R²; flags overfitting when
  the RMSE gap exceeds 0.12 and sets tuning priority accordingly
- **Tuning agent** — an LLM-driven loop (OpenAI `gpt-4o-mini`) that proposes hyperparameter
  changes one lever at a time, chasing validation RMSE and reverting any change that worsens it
- **Final selection** — re-ranks all baseline and tuned models and prints the comparison table

Tool arguments exchanged with the LLM are typed with Pydantic models (e.g. `EvaluateFeatureArgs`),
so agent calls are schema-validated rather than free-form text.

## Tech stack

| Layer | Tools |
|---|---|
| Processing | Apache Spark (PySpark 4.0), Spark MLlib |
| Modelling | Spark XGBoost, Random Forest, Linear Regression |
| Orchestration | LangGraph, LangChain, OpenAI `gpt-4o-mini` |
| Schemas | Pydantic |
| Environment | Google Colab |

## Repository structure

```
notebooks/
  01_Feature_Engineering.ipynb   loading, cleaning, joins, agentic feature selection
  02_Model_Training.ipynb        splits, baselines, agent tuning, final selection
README.md
```

## Running it

Use the Colab badge above. The notebooks read from Google Drive and expect an `OPENAI_API_KEY`
in the environment for the agent nodes. The dataset is not committed — see the Kaggle LinkedIn
Job Postings dataset.

## Limitations and next steps

- The test/validation R² gap (0.71 vs 0.56) suggests split variance worth checking with
  cross-validation rather than a single holdout
- Random Forest underperformed badly (R² 0.34) and was never tuned — likely under-configured
  rather than genuinely unsuited
- Job description free text is unused; sentence embeddings are the obvious next feature source
- Predicting a salary *range* via quantile regression would match how postings actually
  advertise pay

---

Group project for BT4221 at the National University of Singapore.
