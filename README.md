# mgis355-p2-sohail: Startup exit screener

**One line:** a logistic regression that tells a VC screener whether to take a first meeting with a newly funded startup, by estimating its chance of being acquired or going public within 5 years of its first round. Real data (Crunchbase 2013 Snapshot © 2013, CC BY).

- **App:** https://mgis355-p2-sohail.streamlit.app
- **Video:** [PASTE VIDEO LINK]
- **Repository:** https://github.com/aazansohail/mgis355-p2-sohail

## Results on the held-out 2008 cohort (1,648 startups, 189 exits)

| Rule | Meetings per 100 | Exits caught | Cost per 100 startups |
|---|---|---|---|
| Model, threshold 0.13 | 51 | 142 (75%) | $21,371 |
| Baseline: Series A+ or at least $500K | 74 | 156 (83%) | $25,394 |
| Meet everyone | 100 | 189 (100%) | $26,559 |

Costs: $300 per wasted meeting, $3,000 per missed exit. Test AUC 0.680.

## Files

| File | What it is |
|---|---|
| `app.py` | Streamlit app. Loads `model.joblib` once with `@st.cache_resource`; never retrains. |
| `model.ipynb` | Load, check, split, fit, evaluate. Writes `model.joblib` and `app_data/`. |
| `model.joblib` | Trained scikit-learn pipeline (preprocessing + logistic regression). |
| `app_data/` | Scored test set and metadata the app displays (written by the notebook). |
| `data/` | Four trimmed Crunchbase CSVs plus `make_extracts.py`, which builds them from the Kaggle download. |
| `model_card.md`, `data.md`, `verification.md`, `disclosure.md` | Write-ups. |

## Run it

```bash
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute model.ipynb --inplace   # optional: rebuilds model.joblib and app_data/ (needs jupyter)
streamlit run app.py
```

Tested with Python 3.13 and the exact versions in `requirements.txt`.

Data attribution: Crunchbase 2013 Snapshot © 2013, licensed CC BY, via kaggle.com/datasets/justinas/startup-investments.
