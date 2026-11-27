# Model card: startup exit screener

| Field | |
|---|---|
| **Decision and user** | A VC associate or accelerator screener decides, per newly funded startup, whether to take a first meeting. Dozens a week (about 35 a week in the 2008 test data). |
| **Unit of prediction** | One row per company, as of the date of its first recorded funding round. |
| **Target** | `exited = 1` if acquired or IPO within 5 years of the first round, else 0. Excludes: first rounds after 2008-12-12 (no full window) or before 2000; exits dated before the first round; 35 exits with no date; non-companies. |
| **Data** | Crunchbase 2013 Snapshot © 2013 (CC BY), via Kaggle. Real. 8,214 companies, first rounds 2000-2008, outcomes through 2013-12-12. 14.4% exited. |
| **Features** | All known at prediction time: first-round amount, number of investors, company age, round stage, industry, country, US state. Zeros and impossible dates become missing, with a missing flag. |
| **Excluded on purpose** | `status` (the outcome; adding it gives AUC 0.997). `funding_rounds`, `funding_total_usd` (include later rounds). First-round year (every new case is outside the training years). IDs, names, URLs. |
| **Split** | Time split. Train: 6,566 (2000 to Feb 2008, 15.2% exit). Test: 1,648 (Feb to Dec 2008, 11.5%). Settings and threshold picked on the latest 1,314 training rows. A random split gives a similar AUC (0.670) but hides that the model over-predicts on the next cohort (13.6% predicted vs 11.5% actual). |
| **Baseline** | Meet if Series A or later, or first round ≥ $500K (cutoff set on train). Test: 74 meetings per 100, catches 83% of exits, $25,394 per 100 startups. |
| **Model** | Logistic regression (C = 0.01). Test AUC **0.680**. At 0.13: 51 meetings per 100, catches 75% of exits, **$21,371 per 100, $4,023 (16%) below baseline**. Gradient boosting (AUC 0.677, $21,845) did not beat it on test, so it was not shipped. |
| **Threshold and cost** | Wasted meeting $300 (about 3 hours of associate time, assumed). Missed exit 10x = $3,000 (checkpoint assumption). Theory says 1/11 = 0.091; validation picked **0.13** because the model over-predicts on later cohorts. Default 0.5 would flag nobody (top score 0.35). |
| **Where it is weakest** | No investor count recorded (19% of test): AUC 0.53, catches 0 of 20 exits. Non-US: catches 42% of exits vs 85% in the US. |
| **Do not use it for** | Deciding whether or how much to invest (it predicts whether an exit happens, not its size). Startups after 2013 without retraining. Companies past their first round. |
| **What would change the recommendation** | It only beats every alternative when a missed exit costs 3 to 15 wasted meetings; at 20+, meet everyone. With a fixed meeting budget, use the ranking (top N), not the threshold. |
