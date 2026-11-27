# Verification log

## Traced number 1: the test set has 189 exits among 1,648 startups (11.5%)

**Where the app shows it:** tab 1 ("the 11.5% rate among the test-set startups") and tab 2 ("142 of 189", "Exits caught" column).

**Independent route** (`model.ipynb`, "Traced number 1"): re-read the four CSVs from disk and rebuild the test set without the notebook's pipeline or the app's code. First round per company via `sort_values` + `drop_duplicates` (the pipeline uses `groupby().min()`); exit date via a stacked acquisitions + IPOs table with `drop_duplicates` (the pipeline uses `groupby().min()` and a column-wise `min`); then the same exclusions written as fresh filters.

| | Startups | Exits | Base rate |
|---|---|---|---|
| Independent route | 1,648 | 189 | 11.5% |
| Pipeline / app | 1,648 | 189 | 11.5% |

**Matched.**

## Traced number 2: the example startup scores 21.8%

**Where the app shows it:** tab 1 with the default inputs (Series A, $5,000,000, 3 investors, 2 years old, software, United States, California). The app calls `model.predict_proba`.

**Independent route** (`model.ipynb`, "Traced number 2"): no `predict_proba`. Take the fitted parts out of the pipeline and do the arithmetic by hand: log1p the three numbers, fill missing with the stored medians, add the missing flags, standardize with the stored means and scales, build the one-hot vector (including the "infrequent" column), then intercept + coefficients · features, then the logistic function.

| | Logit | Probability |
|---|---|---|
| By hand | -1.2753 | 0.2184 |
| `predict_proba` / app | | 0.2184 (shown as 21.8%) |

**Matched** to four decimals.

**Bonus check, by calculator:** the app says the model "saves $4,023 per 100 startups" vs the baseline. Model: 704 wasted meetings + 47 missed exits x 10 = 1,174 cost units. Baseline: 1,221 meetings, 156 caught, so 1,065 wasted + 33 missed x 10 = 1,395 units. (1,395 - 1,174) x $300 / 1,648 x 100 = **$4,023**. Matched.

## Leakage check: `n_investors`

**Why this feature:** it is the strongest input. Its missing flag has the largest coefficient (-0.275), and startups with no investor count exit at 6% vs 16-17%. That is what a leak looks like. The worry: Crunchbase records were often typed in after the fact (45.5% of exits had their first-round record created after the exit), so a filled-in investor count could reflect that the company later became newsworthy, which is the outcome.

**Test 1, refit without it** (same settings, same split):

| Test rows | n | Exits | AUC with | AUC without | Drop |
|---|---|---|---|---|---|
| All | 1,648 | 189 | 0.680 | 0.647 | 0.033 |
| Round recorded within 1 year (before an exit could be known) | 1,045 | 137 | 0.690 | 0.647 | 0.043 |
| Round recorded 1+ years later (back-filled) | 603 | 52 | 0.641 | 0.623 | 0.018 |

**Test 2, does the pattern exist where leakage is impossible?** Exit rate by investor-count missingness (all 8,214 rows):

| Record created | Investors recorded | Investors missing |
|---|---|---|
| Within 1 year of the round | 17.3% | 6.0% |
| 1+ years later | 15.7% | 6.3% |

**Verdict: real signal, not leakage.** A drop of 0.033 by itself proves nothing either way. What settles it: if the count were filled in because of the exit, its signal would be concentrated in back-filled records. It is the opposite. The signal is *stronger* in records written within a year of the round, before anyone could know the outcome (drop 0.043 vs 0.018), and the 6% vs 17% gap is the same in both groups. The likely real story: a round with no listed investors is usually a small or obscure one. I kept the feature and named its blind spot in the model card: when the count is missing, the model never recommends a meeting.

## What the excluded columns would have done

Each column added to the shipped model and refit:

| Added column | Test AUC | Shipped model: 0.680 |
|---|---|---|
| `status` (acquired / ipo / closed as of 2013) | **0.997** | the outcome itself: a textbook leak |
| `funding_rounds` (rounds raised by 2013) | 0.676 | leaky in principle, no gain |
| `funding_total_usd` (total raised by 2013) | 0.681 | leaky in principle, no gain |

The last two show the other half of the rule: **no jump is not proof of innocence.** Both count money raised after the prediction date, so both stay out even though they did not raise the score.

## Reproducibility

`model.ipynb` was run twice: once in the development environment and once in a fresh virtual environment built only from `requirements.txt` (plus `nbconvert`/`ipykernel` to execute it) in a clean copy of the repo. Every printed output was identical, and `app_data/test_scored.csv` and `app_data/meta.json` had identical MD5 hashes.
