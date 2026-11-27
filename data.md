# Data note (Option A: real data)

## Source and license

- **Dataset:** "Startup Investments" on Kaggle (kaggle.com/datasets/justinas/startup-investments), a CSV export of the **Crunchbase 2013 Snapshot © 2013**: companies, people, funding rounds, acquisitions and IPOs as of December 12, 2013.
- **License:** Crunchbase released the 2013 Snapshot under the Creative Commons Attribution license (CC BY). Required attribution: "Crunchbase 2013 Snapshot © 2013". Source: data.crunchbase.com/docs/2013-snapshot.
- **How I got it:** the Kaggle CSVs. The copy used to build `data/` came from a public GitHub mirror of the same Kaggle files (`BorisVosk/KaggleStartupInvestments`, folder `input_data`). Checks that it is the same data: `funding_rounds.csv` and `acquisitions.csv` are byte-identical (same MD5) to a second, independent mirror (`Elias-Mld/vc-exit-performance-analysis`), and every row count reproduces my checkpoint (462,651 objects, 196,553 companies, 31,507 with a dated round, 8,369 in the 5-year window, 1,203 exits).

## What is in `data/`

`objects.csv` is 285 MB, over GitHub's 100 MB limit, and contains 226,708 people. `data/make_extracts.py` keeps only company records and only the columns the notebook uses, and drops every name, URL, Twitter handle and description. It does not filter rows inside the tables; all cleaning happens in `model.ipynb`.

| File | Rows | Columns kept |
|---|---|---|
| `companies.csv` | 196,553 | id, category_code, status, founded_at, country_code, state_code, funding_rounds, funding_total_usd |
| `funding_rounds.csv` | 52,928 | funding_round_id, object_id, funded_at, funding_round_type, funding_round_code, raised_amount_usd, participants, created_at |
| `acquisitions.csv` | 9,562 | acquisition_id, acquired_object_id, acquired_at |
| `ipos.csv` | 1,259 | ipo_id, object_id, public_at |

`status`, `funding_rounds` and `funding_total_usd` are kept only to build the target and to show what leakage would do. They are never model features. No column identifies a person. Modeling table: 8,214 rows, 7 features, 1 target.

## Problems the file hid, and what I did about each

| # | Problem | Evidence (from `model.ipynb`) | What I did |
|---|---|---|---|
| 1 | **Placeholder dates** | 120 first rounds dated before 2000, 7 of them exactly 1960-01-01. 5 founding dates on 1901-01-01. 21% of founding dates fall on January 1 (year-only precision). 594 companies show a first round before their founding date, 171 by more than a year. | Start the window at 2000-01-01 (drops 120 rows). Founding dates before 1980 become missing. Age more than a year negative becomes missing; within a year, set to 0. |
| 2 | **Zeros that mean missing** | 14.6% of first rounds show $0 raised and 37% show 0 investors (11.6% and 16.2% in the modeling rows). A round of $0 did not happen; the value was not recorded. | Zero becomes missing, plus a "missing" flag. The flag for investors turned out to be the strongest single signal, so I tested it for leakage (`verification.md`). |
| 3 | **Exits with no date** | 600 of 1,259 IPO rows and 29 of 9,562 acquisitions have no date. 35 companies in the window say `acquired` or `ipo` but have no dated exit, so the checkpoint counted them as non-exits. | Drop those 35: we cannot tell whether the exit was inside 5 years. 8,249 rows become 8,214. |
| 4 | **One round type coded two ways** | 1,419 rows typed `series-a` are coded `seed` and 413 are coded `angel`. `venture` rounds are `unattributed` or `partial`. `other` mixes 3,409 debt rounds and 776 grants. | One `stage` label: the code wins when it says seed or angel; debt and grants get their own labels. |
| 5 | **Duplicates** | 150 companies have more than one acquisition row, 5 more than one IPO row, 257 have two or more rounds on their first date. | Use the earliest exit. Sum same-day rounds' amounts and investors; keep the most advanced stage. |
| 6 | **Back-filled records** | Crunchbase records start 2007-05-27. Median lag from first round to record creation: 3,216 days for 2001 rounds, 342 for 2008. 45.5% of exits had their first-round record typed in *after* the exit. | Can't be fixed, only bounded. The time split tests on the most promptly recorded cohort (2008), and the leakage check compares rounds recorded within a year to back-filled ones. Older cohorts over-represent companies that later became notable, which is one reason the training base rate (15.2%) exceeds the test rate (11.5%). |

## Funnel

| Step | Rows |
|---|---|
| Companies with at least one dated funding round | 31,507 |
| After dropping exits dated before the first round | 31,305 |
| First round on or before 2008-12-12 (checkpoint) | 8,369 |
| First round on or after 2000-01-01 | 8,249 |
| After dropping exits on an unknown date | **8,214** (1,185 exits, 14.4%) |
