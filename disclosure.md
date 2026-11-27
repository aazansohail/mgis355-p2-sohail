# AI disclosure

**Agent used:** Claude (Anthropic, Claude Opus 5.5), in an agent session that could run code.

## What the agent did
- **Data access.** Kaggle downloads were blocked in its sandbox, so it found a public GitHub mirror of the same Kaggle files and checked it two ways: MD5 match against a second, independent mirror, and every row count matching my checkpoint (462,651 / 196,553 / 31,507 / 8,369 / 1,203).
- **Code.** Wrote `data/make_extracts.py`, `model.ipynb` (from the five steps and my checkpoint decisions), and `app.py`. Ran the notebook twice in separate environments to confirm identical outputs, and tested the app headlessly, including that no input combination shows `nan`.
- **Data problems.** Found problems 2 to 6 in `data.md`: zeros that mean missing, undated exits, conflicting round codes, duplicates, back-filled records. (Problem 1, the placeholder dates, I had already flagged at the checkpoint.)
- **Docs.** Drafted the model card, data note and verification log from the notebook's outputs.

## What I did
- Chose the decision, the grain, the target definition, the data source, the 10:1 cost ratio, the feature list and the time split in my checkpoint (Nov 15). The final project keeps all of them.
- Found the placeholder first-round dates (1960-01-01, pre-2000) at the checkpoint.
- **[EDIT: list what you personally checked, for example: re-ran `model.ipynb` top to bottom and got the same numbers; traced the 21.8% example by hand; tried the app with a missing investor count.]**
- Deployed the app, and recorded and narrated the video myself.

## Where I overrode the agent
- **[EDIT: name at least one real change you made to its work, and why.]**

## One place the agent was wrong
Its first version of the app showed an "expected cost of meeting vs passing", calculated straight from the predicted probability. When the cost slider moved, the app could say "Pass" while the expected-cost line said meeting was cheaper. The cause: the model over-predicts on the 2008 cohort (13.6% predicted vs 11.5% actual), so its raw probabilities can't be multiplied by costs as if they were exact. The 0.13 threshold already corrects for that, because it was tuned on validation data. The expected-cost display was removed; the action now comes only from the threshold.

A second, smaller one: its first tuning grid put the chosen logistic-regression setting (C = 0.01) on the edge of the grid, so a better value might have been just outside it. Widening the grid to 0.001 confirmed 0.01 is a true best, not an edge artifact.
