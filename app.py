"""Startup exit screener: MGIS 355 Project 2 (Aazan Sohail).

The model is trained in model.ipynb and loaded here with joblib. Nothing is refit in this app.
"""
import json
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent
st.set_page_config(page_title="Startup exit screener", layout="wide")


# ---------- load once ----------
@st.cache_resource
def load_model():
    return joblib.load(ROOT / "model.joblib")


@st.cache_data
def load_meta():
    return json.loads((ROOT / "app_data" / "meta.json").read_text())


@st.cache_data
def load_test():
    return pd.read_csv(ROOT / "app_data" / "test_scored.csv")


model = load_model()
meta = load_meta()
test = load_test()
y = test["exited"].to_numpy()
p_test = test["p_exit"].to_numpy()
base_flag = test["baseline_flag"].to_numpy().astype(bool)
CHOSEN_T = float(meta["threshold"])
BASE_RATE = float(meta["test_base_rate"])

STAGE_LABELS = {
    "seed-angel": "Seed / angel",
    "series-a": "Series A",
    "series-b-or-later": "Series B or later",
    "venture-unlabeled": "Venture round, series not stated",
    "debt": "Debt round",
    "grant": "Grant",
    "other": "Other (private equity, crowdfunding, secondary)",
}
COUNTRY_LABELS = {
    "USA": "United States", "GBR": "United Kingdom", "CAN": "Canada", "ISR": "Israel",
    "DEU": "Germany", "FRA": "France", "IND": "India", "other": "Another country", "unknown": "Not recorded",
}
STATE_LABELS = {
    "CA": "California", "NY": "New York", "MA": "Massachusetts", "WA": "Washington", "TX": "Texas",
    "US-other": "Another US state", "US-unknown": "State not recorded",
}


def money(x):
    return f"${x:,.0f}"


def pct(x, d=0):
    return f"{100 * x:.{d}f}%"


def confusion(flag):
    flag = np.asarray(flag, bool)
    return dict(tp=int((flag & (y == 1)).sum()), fp=int((flag & (y == 0)).sum()),
                fn=int((~flag & (y == 1)).sum()), tn=int((~flag & (y == 0)).sum()))


def cost_per_100(flag, c_fp, c_fn):
    c = confusion(flag)
    return (c["fp"] * c_fp + c["fn"] * c_fn) / len(y) * 100


# ---------- header ----------
st.title("Startup exit screener")
st.markdown(
    "**For a VC associate or accelerator screener who decides, one newly funded startup at a time, "
    "whether to take a first meeting.** It estimates the chance that a startup is acquired or goes public "
    "within 5 years of its first funding round, and turns that into *meet* or *pass*."
)
st.caption(
    "Real data, not synthetic. Source: Crunchbase 2013 Snapshot © 2013 (CC BY), via Kaggle. Companies first funded 2000 to 2008. "
    f"Logistic regression, tested on the {meta['test_rows']:,} startups first funded "
    f"{meta['test_dates'][0]} to {meta['test_dates'][1]}."
)

# ---------- sidebar: costs and threshold ----------
with st.sidebar:
    st.header("Your costs")
    meeting_usd = st.number_input("Cost of a meeting that goes nowhere ($)", min_value=50, max_value=5000,
                                  value=int(meta["meeting_cost_usd"]), step=50,
                                  help="Associate time for prep, the meeting, and notes. Assumed ~3 hours at ~$100/hour.")
    ratio = st.slider("A missed exit costs how many wasted meetings?", min_value=2, max_value=50,
                      value=int(meta["cost_fn"] / meta["cost_fp"]), step=1,
                      help="The checkpoint assumption is 10: missing a startup that exits is at least 10x worse than one wasted meeting.")
    c_fp, c_fn = float(meeting_usd), float(meeting_usd) * ratio
    st.header("Meeting threshold")
    T = st.slider("Take the meeting if the predicted exit chance is at least", min_value=0.02, max_value=0.40,
                  value=CHOSEN_T, step=0.005, format="%.3f")
    st.caption(f"Chosen threshold: **{CHOSEN_T:.3f}**, the lowest-cost cutoff on the validation set at a 10:1 cost ratio. "
               "It was set before the test set was scored.")
    _sens = pd.DataFrame(meta["sensitivity"])
    _near = _sens.iloc[(_sens["fn_to_fp_ratio"] - ratio).abs().argmin()]
    if int(_near["fn_to_fp_ratio"]) != 10 or ratio != 10:
        st.caption(f"For a ratio of {int(_near['fn_to_fp_ratio'])}:1, the validation set picks a threshold of "
                   f"**{_near['threshold']:.3f}**.")

tab_score, tab_good, tab_limits = st.tabs(["1 · Score a startup", "2 · How good is this?", "3 · When not to trust it"])

# ---------- 1. prediction view ----------
with tab_score:
    st.subheader("Enter what you know on the day of the first round")
    r = meta["ranges"]
    c1, c2, c3 = st.columns(3)
    with c1:
        stage = st.selectbox("Round stage", list(STAGE_LABELS), index=list(STAGE_LABELS).index("series-a"),
                             format_func=STAGE_LABELS.get)
        amt_unknown = st.checkbox("Amount not disclosed")
        amount = st.number_input("Amount raised in the first round ($)", min_value=int(r["amount_usd"][0] // 1000 * 1000 + 1000),
                                 max_value=int(r["amount_usd"][1] // 100000 * 100000), value=5_000_000, step=250_000,
                                 disabled=amt_unknown, help="Limited to the range covering 98% of training startups.")
        st.caption("Not disclosed" if amt_unknown else f"= {money(amount)}")
    with c2:
        inv_unknown = st.checkbox("Investor count not recorded")
        n_inv = st.slider("Investors in the round", min_value=int(r["n_investors"][0]), max_value=int(r["n_investors"][1]),
                          value=3, disabled=inv_unknown)
        age_unknown = st.checkbox("Founding date unknown")
        age = st.slider("Company age at the round (years)", min_value=0.0, max_value=float(np.floor(r["company_age_years"][1])),
                        value=2.0, step=0.5, disabled=age_unknown)
    with c3:
        cats = list(meta["categories"]["category"])
        category = st.selectbox("Industry", cats, index=cats.index("software"),
                                format_func=lambda c: c.replace("_", " ") + ("" if meta["categories"]["category"][c] >= 30 else " (rare in training)"))
        countries = [c for c in COUNTRY_LABELS if c in meta["categories"]["country"]]
        country = st.selectbox("Headquarters country", countries, index=countries.index("USA"), format_func=COUNTRY_LABELS.get)
        if country == "USA":
            states = list(STATE_LABELS)
            us_state = st.selectbox("US state", states, index=states.index("CA"), format_func=STATE_LABELS.get)
        else:
            us_state = "non-US"

    case = pd.DataFrame([dict(
        amount_usd=np.nan if amt_unknown else float(amount),
        n_investors=np.nan if inv_unknown else float(n_inv),
        company_age_years=np.nan if age_unknown else float(age),
        stage=stage, category=category, country=country, us_state=us_state,
    )])
    prob = float(model.predict_proba(case)[:, 1][0])
    meet = prob >= T
    rank = float((p_test < prob).mean())

    st.divider()
    m1, m2, m3 = st.columns([1, 1, 1.4])
    m1.metric("Chance of exit within 5 years", pct(prob, 1))
    m2.metric("Action at your threshold", "Meet" if meet else "Pass")
    m3.metric("Ranks above", f"{pct(rank)} of test-set startups")
    st.info(
        f"This startup's estimated chance of being acquired or going public within 5 years is **{pct(prob, 1)}**, "
        f"{prob / BASE_RATE:.1f}x the {pct(BASE_RATE, 1)} rate among the test-set startups, and higher than "
        f"{pct(rank)} of them. It is {'above' if meet else 'below'} your threshold of {pct(T, 1)}, so the call is "
        f"**{'take the meeting' if meet else 'pass for now'}**."
    )
    st.caption("The estimate is a ranking signal, not a valuation. It says nothing about how large an exit would be.")

# ---------- 2. how good is this ----------
with tab_good:
    st.subheader(f"Test set: {meta['test_rows']:,} startups first funded {meta['test_dates'][0]} to {meta['test_dates'][1]}")
    flag = p_test >= T
    cm = confusion(flag)
    cb = confusion(base_flag)
    n = len(y)
    rules = pd.DataFrame([
        dict(Rule=f"Model at threshold {T:.3f}", flag=flag),
        dict(Rule=f"Baseline: Series A or later, or raised at least {money(meta['baseline_cutoff_usd'])}", flag=base_flag),
        dict(Rule="Meet everyone", flag=np.ones(n, bool)),
        dict(Rule="Meet no one (predict no exit)", flag=np.zeros(n, bool)),
    ])
    rows = []
    for _, rr in rules.iterrows():
        c = confusion(rr["flag"])
        rows.append({
            "Rule": rr["Rule"],
            "Meetings per 100 startups": f"{100 * (c['tp'] + c['fp']) / n:.0f}",
            "Exits caught": f"{c['tp']} of {int(y.sum())} ({pct(c['tp'] / y.sum())})",
            "Accuracy": pct((c["tp"] + c["tn"]) / n),
            "Cost per 100 startups": money(cost_per_100(rr["flag"], c_fp, c_fn)),
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    model_cost = cost_per_100(flag, c_fp, c_fn)
    base_cost = cost_per_100(base_flag, c_fp, c_fn)
    all_cost = cost_per_100(np.ones(n, bool), c_fp, c_fn)
    best_other = min(base_cost, all_cost, cost_per_100(np.zeros(n, bool), c_fp, c_fn))
    k = int(base_flag.sum())
    topk = np.zeros(n, bool)
    topk[np.argsort(-p_test, kind="stable")[:k]] = True
    tk = confusion(topk)["tp"]
    verdict = "saves" if model_cost < base_cost else "costs"
    st.success(
        f"At a threshold of {T:.3f}, the model takes **{100 * (cm['tp'] + cm['fp']) / n:.0f} meetings per 100 startups** and catches "
        f"**{pct(cm['tp'] / y.sum())}** of the startups that exited, versus {100 * (cb['tp'] + cb['fp']) / n:.0f} meetings and "
        f"{pct(cb['tp'] / y.sum())} for the baseline rule. At {money(c_fp)} per wasted meeting and {money(c_fn)} per missed exit, "
        f"it **{verdict} {money(abs(base_cost - model_cost))} per 100 startups** compared with the baseline"
        f"{'' if model_cost < best_other else ', and it is not the cheapest option at these settings'}. "
        f"With the same {k:,} meetings as the baseline, the model's top-ranked startups include {tk} exits versus {cb['tp']}."
    )
    st.caption("Accuracy is shown to make a point: 'meet no one' is the most accurate rule and the most expensive one.")

    left, right = st.columns([1, 2])
    with left:
        st.markdown("**Confusion matrix at your threshold**")
        st.dataframe(pd.DataFrame({"Take the meeting": [cm["tp"], cm["fp"]], "Pass": [cm["fn"], cm["tn"]]},
                                  index=["Exited within 5 years", "Did not exit"]), width="stretch")
        st.markdown(
            f"- Wasted meetings: {cm['fp']} x {money(c_fp)} = **{money(cm['fp'] * c_fp)}**\n"
            f"- Missed exits: {cm['fn']} x {money(c_fn)} = **{money(cm['fn'] * c_fn)}**\n"
            f"- Precision: {pct(cm['tp'] / max(cm['tp'] + cm['fp'], 1))} of meetings are with a future exit"
        )
    with right:
        ts = np.round(np.arange(0.02, 0.401, 0.005), 3)
        curve = pd.DataFrame({
            "threshold": np.concatenate([ts, ts, ts]),
            "cost per 100 startups ($)": np.concatenate([[cost_per_100(p_test >= t, c_fp, c_fn) for t in ts],
                                                         np.full(len(ts), base_cost), np.full(len(ts), all_cost)]),
            "rule": ["model"] * len(ts) + ["baseline rule"] * len(ts) + ["meet everyone"] * len(ts),
        })
        lines = alt.Chart(curve).mark_line().encode(
            x=alt.X("threshold:Q", title="Threshold (predicted exit chance)"),
            y=alt.Y("cost per 100 startups ($):Q", scale=alt.Scale(zero=False)),
            color=alt.Color("rule:N", legend=alt.Legend(orient="bottom", title=None)),
            strokeDash=alt.condition(alt.datum.rule == "model", alt.value([1, 0]), alt.value([5, 4])),
        )
        rule = alt.Chart(pd.DataFrame({"t": [T]})).mark_rule(color="gray").encode(x="t:Q")
        st.markdown("**Screening cost as the threshold moves**")
        st.altair_chart(lines + rule, width="stretch")
        st.caption("Gray line: your threshold. Lower is better. Costs use the values in the sidebar.")

    st.markdown(
        f"**Ranking quality:** test AUC {meta['test_auc']:.3f} for logistic regression, versus "
        f"{meta['test_auc_other']:.3f} for a gradient-boosted model that had to beat it to be shipped and did not. "
        "An AUC of 0.68 means the model ranks a random exited startup above a random non-exited one about two times in three: "
        "useful for sorting a pipeline, far from certain on any single company."
    )
    with st.expander("Which threshold fits other cost ratios (chosen on validation, scored on test)"):
        sens = pd.DataFrame(meta["sensitivity"])
        st.dataframe(pd.DataFrame({
            "Missed exit = N wasted meetings": sens["fn_to_fp_ratio"],
            "Threshold": sens["threshold"].map(lambda v: f"{v:.3f}"),
            "Meetings per 100": sens["meetings_per_100"].round(0).astype(int),
            "Model beats baseline, meet-all and meet-none?": sens["model_beats_all_others"].map({True: "yes", False: "no"}),
        }), hide_index=True, width="stretch")

# ---------- 3. limits ----------
with tab_limits:
    st.subheader("When not to trust it")
    flag = p_test >= T
    seg = []
    groups = {
        "Investor count not recorded": test["n_investors"].isna().to_numpy(),
        "Investor count recorded": test["n_investors"].notna().to_numpy(),
        "Outside the US": (test["region"] == "non-US").to_numpy(),
        "United States": (test["region"] == "US").to_numpy(),
        "Seed / angel first round": (test["stage"] == "seed-angel").to_numpy(),
        "Series B or later first round": (test["stage"] == "series-b-or-later").to_numpy(),
    }
    for name, m in groups.items():
        caught = int((flag & m & (y == 1)).sum())
        seg.append({"Segment": name, "Startups": int(m.sum()), "Exits": int(y[m].sum()),
                    "Exits caught at your threshold": f"{caught} of {int(y[m].sum())} ({pct(caught / max(y[m].sum(), 1))})",
                    "Meetings per 100": f"{100 * flag[m].mean():.0f}"})
    st.dataframe(pd.DataFrame(seg), hide_index=True, width="stretch")
    w = meta["weakest_segment"]
    no_inv = groups["Investor count not recorded"]
    nonus = groups["Outside the US"]
    st.warning(
        f"**Weakest segment: startups whose first round lists no investors.** Test AUC there is {w['auc']:.2f} (close to a coin flip). "
        f"At your threshold the model catches {int((flag & no_inv & (y == 1)).sum())} of their {int(y[no_inv].sum())} exits, "
        "because a missing investor count pulls every score down. If you know the investors, enter them. "
        f"Outside the US it also misses more: it catches {pct((flag & nonus & (y == 1)).sum() / max(y[nonus].sum(), 1))} of non-US exits "
        f"versus {pct((flag & ~nonus & (y == 1)).sum() / max(y[~nonus].sum(), 1))} of US exits."
    )
    sens = pd.DataFrame(meta["sensitivity"])
    wins = sens[sens["model_beats_all_others"]]["fn_to_fp_ratio"]
    lo, hi = int(wins.min()), int(wins.max())
    hi_next = int(sens.loc[sens["fn_to_fp_ratio"] > hi, "fn_to_fp_ratio"].min())
    lo_prev = int(sens.loc[sens["fn_to_fp_ratio"] < lo, "fn_to_fp_ratio"].max())
    st.markdown(
        f"""
**Range it was trained on**
- First rounds dated {meta['train_dates'][0]} to {meta['train_dates'][1]} (training) and tested on {meta['test_dates'][0]} to {meta['test_dates'][1]}.
- Amounts {money(meta['ranges']['amount_usd'][0])} to {money(meta['ranges']['amount_usd'][1])}, 1 to {meta['ranges']['n_investors'][1]} investors, company age 0 to {meta['ranges']['company_age_years'][1]:.0f} years. The inputs on tab 1 are limited to these ranges.

**Calibration drifts with the market.** On the 2008 test cohort the model predicted an average exit chance of {100 * p_test.mean():.1f}%; the actual rate was {pct(BASE_RATE, 1)}. Read the score as a ranking first and a probability second.

**Do not use it for**
- Deciding whether to invest or how much. It predicts whether any exit happens, including small acquisitions, not how big the return is.
- Startups funded after 2013 without retraining. The data stops in December 2013, before today's seed market, and exit rates move with the cycle.
- Companies already past their first round. Every row is a company on the day of its first round.

**What would change the recommendation**
- The model is the cheapest option only when a missed exit costs between {lo} and {hi} wasted meetings (the table on tab 2). At {hi_next} or more, meeting everyone is as cheap or cheaper; at {lo_prev} or less, meeting no one is.
- If the screener can only take a fixed number of meetings, use the model as a ranking (take the top N) rather than a threshold.

**Data:** real, not synthetic. Crunchbase data is crowd-sourced and was partly back-filled after 2007, so older records over-represent companies that later became notable.
"""
    )
