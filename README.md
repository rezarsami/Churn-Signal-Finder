# Churn Signal Finder — Retention Ops Console

**A product-operations case study: mining a churn signal out of a messy user export, testing that it's real, and quantifying the revenue at stake.**

Built to demonstrate the two skills product-operations-analyst postings ask for most — **data mining** (segmenting a raw dataset to find a non-obvious pattern) and **dashboard building** (turning that finding into a decision-ready view) — with the statistical rigor to back the claim up.

---

## The problem

Northwind (a fictional team-scheduling SaaS) has a blended churn rate of **17.9%** — unremarkable on its own. Leadership wants to know whether there's anything actionable underneath that number.

## The finding

Testing churn across **all 16 plan × region cells** surfaces exactly one pocket the blended rate hides:

> **Business-tier customers in APAC churn at 30% — 1.7× baseline.** The pocket appears only after Sep 2024, coincides with a **3.1× jump in support tickets** and a CSAT drop from 4.0 to 3.0, and puts **~$178K in monthly MRR at risk**. At an assumed 40% save rate, recovering it is worth **~$850K/year**.

That signature — regional + tier-specific + time-bounded + ticket/CSAT-corroborated — points to a product defect, not a pricing problem. The recommendation is a targeted save-play plus an Eng defect investigation, not a discount.

## Why this is more than a heatmap

The analysis doesn't just eyeball the highest cell. It **tests every cell** against the rest of the customer base with a two-proportion z-test, **corrects for running 16 comparisons** (Bonferroni), and flags a pocket only if it's *both* statistically distinguishable from baseline *and* large enough to act on (≥1.4× the baseline rate). Of 16 cells, **exactly one** clears both bars. The other 15 stay quiet — which is what makes the one flag credible instead of cherry-picked.

The synthetic data is generated from a **documented logistic churn model** with the APAC-Business effect planted as a known coefficient (`ground_truth.json`). This lets the project prove something a real dataset can't: that the *method* recovers a known signal and doesn't invent signals where none exist.

---

## Why this maps to the role

| Job requirement | Where it shows up here |
|---|---|
| Data mining | Cleaning pipeline + plan × region × cohort segmentation that isolates the anomaly |
| Statistical rigor | Two-proportion tests, multiple-comparison correction, magnitude threshold |
| Dashboard building | Self-contained interactive console (KPIs, tested heatmap, trend, revenue views) |
| Turning data into decisions | Every view ends in a dollar figure and a recommended action |
| Working with messy data | Raw export ships with duplicates, null regions, inconsistent casing — handled explicitly |
| Reproducibility | Rerun `gen_data.py` → `analysis.py` and every dashboard number regenerates |

---

## What's in this folder

- **`churn_signal_finder_dashboard.html`** — the interactive dashboard. Open in any browser; no server needed.
- **`analysis.py`** — the cleaning, testing, and sizing pipeline (pandas + scipy).
- **`gen_data.py`** — generates the synthetic dataset from a documented logistic model.
- **`ground_truth.json`** — the planted model coefficients, for verifying the method.
- **`churn_users_raw.csv`** — the raw dataset the analysis runs on.

## Run it yourself

```bash
pip install pandas numpy scipy
python gen_data.py      # regenerates the raw CSV + ground_truth.json
python analysis.py      # cleans, tests, sizes, writes dashboard_data.json
# then open churn_signal_finder_dashboard.html
```

## The pipeline

**Extract** → 12,080 rows via pandas · **Clean** → drop 80 dupes, fill null regions, fix plan casing, impute engagement medians · **Mine + Test** → two-proportion z-test on all 16 plan × region cells, Bonferroni-corrected, magnitude-gated · **Ship** → export to JSON, render as the console.

## Stack

Python · pandas · numpy · scipy · Chart.js

---

## How to talk about it in an interview

**The 30-second version:**
> "A blended 18% churn rate hid a pocket — Business-tier APAC customers churning at 30%. I found it by testing all 16 plan × region cells against baseline, correcting for multiple comparisons, and keeping only cells that were both significant and big enough to matter. One survived. I traced it to a September break, corroborated it with support tickets and CSAT, and sized it at ~$178K monthly MRR at risk — about $850K/year recoverable at a 40% save rate. Then I packaged it as a dashboard the CS and Eng teams could act on."

**Questions you should be ready for (and honest answers):**

- *"Why 1.4× as the flag threshold?"* — It's a business judgment, not a statistical rule. I wanted cells that are both statistically real *and* materially large. I'd tune it with stakeholders; the point is separating "significant" from "worth acting on," since with a big enough n even trivial differences turn significant.
- *"The data is synthetic — didn't you just find what you planted?"* — Yes, deliberately. The planted effect is the ground truth that lets me verify the *method* works: it recovers the known signal and, just as importantly, doesn't flag the 15 cells where nothing was planted. Swap in a real export and the same pipeline runs unchanged.
- *"Where's the 40% save rate from?"* — It's an explicit assumption, flagged as one. A finding isn't an outcome until someone acts on it, so I framed the money as recoverable-under-an-assumption rather than a guaranteed number.

> Dataset is synthetic and built specifically for this demonstration. The anomaly is intentionally planted so the analysis has a known truth to recover.
