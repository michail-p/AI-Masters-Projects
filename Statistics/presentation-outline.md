# Presentation

**Title:** Micro-breaks as a Solution to Productivity Slumps in a UX Design Firm  

---

## 1. Hook & Story

- Introduce the firm: 30-person UX team noticing afternoon productivity dips.
- Describe baseline practice (only a lunch break) versus pilot policy (5-minute micro-break every 90 minutes).
- State the managerial questions:
  1. Is current productivity already at the 100-point target?
  2. Do structured micro-breaks improve output enough to justify adoption?

*Visual:* Slide with the story timeline and the two schedules.

---

## 2. Data & Exploration

- Dataset: 15 designers measured under both schedules (paired design).
- Show key descriptive table (mean, SD) and boxplot/jitter chart.
- Mention that scores span 90–104, no extreme outliers.

*Visuals:* Boxplot + density facets. Highlight overlap but slight shift upward for break schedule.

---

## 3. Statistical Questions & Methods

- **Q1:** Baseline adequacy
  - Hypotheses: H0 μ = 100 vs. H1 μ < 100 at α = 0.05.
  - Test: One-sample t-test (justified by approximate normality; independence across employees).
- **Q2:** Micro-break effect
  - Hypotheses: H0 mean diff = 0 vs. H1 > 0 at α = 0.05.
  - Test: Paired t-test (same employees; differences checked with Shapiro–Wilk, QQ-plot).
- Mention assumptions checked (normality of differences, independence, measurement scale).

*Visual:* Slide listing hypotheses and decisions, with small QQ-plot inset.

---

## 4. Results & Interpretation

- Baseline mean = 96.5 (CI 94.5–98.6), t = −3.69, p = 0.001 → Baseline underperforms target.
- Paired difference mean = 3.47 (CI 0.08–6.86), t = 2.18, one-sided p = 0.047.
- Effect size Hedges’ g ≈ 0.56 (moderate practical impact).
- Translate numbers into business terms: expected 3–4 point gain brings average to 100.

*Visual:* Table summarising statistics; highlight significant results.

---

## 5. Conclusions & Recommendations

- Adopt micro-break schedule across UX teams with monitoring.
- Without breaks, target is consistently missed; micro-breaks close the gap.
- Suggest rollout plan plus follow-up measurement.

*Visual:* Bullet list of recommendations and next steps.

---

## 6. Limitations & Future Work

- Small sample, single company; short timeframe.
- Metrics internal; need validation/external replication.
- Future: monitor long-term effects, split by seniority, gather qualitative feedback.
