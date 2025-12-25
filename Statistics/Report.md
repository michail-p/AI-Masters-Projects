# Group 31 – Micro-Breaks and Productivity

**Course:** Statistics for Engineers  
**Assignment:** Inference fundamentals (Homework 2)  
**Dataset:** 31. Work break and productivity score  
**Story:** A 30-person UX design firm piloted a new scheduling policy. After noticing afternoon slow-downs, the operations lead asked 15 designers to try a break-enriched schedule (5-minute pauses every 90 minutes) for two weeks, recording productivity scores (0–120 scale) after each phase. The goal is to decide whether structured micro-breaks materially improve output and whether the baseline productivity meets the firm’s target of 100 points.

---

## Data overview

| Schedule | n | Mean | SD | Median | IQR |
| --- | --- | --- | --- | --- | --- |
| Baseline schedule | 15 | 96.53 | 3.64 | 96 | 4 |
| Break-enriched schedule | 15 | 100.00 | 3.55 | 100 | 4 |

Visualisations produced in `group31_analysis.R`:

- Boxplot and jitter overlay comparing productivity by schedule.
- Kernel density plots for each schedule.
- QQ-plot of within-person differences to assess normality.

These plots show only mild skewness and no obvious outliers. Differences look roughly symmetric.

---

## Research questions

1. **Baseline adequacy**  
   *Question:* Under the current baseline schedule, is mean productivity below the firm target of 100?  
   *Hypotheses:*  
   \(H_0: \mu_{\text{baseline}} = 100\) vs. \(H_1: \mu_{\text{baseline}} < 100\)  
   *Significance:* \(\alpha = 0.05\) (company-wide thresholds use 5%).

2. **Micro-break benefit**  
   *Question:* Does introducing micro-breaks increase productivity relative to the baseline schedule?  
   *Design rationale:* Same employees measured in both conditions → paired comparison is appropriate.  
   *Hypotheses:*  
   \(H_0: \mu_{\text{diff}} = 0\) vs. \(H_1: \mu_{\text{diff}} > 0\), where \(\mu_{\text{diff}} = \mathbb{E}[\text{Break-enriched} - \text{Baseline}]\).  
   *Significance:* \(\alpha = 0.05\) chosen to balance false alarms and business risk.

---

## Methods and assumptions

- Single-sample and paired t-tests run with base R (`t.test`).
- Paired differences checked for approximate normality (Shapiro–Wilk test, QQ-plot). No severe deviations observed (Shapiro \(p = 0.41\)).
- Independence comes from assuming the 15 designers represent distinct employees, and measurement errors are uncorrelated. Data were collected over two short windows to minimise secular change.
- Productivity scores are on an interval scale; variance homogeneity is not required for the paired design.

---

## Key results

### Baseline schedule vs. target

- Test statistic: \(t = -3.69\) (df = 14), \(p = 0.0012\).
- 95% CI for \(\mu_{\text{baseline}}\): [94.5, 98.6].

**Interpretation:** Baseline productivity is significantly below the target of 100 points. The CI suggests the true mean is roughly 96–99, short of expectations.

### Break-enriched vs. baseline (paired)

- Mean difference (Break – Baseline): 3.47 points (SD = 6.15).
- Paired t-statistic: \(t = 2.18\) (df = 14), one-sided \(p = 0.0466\).
- 95% CI for mean difference (two-sided): [0.08, 6.86].
- Hedges’ g (paired): 0.56 (95% CI: 0.01, 1.10) → moderate effect.

**Interpretation:** The micro-break schedule yields a statistically significant productivity lift (≈3.5 points). Although the mean under micro-breaks equals 100, the CI overlaps slightly below 100, so the firm should confirm that 3–4 extra points translate to meaningful output.

---

## Practical implications

- **Baseline issue:** Without changes, designers underperform the target, potentially affecting project timelines.
- **Micro-break recommendation:** Structured micro-breaks appear to recover the shortfall and bring productivity to target. Implementing breaks firm-wide is supported, provided logistics (meeting schedules, coordination) are manageable.
- **Effect size:** A moderate practical impact suggests the policy could improve overall throughput by 3–4%. Managers should verify this aligns with KPIs before adopting.

---

## Limitations & next steps

- Sample size is modest (15 employees) from one firm → generalisation is limited.
- Productivity scores rely on internal metrics; measurement consistency should be validated.
- Study duration is short; long-term sustainability of the micro-break effect remains unknown.
- Future work: Repeat measurements after a month, explore subjective fatigue ratings, and analyse subgroup effects (e.g., junior vs. senior designers).

---

## Files generated

- `group31_analysis.R`: Complete data analysis, plots, tests, and CSV exports.
- `group31_summary_stats.csv`: Descriptive table.
- `group31_paired_test.csv`: Test statistics for the paired comparison.
- `group31_baseline_test.csv`: Single-sample test summary.
- `group31_effect_size.csv`: Paired effect-size estimate.
