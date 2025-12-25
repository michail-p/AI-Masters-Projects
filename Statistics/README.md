# Statistics – Micro-Breaks and Productivity Analysis

**Course:** Statistics for Engineers  
**Assignment:** Inference Fundamentals (Homework 2)  
**Group:** 31  
**Author:** Michail Pettas

---

## 📋 Overview

This project analyzes the impact of structured micro-breaks on workplace productivity in a UX design firm. Using statistical inference methods, we investigate whether implementing 5-minute pauses every 90 minutes improves designer productivity.

---

## 🗂️ Project Structure

```
Statistics/
├── Analysis.R              # Complete R analysis script
├── Report.md               # Full analysis report
└── Presentation Outline.md # Presentation notes
```

---

## 📊 Study Design

### Background

A 30-person UX design firm piloted a new scheduling policy after noticing afternoon productivity slowdowns.

### Participants

- **Sample size:** 15 designers
- **Design:** Within-subjects (paired comparison)
- **Duration:** Two weeks per condition

### Conditions

1. **Baseline Schedule:** Standard workday with single lunch break
2. **Break-Enriched Schedule:** 5-minute micro-breaks every 90 minutes

### Outcome

- Productivity score (0-120 scale)
- Firm target: 100 points

---

## 🔬 Research Questions

### Question 1: Baseline Adequacy

*Is baseline productivity below the firm target of 100?*

**Hypotheses:**

- H₀: μ_baseline = 100
- H₁: μ_baseline < 100
- α = 0.05

### Question 2: Micro-Break Benefit

*Do micro-breaks increase productivity?*

**Hypotheses:**

- H₀: μ_diff = 0
- H₁: μ_diff > 0 (Break-enriched - Baseline)
- α = 0.05

---

## 📈 Key Results

### Descriptive Statistics

| Schedule | n | Mean | SD | Median |
|----------|---|------|----|---------|
| Baseline | 15 | 96.53 | 3.64 | 96 |
| Break-enriched | 15 | 100.00 | 3.55 | 100 |

### Statistical Tests

**Baseline vs Target (100):**

- t = -3.69, df = 14, p = 0.0012
- 95% CI: [94.5, 98.6]
- ✅ Baseline is significantly below target

**Paired Comparison:**

- Mean difference: 3.47 points
- t = 2.18, df = 14, p = 0.0466 (one-sided)
- 95% CI for difference: [0.08, 6.86]
- Hedges' g = 0.56 (moderate effect)
- ✅ Micro-breaks significantly improve productivity

---

## 🚀 Usage

### Running the Analysis

```r
# Open R or RStudio
# Set working directory
setwd("path/to/Statistics")

# Run the analysis script
source("Analysis.R")
```

### Required Packages

```r
install.packages(c("tidyverse", "broom", "effectsize"))
```

---

## 📊 Generated Outputs

| File | Description |
|------|-------------|
| `group31_summary_stats.csv` | Descriptive statistics table |
| `group31_baseline_test.csv` | Single-sample t-test results |
| `group31_paired_test.csv` | Paired t-test results |
| `group31_effect_size.csv` | Effect size estimates |

### Visualizations

- Boxplot with jitter overlay comparing schedules
- Kernel density plots for each condition
- QQ-plot for normality assessment

---

## 📦 Requirements

- **R 4.x+**
- **Packages:**
  - tidyverse (data manipulation & visualization)
  - broom (tidy statistical output)
  - effectsize (effect size calculations)

---

## ✅ Conclusions

1. **Baseline productivity is inadequate:** Mean of 96.5 is significantly below the 100-point target
2. **Micro-breaks work:** Structured breaks yield a statistically significant ~3.5 point improvement
3. **Effect size is moderate:** Hedges' g = 0.56 suggests meaningful practical impact
4. **Recommendation:** Implement firm-wide micro-break policy

---

## ⚠️ Limitations

- Small sample size (n=15) limits generalizability
- Short study duration (2 weeks)
- Single firm studied
- Productivity metric is internal/subjective

---

## 📚 Methods

- Single-sample t-test (baseline vs target)
- Paired t-test (within-subjects comparison)
- Shapiro-Wilk normality test
- Effect size calculation (Hedges' g)
