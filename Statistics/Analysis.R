# group31_analysis.R
# Analysis for Assignment 2: Inference fundamentals
# Story: Impact of structured micro-breaks on productivity in a UX design firm

library(tidyverse)
library(broom)
library(effectsize)

# -----------------------------------------------------------------------------
# 1. Data entry ----------------------------------------------------------------
# Productivity scores for 15 UX designers under two scheduling policies.
# Group A: Baseline schedule (single lunch break only)
# Group B: Break-enriched schedule (5-minute micro-break every 90 minutes)

productivity <- tibble(
  employee_id = 1:15,
  baseline = c(91, 96, 92, 100, 94, 95, 96, 93, 99, 99, 100, 95, 94, 100, 104),
  break_enriched = c(103, 95, 104, 103, 104, 103, 98, 97, 98, 98, 98, 101, 101, 104, 93)
)

productivity_long <- productivity |>
  pivot_longer(cols = c(baseline, break_enriched),names_to = "schedule",values_to = "score") |>
  mutate(schedule = factor(schedule, levels = c("baseline", "break_enriched"), labels = c("Baseline schedule", "Break-enriched schedule")))

# -----------------------------------------------------------------------------
# 2. Data visualization --------------------------------------------------------
# Exploratory plots to support the narrative and assumption checks.

viz_box <- ggplot(productivity_long, aes(x = schedule, y = score, fill = schedule)) +
  geom_boxplot(alpha = 0.7, width = 0.5, outlier.shape = NA) +
  geom_jitter(width = 0.1, size = 2, alpha = 0.8) +
  scale_fill_brewer(palette = "Set2") +
  labs(title = "Productivity scores by schedule", x = NULL, y = "Productivity score") +
  theme_minimal(base_size = 12) +
  theme(legend.position = "none")

ggplot(productivity_long, aes(x = score, fill = schedule)) +
  geom_density(alpha = 0.35) +
  facet_wrap(~ schedule) +
  scale_fill_brewer(palette = "Set2") +
  labs(title = "Distribution of productivity scores", x = "Score", y = "Density") + theme_minimal(base_size = 12)

viz_box

# -----------------------------------------------------------------------------
# 3. Descriptive statistics ----------------------------------------------------

summary_stats <- productivity_long |>
  group_by(schedule) |>
  summarise(n = n(), mean = mean(score), sd = sd(score), median = median(score), iqr = IQR(score))
summary_stats

# -----------------------------------------------------------------------------
# 4. Single-group question -----------------------------------------------------
# Q1: Under the baseline schedule, is mean productivity below the firm target of 100?
# H0: mu_baseline = 100
# H1: mu_baseline < 100 (alpha = 0.05)

baseline_test <- t.test(productivity$baseline, mu = 100, alternative = "less")
baseline_ci <- t.test(productivity$baseline, conf.level = 0.95)

baseline_test
baseline_ci$conf.int

# -----------------------------------------------------------------------------
# 5. Comparison question -------------------------------------------------------
# Q2: Do micro-breaks increase productivity relative to the baseline schedule?
# The same employees were observed twice ➜ paired analysis.
# H0: mean difference (break_enriched - baseline) = 0
# H1: mean difference > 0 (alpha = 0.05)

paired_test <- t.test(productivity$break_enriched, productivity$baseline, paired = TRUE, alternative = "greater")
paired_test

# Effect size: paired Cohen's d ------------------------------------------------
paired_d <- cohens_d(productivity$break_enriched, productivity$baseline, paired = TRUE, hedges.correction = TRUE)
paired_d

# -----------------------------------------------------------------------------
# 6. Assumption checks ---------------------------------------------------------
# Normality of paired differences (Shapiro-Wilk) and QQ-plot.

diffs <- productivity$break_enriched - productivity$baseline

shapiro.test(diffs)

ggplot(tibble(diff = diffs), aes(sample = diff)) +
  stat_qq() +
  stat_qq_line(color = "steelblue", linewidth = 0.7) + 
  labs(title = "QQ-plot of productivity differences", x = "Theoretical quantiles", y = "Observed quantiles") +
  theme_minimal(base_size = 12)

# -----------------------------------------------------------------------------
# 7. Model summaries for reporting --------------------------------------------

baseline_glance <- tidy(baseline_test)
paired_glance <- tidy(paired_test)

baseline_glance
paired_glance

# -----------------------------------------------------------------------------
# 8. Export-ready tables -------------------------------------------------------

write_csv(summary_stats, "group31_summary_stats.csv")
write_csv(tidy(paired_d), "group31_effect_size.csv")
write_csv(baseline_glance, "group31_baseline_test.csv")
write_csv(paired_glance, "group31_paired_test.csv")
