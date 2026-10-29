# Data Privacy 2

Assignment 2 (Seminar) for the **Data Privacy course (5DV241)** at Umeå University.

This research seminar module explores cutting-edge privacy vulnerabilities in modern machine learning systems, focusing on **information leakage via model explanations** and **reconstruction attacks in machine unlearning**.

## What This Project Covers

- **Privacy Risks of Model Explanations:**
  - In-depth analysis of Shokri et al. (IEEE S&P 2021): *"On the Privacy Risks of Model Explanations"*.
  - Membership inference attacks against black-box and grey-box explanation interfaces (SHAP, feature influence, saliency maps).
  - Quantifying how much additional information explanations leak compared to prediction outputs alone, especially for outlier samples.
- **Machine Unlearning Reconstruction Attacks:**
  - Technical critique and Q&A analysis of Bertran et al. (NeurIPS 2024): *"Reconstruction Attacks on Machine Unlearning: Simple Models Are Vulnerable"*.
  - Evaluating parameter-difference attacks ($\theta^+ - \theta^-$) before and after unlearning requests.
- **Presentation & Written Report:**
  - Comprehensive seminar presentation slides on explanation privacy risks and countermeasures (differential privacy for explanation methods, regularization).
  - Formal seminar report with threat model diagrams and discussion of real-world mitigation strategies.

## Files

- `Privacy Risks of Model Explanations.pptx` - Presentation slide deck delivered in the seminar
- `Questions_Bertran2024.md` - Discussion questions and technical evaluation of unlearning attacks
- `report/Pettas_Report_Shokri2021.pdf` - Written seminar report
- `report/make_figure.py` - Python script generating explanation leakage flowcharts
- `report/figure1.png` / `report/figure1.svg` - Architecture diagram of explanation-based inference attacks

## Tech Stack

- Python 3 (matplotlib, svgwrite for architectural diagrams)
- Microsoft PowerPoint
- HTML/CSS (report generation)

## Notes

This is course assignment work intended for academic use.
