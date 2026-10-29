# Questions for the presenter: Bertran et al. (2024), "Reconstruction Attacks on Machine Unlearning: Simple Models Are Vulnerable"

Asked by Michail Pettas, 2 October 2026. There is only about 2 minutes of Q&A, so pick 1–2 from the top.

## Main questions

1. **White-box threat model.** The attacker needs the exact parameters β+ and β− before and after the deletion. Most deployed models are only reachable through an API. For linear models, though, the parameters can be extracted from about d + 1 prediction queries. Does that make the white-box assumption close to harmless for the models the paper targets? Or does the attack break when β± is only estimated?

2. **Batched deletions.** The experiments delete exactly one record and retrain. In practice, providers batch requests and retrain periodically, often adding new data at the same time. Then Ĥ(β+ − β−) is a sum of many gradients. How quickly does reconstruction quality drop as the batch grows, and is batching already an effective, cheap defence?

3. **Defences are not evaluated.** The conclusion recommends differential privacy, but no DP or noisy "certified removal" model is attacked. What ε or noise level would push HRec back to the Avg/MaxDiff baseline? What would that cost in accuracy on ACS Income, where r² is only about 0.3?

## Follow-up questions

1. **Public data assumption.** The Hessian/covariance is estimated from public samples of the *same* distribution. How sensitive is the attack to distribution shift, or to having only a few hundred public samples? And to the unknown regularisation strength λ, which the attack sets to 0?

2. **Is cosine similarity the right harm measure?** For tabular records, harm comes from recovering specific sensitive attributes, such as exact income or disability status, not from the overall direction of the vector. How accurately are individual attributes recovered, and what counts as a "successful" reconstruction?

3. **Beyond the last layer.** The Newton step is exact for linear regression and approximate for convex losses. Does anything survive for full fine-tuning of a deep network, where the embedding φ also changes after deletion?

4. **Legal paradox.** Membership inference against these linear models is at random-guess level (Appendix A), yet exercising the GDPR right to erasure enables near-perfect reconstruction. Should controllers delay or batch erasures, or tell users about this risk? Would that conflict with the "without undue delay" requirement of Article 17?

5. **Link to my paper (Shokri et al. 2021).** H⁻¹∇ℓ is exactly the influence-function quantity behind example-based explanations (Koh & Liang). If a system offers both influence-based explanations and unlearning, do the two leaks compound? Could explanations give the attacker the gradient information even without seeing β−?
