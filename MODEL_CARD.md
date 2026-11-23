# Model card: outreach uplift model

| | |
|---|---|
| **Purpose** | Estimate, for each customer, how much a lender's call changes the chance they apply for a home loan, and call the customers it helps most. |
| **Not for** | Credit decisions or pricing. It decides who gets a call, not what anyone is offered. |
| **Model** | S-learner on LightGBM (best of S-, T- and X-learners here), compared against propensity ranking. |
| **Inputs** | Income, savings and deposit readiness, rent, research activity, savings growth, buy-now-pay-later use, tenure, recent credit decline, recent marketing contacts. **Age is never an input.** |
| **Training data** | 80,000 synthetic customers from a randomised experiment (half called at random). 60/40 train/test split, stratified by called and applied. |
| **Ground truth** | Because the data is synthetic, every customer's true uplift is known, so models are scored against it as well as with the observable Qini metrics. |

## Performance (32,000 held-out customers)

| Ranking | Qini coefficient | Extra applications per 1,000 calls (top 20%) |
|---|---|---|
| Random | 0.07 | 25 |
| Propensity | 0.30 | 68 |
| S-learner | 0.55 | 108 |
| Oracle (true uplift) | 0.58 | 115 |

## Responsible use

- Uplift targeting also avoids **sleeping dogs**, customers a call puts off. Only 0.4% of the S-learner's top 20% are sleeping dogs, against 15% for propensity.
- Real use needs a randomised holdout in every campaign, so the effect can be measured and the model retrained on fresh experimental data.

## Known limitations

- Synthetic data: shows the method, not the size of the gain on a real customer base.
- One train/test split. Bootstrapped intervals are a next step.
- Call cost and capacity are not modelled.
