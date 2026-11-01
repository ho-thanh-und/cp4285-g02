# Session-Aware Product Re-ranking

## Methodology

Re-rank Diginetica's complete product candidate lists using only information
available before each query. A session is a supplied session ID within one date,
with up to ten distinct prior clicked products.

Train a temporal-feature LambdaMART ranker and **RecBole GRU4Rec** with BPR loss.
Use chronological training, validation and confirmation periods, with a one-day
embargo on historical statistics. Select models and policies on validation only.
The selected hybrid combines 75% normalized tree score and 25% RecBole score
when click history exists; otherwise, it uses the tree ranking.

Greedy re-ranking balances relevance with product-name token Jaccard distance.
Compare a fixed weight with session-adaptive weights, selected to maximize
validation diversity within a 2% relative NDCG loss budget.

## Results

Confirmation uses **9,883 queries and 1,481,803 candidates**, with session IDs
excluded from the earlier test sample. Higher NDCG@10 means better click-based
ranking; higher ILD@10 means greater token variety.

| Method | NDCG@10 | ILD@10 |
|---|---:|---:|
| Original ranker | 0.19066 | 0.89644 |
| RecBole GRU4Rec | 0.18066 | 0.91684 |
| Feature-only component | 0.22676 | 0.89997 |
| Validation-selected hybrid | 0.22476 | 0.90278 |
| Hybrid + fixed diversification | 0.22261 | 0.92974 |
| Hybrid + adaptive diversification | 0.22443 | 0.91378 |

The selected hybrid improves NDCG by **17.9%** over the original ranker
(paired 95% confidence interval: **15.6–20.9%**). Fusion's validation gain does
not generalize over its feature-only component. Fixed diversification adds
**3.0% token diversity** for a **1.0% NDCG reduction**. The adaptive policy
selects equal branch weights, adding no value from the diversity threshold.

These offline results measure recorded clicks and lexical variety, not shopper
satisfaction or sales.
