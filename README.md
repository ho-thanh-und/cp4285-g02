# Session-Aware Product Re-ranking

## Methodology

Re-rank Diginetica's complete candidate lists using information available before
each query. Daily sessions provide up to ten distinct prior clicked products.
Historical statistics and collaborative item associations use earlier data with
an embargo; persistent-user context has an anonymous-user control.

Train temporal/collaborative LambdaMART models and **RecBole GRU4Rec and SASRec**.
Compare BPR and cross-entropy configurations. Nominate models on development data,
choose on a fresh model-selection sample, then freeze before confirmation.
The selected hybrid uses 90% normalized tree score and 10% GRU4Rec/CE score
when click history exists, otherwise the tree score.

Greedy re-ranking balances relevance and token Jaccard distance. Select fixed
and session-adaptive weights within a 2% validation NDCG loss budget.

## Results

All rows use the same **9,865 newly reserved confirmation queries and 1,492,509
candidates**. Session IDs exclude both earlier test/confirmation samples.

| Method | NDCG@10 | Token ILD@10 |
|---|---:|---:|
| Original ranker | 0.18261 | 0.90329 |
| Previous hybrid | 0.22412 | 0.90878 |
| RecBole SASRec/CE | 0.20369 | 0.91374 |
| Selected collaborative hybrid | 0.23116 | 0.90887 |
| Hybrid + fixed diversification | 0.22970 | 0.93312 |
| Hybrid + adaptive diversification | 0.22998 | 0.91581 |

The selected model improves NDCG **3.1%** over the previous hybrid
(paired 95% interval: **1.9–4.3%**). Its component tree scores 0.23146,
so fusion adds no demonstrated confirmation benefit. Fixed diversification adds
**2.7% token variety** for a **0.6% NDCG reduction**. The adaptive policy again
selects equal diversity-branch weights and becomes a history-length gate.

These offline results measure recorded clicks and lexical variety, not shopper
satisfaction or sales.
