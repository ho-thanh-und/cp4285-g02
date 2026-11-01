# Second experiment: RecBole and stronger relevance models

Started 26 October 2026. This log records the work in progress. The first experiment's frozen models, test cache and results remain unchanged.

## Objective

Improve full-candidate query-ranking NDCG without changing the task, filtering test targets to easier examples, reducing candidate lists or leaking future interactions. Keep the relevance/diversity project question and evaluate diversification again after selecting a stronger scorer.

## Changes implemented before confirmation evaluation

- Reserved 10,000 later-date queries after excluding all supplied session IDs used in the original test sample. The selection uses seed 20261026. Its query-ID hash is in `config/optimization_protocol.json`. No new model outcome metrics were computed on this sample before selection.
- Increased the sampled April training pool from 12,000 to 60,000 queries. Every selected query retains all candidates. The positive-candidate criterion leaves 59,594 training queries and 8,747,557 candidates.
- Retained exactly the earlier validation sample: 5,932 evaluable queries, with all 890,314 candidates.
- Added time-aware click, impression, view and purchase aggregates. All use source dates at least two days before the prediction date. Validation aggregates stop on 29 April and confirmation aggregates stop on 14 May. Thus no validation or confirmation target enters its split's global statistics.
- Changed the click-rate impression denominator to released-label queries only. Hidden-label impressions do not count as observed nonclicks.
- Added one-week and one-month statistics, click trends, within-list feature ranks, product-token length, category and price/context features. There are 40 total features versus the original 12.
- Kept browsing views out of within-session history because their relative clock integration is unresolved. Their explicit event dates can still support embargoed historical popularity aggregates. Purchases outside the query observation interval are excluded from these aggregates.

## RecBole is part of the actual implementation

The project uses RecBole 1.2.1's Config, dataset creation, sequential augmentation, negative sampler, GRU4Rec model, BPR loss and Trainer. It is not just an unused dependency. Configuration is in `config/recbole_gru4rec.yaml` and implementation is in `src/session_rerank/recbole_model.py`.

The atomic training file contains 756,506 deduplicated pre-30-April click events over 261,250 daily-session pseudousers before RecBole's training-only minimum-support filters. The atomic timestamps are ordinal sequence positions after sorting original click clocks, avoiding loss of millisecond ordering when RecBole converts numeric features to float32.

RecBole handles training. A Trainer validation adapter scores exactly the same supplied query candidates as the ranking experiment. This deliberately replaces default leave-one-out/full-catalog validation, which would measure a different task. Each prefix is encoded once, and candidate embedding dot products reproduce GRU4Rec's `predict` scoring. Empty or entirely unseen histories fall back to recent popularity. Cold candidates remain in evaluation and receive a low model score when no learned item embedding exists.

Training sequences retain repeated events. Query evaluation keeps the first experiment's most recent ten distinct-click prefix definition. This training/inference difference is disclosed rather than silently changing the comparison population.

The environment `.venv-recbole` uses Python 3.11 and pinned NumPy 1.26, pandas 2.2, SciPy 1.12, PyTorch 2.5.1, Ray 2.6.3, PyArrow 14 and TensorBoard 2.18. These avoid RecBole 1.2.1's incompatibilities with NumPy 2, pandas 3 and newer serialization/dependency behavior. The original `.venv` remains intact. `requirements-recbole.lock.txt` records the working installation.

## Validation search

Compare the original frozen ranker, a tuned 12-feature model, richer features at 20,000 training queries, richer features at 60,000 training queries, and RecBole GRU4Rec. Select using validation NDCG@10. RecBole's epoch is selected by the same validation metric. If score fusion is tested, its weight must also be selected exclusively on validation.

Then choose fixed and session-aware diversification policies against the selected scorer using the same validation 2% relative NDCG loss budget. Freeze the complete selection and model hashes before opening confirmation outcomes. Report paired confidence intervals and diversity robustness, not just the largest point estimate.

## Interpretation safeguards

NDCG values from different task definitions, candidate sets, target filters or cutoff values are not directly comparable. The original competition paper reports a different weighted query-full/query-less protocol. We will not claim to reach its reported score by comparing our NDCG@10 directly with its leaderboard number.

The confirmation sample shares the broader date period of the earlier inspected test. It is a newly reserved sample with disjoint supplied session IDs, not an independent dataset or a new temporal replication. The first experiment informed the development direction, and this should be disclosed.

## Primary sources

- RecBole sequential quick start: https://recbole.io/docs/get_started/started/sequential.html
- RecBole GRU4Rec configuration and loss: https://recbole.io/docs/user_guide/model/sequential/gru4rec.html
- Official GRU4Rec implementation: https://github.com/RUCAIBox/RecBole/blob/master/recbole/model/sequential_recommender/gru4rec.py
- Competition solution motivating historical, time-based and session features: https://arxiv.org/abs/1708.04479

The runs are complete. See `OPTIMIZATION_RESULTS.md` for every measured result and the interpretation. No model or policy was retuned after confirmation.

## Completed additions during validation development

The validation search added an optional eight-feature prior-view family (48 features total) and 63-leaf tree variants. Same-session/date strict view timestamps reproduce approximately 37.6% validation coverage. These models retain a shared relative-clock assumption, and no-view controls remain in the comparison. Seven new tree variants were evaluated alongside the original model and RecBole GRU4Rec. The GRU trained for 12 epochs and selected human-numbered epoch 7, with 479,280 training-prefix examples and 44,842 item tokens including padding. Gated score fusion selected 75% tree and 25% RecBole on validation. See the result report for its failure to improve confirmation over the tree alone.
