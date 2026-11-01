# Third optimization experiment

This run preserves every earlier model, result and confirmation sample. Its task remains binary click ranking on complete supplied candidate lists, evaluated at NDCG@10.

## Error analysis

On the existing validation sample, 51.9% of queries have no observed clicked candidate in the top ten of the strong feature-only ranker. Only 435/5,932 queries have an observed target repeated in the click prefix. Those queries average NDCG 0.6363; remaining queries average 0.2128. There are 756 known-user queries but only 114 with a past known-user click profile available under the safe cutoff.

Released-label query metadata shows a large identity-coverage shift: January/February have no missing user IDs, March is about 47% anonymous, April about 86% and May about 88%. These are changes in released identifier availability, not evidence that real shoppers changed behaviour. The chronological benchmark therefore particularly tests anonymous/cold-context recommendations.

## New learned models

Actual RecBole GRU4Rec and SASRec use full-catalog cross-entropy training instead of the earlier one-negative BPR objective. Both use the existing pre-30-April atomic interaction file and RecBole's dataset, augmentation, model, optimizer and Trainer. The same full query candidate lists determine validation checkpoints. These are training objectives over different negative information, not a change to evaluation candidates.

GRU4Rec uses embedding 64, hidden state 128, dropout 0.2, learning rate 0.003, batch 1,024 and at most eight epochs. SASRec uses hidden size 64, one attention layer, two heads, inner size 128, dropout 0.2, learning rate 0.001, batch 512 and at most ten epochs. Both use patience two. The runs used the local Apple MPS backend; CPU inference uses the saved weights. Hardware/backend differences can prevent bitwise reproducibility.

## Collaborative and identity context

A sparse binary daily-session/item matrix is constructed from past clicks. Its item Gram matrix yields co-occurrence associations, normalized by the geometric mean of session frequencies plus shrinkage 10. This is custom causal collaborative feature extraction, not a claim to run RecBole ItemKNN. The actual RecBole models are the GRU and SASRec above.

For each query, compute similarity from its existing click prefix to each supplied candidate. Add last-click, maximum, mean and recency-weighted collaborative scores. For available persistent user IDs, add features from the twenty most recent distinct past clicked products. Separately add older-date context from the supplied session ID. Persistent profiles are additional information beyond the operational daily session; an anonymous-user control removes the user-specific features while preserving session and collaborative context.

To prevent target leakage, April training graphs/profiles are frozen in weekly blocks: through 30 March, 6 April, 13 April and 20 April for query blocks 1–7, 8–14, 15–21 and 22–29 April. Validation and new model-selection context stops on 29 April. New confirmation context stops on 14 May. Every cutoff precedes the earliest prediction it serves by at least a full day. No current-query or future clicks contribute.

All models retain the existing 48 features and add fifteen contextual features. Compare 31- and 63-leaf LambdaMART models, plus a 31-leaf anonymous-user control. Model fitting uses the same 59,594 April queries, 8,747,557 candidates and 5,932 development queries as the second experiment.

Four historical clicks reference two products absent from the released product catalog. The new graphs/profiles explicitly exclude these events. A sentinel-index handling issue in the preparation code was also fixed so an unknown ID cannot increment the final catalog row. That final product appears in none of the earlier train/validation/confirmation candidate samples, so the four erroneous aggregate increments did not affect those recorded candidate features. Earlier caches/results remain unchanged.

## Fresh model selection and confirmation

Before new fitting, reserve 10,000 confirmation queries using seed 20261028, excluding every supplied session ID in either earlier inspected test/confirmation sample. Independently reserve 8,000 May development queries using seed 20261029, excluding session IDs in the earlier validation sample. Their IDs/hashes are in `config/iteration3_protocol.json`.

All epochs and candidate settings use the old development sample. Nominate the previous frozen hybrid, the best new/previous tree, the best new neural model, the best validation fusion and the anonymous control. Open the new model-selection sample once after fitting all candidates. It retains 7,921 evaluable queries. Select the highest NDCG nominee, then freeze model/config hashes. Diversity policies are tuned only on the old development sample using the selected scorer and the same 2% relative NDCG budget.

Evaluate the frozen selection once on the new confirmation sample. Report paired session-cluster bootstrap uncertainty against the previous frozen hybrid and original model on the same new queries. This is the third sample from the same broad period; it is not an independent dataset or temporal replication. Report all previous experiments and all unsuccessful models. No score target such as 0.9 justifies changing candidates, labels or test filters.

## Files and reproduction

- `src/session_rerank/iteration3.py`: reservation, causal graphs/context and tree fitting.
- `src/session_rerank/neural_search.py`: actual additional RecBole training/inference.
- `src/session_rerank/iteration3_report.py`: nominee selection, freeze, confirmation and paired comparisons.
- `config/recbole_gru_ce.yaml` and `config/recbole_sas_ce.yaml`: new RecBole settings.
- `results/iteration3/`: complete development histories, fresh selection scores and confirmation outputs.

Run commands from the repository root with the pinned RecBole environment. During source development, prefix commands with `PYTHONPATH=src` or reinstall the package after editing it. Preserve immutable recorded models/results and use a clean checkout for reproduction.

## Reproduction commands

Use the existing pinned Python 3.11 RecBole environment. From a clean source checkout:

```bash
uv venv --python 3.11 .venv-recbole
uv pip install --python .venv-recbole/bin/python -r requirements-recbole.lock.txt
uv pip install --python .venv-recbole/bin/python --no-deps .
.venv-recbole/bin/python -m session_rerank.iteration3 run
.venv-recbole/bin/python -m pytest -q
```

The command reconstructs prerequisite feature caches/reference models if missing, then trains the new candidates, selects once and confirms once. Existing recorded stages are reused; it does not overwrite frozen outcomes. Full reproduction can take substantial CPU/GPU time and additional disk space. The actual runs used local MPS for the new neural models; scores can vary slightly with hardware.

The final results are in `ITERATION3_RESULTS.md`. The README remains minimal. All local method/result notes and generated data/models remain Git-ignored.
