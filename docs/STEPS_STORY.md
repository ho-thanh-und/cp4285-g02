# Current STePS storyline after optimization

Use this version with `OPTIMIZATION_RESULTS.md`. The initial storyline is preserved in `STEPS_STORY_V1.md`. This is slide copy and speaker guidance, not a designed PowerPoint deck.

Working title: **When does session information improve product re-ranking?**

## Slide 1: The decision

How much product variety should an e-commerce ranking introduce without losing relevance?

Introduce Group 2 and the Diginetica query-candidate task. Present relevance/diversity balance as a question to investigate rather than promising an adaptive system must win.

## Slide 2: The dataset finding

90.23% of the original 923,127 candidate lists contain one category. Category variety therefore cannot distinguish most re-rankings. We measure variation in anonymized product-name token sets instead.

Explain that 62.28% of queries lack user IDs, but supplied session IDs still exist. Avoid equating anonymity with having no session information.

## Slide 3: The question and comparison

Does a session-diversity rule improve on a tuned fixed diversification weight, using the same strong relevance scorer?

Compare a fixed rule, a history-length gate and a diversity threshold. A secondary engineering investigation asks whether recent information and a RecBole sequence model strengthen the relevance baseline. These questions connect optimization to the original project rather than replacing it with a leaderboard chase.

## Slide 4: Information available at prediction

Daily session = supplied session ID plus date. The click prefix contains at most ten distinct products strictly before the query. Current-query clicks supply labels only.

Global training statistics use a one-day embargo. Validation statistics stop on 29 April and confirmation statistics on 14 May 2016. Train on April queries, select with 1–14 May queries, and confirm on reserved 16 May–1 June queries. Keep whole candidate lists. Describe prior-view features as a separate assumption-dependent family with a no-view control.

## Slide 5: RecBole and model ablations

RecBole GRU4Rec trains on click sequences with BPR loss. RecBole supplies the dataset, augmentation, sampler, model, optimizer and Trainer. A validation adapter scores the original complete candidate lists.

Compare the original ranker with a tuned 12-feature model, 40-feature temporal models, optional 48-feature view models, and the RecBole model. Selection chooses a hybrid with tree weight 0.75 when click history exists. Keep the full hyperparameter table for questions.

## Slide 6: A stronger relevance baseline

Main figure: `results/optimization/figures/confirmation_relevance.png`.

The frozen hybrid scores **0.22476 NDCG@10** against **0.19066** for the original model on the same 9,883 confirmation queries. That is **+17.89% relative**, with paired 95% interval **+15.56% to +20.87%**.

Explain the unsuccessful result too: RecBole alone scores 0.18066. Fusion helps validation but does not beat its component tree on confirmation. The 40-feature no-view control scores 0.22696. The benefit of richer information is clearer than a benefit from extra neural complexity. The frozen hybrid remains the primary selected result.

## Slide 7: Fixed versus adaptive diversification

Main figure: `results/optimization/figures/confirmation_diversity.png`.

Fixed weight 0.7 yields **+2.99% Jaccard ILD** at **-0.95% NDCG** relative to the selected relevance scorer. Cosine ILD rises **+4.56%**, a numerical distance robustness check.

The adaptive search again chooses 0.6 for both low- and high-diversity prefixes. It equals the history-length gate. It preserves slightly more relevance and yields less diversity than the fixed policy. The threshold itself adds no value in this experiment. Do not call the fixed policy universally better when the policies offer different trade-offs.

## Slide 8: What the study establishes

Stronger temporal/context features improve ranking relative to the initial benchmark. RecBole provides a real sequence-model comparison, but its extra complexity does not automatically generalize better. Simple fixed diversification provides a measurable lexical-variety gain. The tested session-diversity threshold adds no demonstrated benefit beyond a gate.

Results concern observed clicks and anonymized tokens. They do not establish shopper satisfaction, sales, causal intent changes or independently validated semantic variety. The team should reproduce the results and be able to explain every choice before presenting.

## Short visitor explanation

“We studied whether a shopping session helps choose how varied product recommendations should be. We built a stronger ranking baseline and compared it with a RecBole sequence model. On a reserved sample, the selected ranking improved our ranking metric by about 18% over the first version. A simple diversification rule then increased product-token variety by about 3% for about a 1% ranking-quality loss. The tested session-diversity rule did not add value beyond a simpler history rule. This shows why recent information and fair baselines matter before adding more personalization.”

## Questions to prepare for

**Why is NDCG about 0.22?** We keep the entire supplied candidate set. Random ranking gives 0.07541 on the same queries, and performance varies with list length. NDCG is not a percentage accuracy. Using shorter sampled lists could produce a higher number while measuring a different task.

**Did RecBole improve the result?** It helped validation fusion slightly, but that advantage did not carry over to confirmation. We report both outcomes rather than attributing the overall gain to RecBole.

**Why keep a model that is not the highest on confirmation?** We selected before opening confirmation. Replacing it with whichever model wins on confirmation would turn confirmation into another tuning set.

**What is the session threshold?** The daily session boundary, the query-time summarization point and the minimum number of distinct clicks are separate definitions. The diversity rule needs at least two distinct products and searches 2, 3 and 5 on validation.

**Can a different session or token representation change the result?** Yes. The no-view control and cosine distance check cover some assumptions, but broader semantic/session sensitivity remains a limitation rather than an untested claim of robustness.

**Are the samples independent of the earlier test?** Supplied session IDs are disjoint, but the new confirmation sample uses the same broader dates and dataset. It is not an independent dataset or temporal replication.

**Where are the references?** Cite Diginetica's original competition, Gharahighehi and Vens (2021), the official RecBole documentation and paper, Wu, Yan and Si (2017), and Yin et al. See the README URLs. Distinguish published findings from this team's measurements.
