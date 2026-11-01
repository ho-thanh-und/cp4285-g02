# STePS presentation storyline

Working title: **Does session diversity improve product re-ranking?**

This is slide copy and speaker guidance for the final presentation, grounded in the first recorded experiment. It is not an exported PowerPoint deck. Keep results labeled as this experiment until the team completes the checks in NEXT_STEPS.md. Adjust the slide count to the actual presentation duration.

## A 45–60 second explanation for visitors

“Online shops need to choose between showing close alternatives and showing a broader range of products. We tested whether the products someone has already clicked can help choose that balance. In Diginetica, 90% of candidate lists already contain only one category, so we measured variety through the anonymized product-name tokens. We built a relevance ranker, then compared a fixed diversification weight with session-aware rules. On our later-date test, a fixed rule increased token variety by about 1.9% at a 1.3% reduction in ranking quality. The tested session-diversity rule did not improve on it. Only about 28% of test queries had enough visible history to estimate session diversity. Our result shows why simple baselines and the available session information matter before adding personalization.”

For a specialist, replace “ranking quality” with “binary click-based NDCG@10.” For a general visitor, explain that the measure rewards clicked products appearing near the top.

## Slide 1: Does session diversity improve product re-ranking?

On slide:

- Diginetica e-commerce search candidates
- Nguyen Dinh Nam, R Arulkumaran, Ho Chi Thanh
- CP4285 Group 2, STePS 2026

Speaker: Introduce the decision the project studies. Do not promise that personalization wins or that the experiment improves sales.

## Slide 2: Most candidate lists offer variety within one category

On slide:

- **90.23%** of 923,127 lists contain one category
- Category count cannot reveal most changes within those lists
- We test variety using product-name token overlap

Evidence: `results/data_audit.json`, full original archive.

Speaker: This is the specific dataset finding motivating the project. Anonymized tokens preserve overlap but do not reveal readable product names or prove useful semantic variety. Avoid a generic filter-bubble claim.

## Slide 3: The research question

On slide:

“Can session diversity improve the relevance–variety balance beyond a tuned fixed rule?”

- Compare with the same relevance model and candidates
- Test whether diversity helps beyond history length alone
- Select policies with a 2% relative NDCG loss budget on validation

Speaker: The two hypotheses are that session-conditioned diversification improves on a fixed weight, and that diversity adds information beyond simply having a longer history. A negative result is a valid answer. Starting broad and becoming focused is a secondary heuristic, not an established premise.

## Slide 4: Information available at each query

On slide:

- Daily session = supplied session ID and event date
- History = up to ten distinct clicks strictly before the query
- Current-query clicks supply labels only
- Chronological train, validation and test with one-day gaps

Supporting annotation: Statistics through 30 March. Train 1–29 April. Validation 1–14 May. Test 16 May–1 June, all 2016.

Speaker: We excluded browsing views from this first model because their timing needs further verification. A minimum history threshold of two or more distinct clicks differs from an inactivity timeout. The test contains 9,869 evaluable sampled queries, not all 923,127 original queries.

## Slide 5: Simple controls test the value of adaptation

On slide, a compact editable table in the eventual deck:

| Method | What changes? |
|---|---|
| Relevance ranker | Learned click-based relevance |
| Fixed re-ranking | One validation-selected weight |
| History gate | Diversify only with enough history |
| Diversity threshold | Choose weight using session diversity |

Small formula: `score = weight × relevance + (1 − weight) × variety`

Speaker: All re-rankers share the same LambdaMART scorer and candidates. The first product remains the relevance winner. Also evaluate popularity, repeat-click and progression rules, but keep their full details for questions. The paper inspired the adaptation, while its original SR/SKNN method differs from ours.

## Slide 6: Fixed re-ranking improves token variety at a measured relevance cost

Main visual: `results/figures/test_tradeoff.png`. The plot zooms near the relevance baseline. Its axis does not begin at zero. Keep the dashed reference explained as 2% relative loss against the test relevance score, not a threshold fitted on test.

On slide:

| Test method | NDCG@10 | Token ILD@10 |
|---|---:|---:|
| Relevance ranker | 0.19463 | 0.89664 |
| Fixed re-ranking | 0.19209 | 0.91377 |
| Diversity policy | 0.19219 | 0.90978 |

Callout: Fixed versus relevance: **+1.91% ILD, −1.31% NDCG**.

Speaker: These are relative changes. The corresponding absolute changes are +0.01714 and −0.00255. Paired 95% bootstrap intervals for the relative changes are +1.84% to +1.98% ILD and −1.78% to −0.81% NDCG. Whether that exchange is worthwhile depends on the application. Do not call 0.19463 “19.5% accuracy.”

## Slide 7: The diversity policy reduces to a history-length rule

Main visual: `results/figures/test_history.png`.

On slide:

- **27.7%** of test queries have at least two prior distinct clicked products
- Validation selects weight **0.6 in both diversity branches**
- The diversity threshold therefore makes no difference

Speaker: The adaptive policy and history gate produce identical recommendations. Against fixed re-ranking, NDCG differs by +0.000105, with a paired interval spanning zero. ILD is lower by 0.003997. We found no benefit from this tested diversity feature/rule. Sparse history is a plausible constraint, but we have not established it as the causal explanation. Do not describe the shuffled control as a separate success: it is identical because the selected threshold is inactive.

## Slide 8: What the experiment establishes

On slide:

- Fixed diversification offers a measured lexical variety gain
- The tested session-diversity rule adds no benefit over simpler controls
- Conclusions concern recorded clicks and anonymized token variety

Speaker: We cannot claim that shoppers become more satisfied, that sales rise, or that filter bubbles shrink. Our contribution is a reproducible comparison that identifies when an appealing personalization idea lacks supporting evidence in the available data. A stronger future method needs reliable additional information and fresh confirmation.

## Backup slide: Metric definitions

- NDCG@10: rank-discounted recovery of distinct clicked candidates, normalized by the ideal order
- Recall@10: clicked candidates recovered in the first ten divided by all distinct clicked candidates
- ILD@10: mean Jaccard distance between the selected products' name-token sets
- “+1.91%” means `(new − baseline) / baseline × 100`, not +1.91 absolute ILD units

Queries with fewer than ten candidates use their available list. All methods evaluate the same queries and full candidate sets. The table in `results/test_summary.csv` contains Recall as a supporting measure. Do not add a long list of novelty, coverage and ranking metrics to the main story.

## Backup slide: Study limitations

- Binary clicks reflect the original exposure policy and position bias
- Hidden-label queries limit observable history
- Daily sessions and click-only prefixes are deliberate operational choices
- Token distance does not independently validate meaningful diversity
- One chronological split and a sample of released-label queries limit generalization

Evidence source: `docs/EXPERIMENT.md`. Explain the implication of each limit rather than reading a checklist.

## Backup slide: References and attribution

- Diginetica, CIKM Cup 2016 Track 2, original released dataset
- Gharahighehi and Vens (2021), Personalizing Diversity Versus Accuracy in Session-Based Recommender Systems, DOI 10.1007/s42979-020-00399-2
- Yin, Fang, Sun and Ong, Understanding Diversity in Session-Based Recommendation, DOI 10.1145/3600226
- Wu, Yan and Si (2017), Ensemble Methods for Personalized E-Commerce Search Challenge at CIKM Cup 2016, arXiv:1708.04479
- Laktionov and Grozin, original competition sample baseline, provenance only

Use DOI/URL links in the final speaker notes. Distinguish the literature's findings from this team's experiment. Explain AI assistance accurately if asked and follow any course disclosure requirements that the team subsequently confirms.

## Likely questions

**Why not use category diversity?** It is constant for 90.23% of candidate lists. Tokens provide within-category variation, although their interpretation is limited.

**What does anonymous mean here?** Missing user ID. A session ID still exists, and the benchmark uses local session information for every query regardless of whether a user ID is present.

**Why two clicks?** Pairwise diversity needs two distinct items. We also searched minimums of three and five using validation, not test.

**Why not a transformer?** The current question concerns the added value of diversification policy given a common ranker. A larger relevance model would be a separate experiment. Its complexity alone would not establish that session-diversity adaptation works.

**Is the adaptive policy better?** Not in this experiment. It becomes a simpler gate, and fixed re-ranking yields higher diversity at similar NDCG.

**Does the user actually like the more diverse list?** This offline dataset cannot answer that. A controlled online study or another valid feedback design would be needed.

**Is the 2% budget justified?** It is a predeclared experimental tolerance, not a measured business preference. The trade-off curve lets a reader see alternatives.

**Could a different session definition change the result?** Yes. That is a high-value sensitivity check, and it must use development data or a fresh confirmation protocol.

**Can you claim interests narrow over a session?** No. The progression policy is a testable heuristic. The current study does not establish a causal change in shopper intent over time.

**What did each person do?** Describe the team's actual reviewed contributions. The suggested ownership in NEXT_STEPS.md is a plan, not a record of completed student work.
