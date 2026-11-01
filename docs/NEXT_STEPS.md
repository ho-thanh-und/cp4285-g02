# Current status after the RecBole optimization run

Completed: full original-data audit, two recorded experiments, actual RecBole GRU4Rec training, seven new tree-model ablations, validation-only fusion/policy selection, reserved confirmation, paired confidence intervals and two lexical-distance evaluations. Current results and limitations are in `OPTIMIZATION_RESULTS.md`; the current presentation outline is `STEPS_STORY.md`.

The research code and experiment evidence are ready for team review and reproduction. The designed final deck and rehearsal remain. Do not tune against the now-inspected confirmation results. Any additional optimization needs a newly reserved confirmation plan or must be explicitly exploratory.

# Preparation for 11 November 2026 STePS

Dates below use Singapore time. This is a proposed team plan, not a claim of agreed individual contributions.

| Date | Work | Suggested lead | Reviewable outcome |
|---|---|---|---|
| 26 October | Original data, audit, first chronological experiment | Whole team | Current repository and recorded results |
| 27–28 October | Independent reproduction and code walkthrough | Thanh | Reproduced metrics and questions about labels/history |
| 28–30 October | Session-definition and token-distance sensitivity on development data | Arul | Bounded sensitivity analysis with stated assumptions |
| 30–31 October | Check interpretation and primary literature, including timing | Nam | Correct citations and defensible claims |
| 1 November | Agree final research question and experiment scope | Whole team | Frozen storyline and any new confirmation protocol |
| 2–3 November | Inspect grounded ranking examples | Arul + Thanh | Explainable item/token example, with hidden identities acknowledged |
| 4–5 November | Assemble final slide deck/poster using recorded evidence | Whole team | Presentation with a clear question and answer |
| 6–7 November | Rehearse explanation and technical questions | Whole team | Every member can explain the split, policy and limitations |
| 8 November | Content freeze | Whole team | Checked figures and references |
| 9–10 November | Rehearse, prepare offline backups | Whole team | PDF/PPTX backups and recorded example |
| 11 November | STePS presentation | Whole team | Present findings and limitations clearly |

## Highest-value follow-ups

1. Confirm the session definition. The daily interpretation is conservative but truncates cross-date histories. Compare on development data before changing the main claim. Do not choose a definition because it makes the test score look better.
2. Check whether name-token distance measures meaningful variation. Both the optimizer and primary diversity metric use Jaccard, so a larger ILD alone is expected. A different token metric can test numerical robustness, but anonymized tokens still limit semantic validation.
3. Preserve the simple controls. The current adaptive policy equals the history gate. A future policy must beat that gate as well as a tuned fixed weight to justify using a diversity feature.
4. Choose any new confirmation sample before further optimization. The current test has been inspected. Keep its results as the first experiment and disclose subsequent tuning as exploratory unless supported by fresh confirmation.
5. Prepare a clear grounded example. Use actual anonymized product IDs and token overlaps. A shopper-facing illustration with readable shoes/jackets can be used only if visibly marked as an illustrative example, not a recovered dataset record.

## Scope decisions

GRU4Rec is now implemented through RecBole and included in the experiment. Keep additional transformers, synthetic augmentation and a learned controller outside the main experiment unless they resolve a clearly stated question. They add work and interpretation risk without resolving the central question exposed by the current results. The current LambdaMART ranker gives a competitive practical baseline within this repository, but it has not been established as state of the art.

The current test is a random sample from the later-date released-label pool. Larger evaluation with the same frozen policies can improve precision, but should be labeled a prespecified confirmation extension rather than repeatedly searched for a favorable result. Statistical uncertainty and practical usefulness must both appear in the final explanation.
