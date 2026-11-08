# Ethical analysis

Draft for the CP4285 report. Measured results come from
`uv run main.py exposure` (config `config/exposure.json`, code
`src/session_rerank/exposure.py`); experiment details are in
`EXPERIMENT_LOG.md` §6k.

## 1. Who is affected

| Stakeholder | What the recommender changes for them | Main risk |
|---|---|---|
| Shoppers | Which products they see first on the home page, category pages and "You may also like" | Narrow, popularity-driven choice; a feeling of being tracked |
| Sellers / brands, especially of niche products | How often their products are shown at all | Products with few past clicks are rarely or never shown (a rich-get-richer loop) |
| The shop operator | Clicks and sales, and the data the site collects | Optimising short-term clicks at the cost of catalog breadth and trust |
| Visitors of our demo website | Their own interaction logs | Collection of behaviour data beyond what the demo needs |

## 2. Popularity bias and exposure concentration (measured)

April clicks are very concentrated. Of 184,047 products, **3,761 (2%) receive
half of all April clicks** (the "head"), 41,754 receive the other half (the
"tail"), and **138,532 (75%) received no click at all**.

We replayed every evaluated method on the selection week (May 9–14). That is
20,000 home-page cases, 20,000 item pages and 966 category-page queries, all
ten slots counted equally. The table compares the share of shown slots taken
by head products with the head share of the products shoppers actually
clicked next. A ratio above 1 means the method concentrates on popular
products more than shoppers' own choices do.

| Page | Method | Distinct products shown | Head share | No-click products | Ratio to shoppers |
|---|---|---:|---:|---:|---:|
| Home | Popularity (cold start) | 12 | 100% | 0% | 3.32 |
| Home | Item-kNN | 27,461 | 32.4% | 0% | 1.07 |
| Home | Item-kNN + fixed 0.8 (previous home page) | 27,431 | 31.3% | 0% | 1.04 |
| Home | Two-tower | 27,670 | 39.1% | 12.5% | 1.30 |
| Home | SASRec, relevance only | 26,199 | 39.7% | 5.4% | 1.32 |
| Home | **SASRec + 6 + 4 (frozen)** | 26,652 | **39.1%** | 5.6% | **1.30** |
| Home | Shoppers' later clicks | 14,095 | 30.1% | 14.1% | 1.00 |
| Item | Category popularity | 8,012 | 72.8% | 0.8% | 2.42 |
| Item | Item-kNN, relevance only | 32,070 | 30.2% | 0.3% | 1.00 |
| Item | **Item-kNN + fixed 0.8 (frozen)** | 32,296 | **28.9%** | 0.4% | **0.96** |
| Item | Shoppers' later clicks | 14,095 | 30.1% | 14.1% | 1.00 |
| Category | Site's original order | 2,818 | 24.9% | 24.7% | 0.62 |
| Category | Popularity sort | 2,796 | 72.2% | 0.8% | 1.80 |
| Category | Item-kNN | 3,961 | 62.7% | 0.8% | 1.57 |
| Category | **LightGBM purchase_session (frozen)** | 3,349 | **68.1%** | 2.1% | **1.70** |
| Category | Shoppers' clicks | 1,357 | 40.0% | 10.9% | 1.00 |

What this shows:

- **Popularity is the most concentrated by far.** Cold-start popularity on the
  home page shows the same 12 products to every visitor. Category popularity
  on the item page puts 73% of slots on the 2% head. This is the cost of
  cold start and the reason popularity is used only until the first click.
- **Item-kNN lists match shoppers' own concentration.** On the item page and
  the previous home page they put 29–31% of slots on head products, about the
  same as shoppers' clicks (30%). They show 27,000–32,000 different products in
  a week, roughly twice as many as shoppers clicked.
- **The new home page (SASRec + 6 + 4) trades exposure for accuracy.** It is
  16% more accurate than item-kNN + fixed 0.8 on the selection week (NDCG
  0.118 vs 0.102), but it puts 39% of slots on the 2% head (1.3 times
  shoppers' share, against 1.04), and shows slightly fewer distinct products
  (26,652 vs 27,431). Like the category-page ranker, the more accurate model
  leans more on popular products. It does reach some products with no April
  clicks (5.6% of slots), which it learned from February and March sessions.
- **The diversity reranking helps only a little.** Fixed alpha 0.8 lowers the
  head share by about one point on both pages and barely changes how many
  products are shown. It diversifies names and categories within one list,
  which is a different goal from spreading exposure across the catalog.
- **Products without past clicks are hard to reach.** 14% of what shoppers
  clicked next had no April clicks. Item-kNN cannot recommend those products,
  because it needs co-clicks. SASRec shows them only if they were clicked in
  February or March (5.6% of slots). The two-tower gives them 12.5% of
  home-page slots, using name, category and price.
- **On category pages, accuracy and exposure pull in opposite directions.**
  The site's own order is the most long-tail (25% head, 25% no-click), but
  also the least accurate (click NDCG@10 0.065). The frozen LightGBM ranker
  is the most accurate (0.203 against 0.174 for popularity sort), but it
  moves exposure to popular products: 68%
  head, 1.7 times shoppers' share. Products without April clicks fall from a
  quarter of the first ten slots to 2%. The ranker learned that popularity
  predicts clicks, and that learning reinforces popularity.

Gini coefficients are close to 1 for every method (0.92–1.00 over the whole
catalog), because even 32,000 shown products are a small part of 184,000.
Coverage and head share separate the methods more clearly, so the report uses
those.

Possible mitigations, not yet evaluated:

- give a fixed share of category-page slots to products with few clicks (an
  exploration slot);
- cap the popularity features' influence for products with little history;
- use the two-tower's content signal to fill the item-kNN pool when a
  product has few co-click neighbours;
- widen SASRec's home-page pool beyond 20 products, or reserve one of the
  four diverse slots for a tail product.

## 3. Feedback loops

Every model here learns from clicks on lists the site itself chose to show.
A product the site never showed cannot collect clicks, so it stays unpopular
in the next training round, and a recommender trained on popularity or
co-clicks shows it even less. This is the mechanism behind the concentration
measured in §2. In the website we would counter it in three ways:

- keep the diverse tail (positions 7–10 of the home page and "You may also
  like"), which shows products the relevance ranking would not;
- monitor coverage, Gini and the head share of what is actually shown, from
  the website's own logs, alongside click-through rate;
- retrain on logged data with the position of each shown product recorded,
  so a later model can correct for where a product was shown.

## 4. Biases in the training data

- **Position bias.** In the logged category pages, rank 1 has a 3.1% click
  rate and ranks 51–100 have 1.0% (`EXPERIMENT_LOG.md` §2). The fitted curve,
  click rate ∝ rank^−0.30, is flat compared with web search, but products
  shown high in the original list were still clicked more partly because of
  where they were. Labels learned from these clicks reward the site's old
  ordering as well as the product.
- **Selection of sessions.** The dataset contains only sessions with at least
  one click, so shoppers who left without clicking are invisible. The
  evaluation says nothing about them.
- **Products that were never shown cannot be judged.** Offline NDCG counts a
  recommended product that the shopper never saw as a miss. This penalises
  exactly the long-tail products that diversity is meant to surface, so the
  offline cost of diversity is an upper bound.
- **Clicks are not satisfaction.** Purchases are too rare in the evaluation
  weeks to optimise for, so every model optimises clicks, which can reward
  attention-grabbing products rather than useful ones.

## 5. Privacy

- **Dataset.** Diginetica (CIKM Cup 2016) is released for research with
  hashed product, category and session identifiers, and hashed name tokens.
  Our design uses no user IDs at all: every recommendation depends only on
  the current anonymous session.
- **Demo website logging.** The website will log interactions for evaluation.
  We commit to:
  - a random session identifier per visit, with no account, cookie-based
    cross-visit tracking or fingerprinting;
  - no IP addresses, device details or free text stored with the logs;
  - only the fields evaluation needs (session ID, page, shown products and
    their positions, clicked product, time);
  - a visible notice that interactions are recorded for a course project,
    with the option to browse without logging;
  - deletion of the logs after grading.
- **Inference risk.** Even an anonymous session reveals interests. Session
  data is used only within that session and is not joined across visits.

## 6. Transparency

- Lists will be labelled ("Popular now", "You may also like"), so a shopper knows
  the page is a recommendation rather than a neutral catalog order.
- The website will not mix sponsored placement into these lists.
- Possible addition: a short reason per product ("because you viewed …"),
  which item-kNN supports directly.

## 7. AI-generated product names in the demo

The dataset's product names are hashed tokens, so the demo shows names
generated by an AI model. These names are fictional:

- the website states that product names and images are generated for the
  demo and do not describe real products;
- generated names must not use real brand names or imitate a real seller;
- names are checked for offensive or stereotyped wording before use;
- every reported number is computed on the original hashed data, never on the
  generated names.

## 8. Limitations of this analysis

- There is no seller or brand field, so fairness between sellers can only be
  approximated by exposure across products.
- Exposure is measured on offline replays of one week (selection week, May
  9–14). Real exposure depends on how shoppers navigate the website.
- Each of the 10 slots counts the same; on the website, position 1 receives
  more attention than position 10.
