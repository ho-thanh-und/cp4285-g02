"""Home page: session-to-product recommendations over the whole catalog.

Product rules (EXPERIMENT_LOG.md §7, §8.1): no hard filter (any product can be
recommended), popularity for cold start, a two-tower model after interaction,
clicks only, and no penalty for products that were not clicked. Cold start
(no clicks yet) is plain popularity and can only be judged online.

Session two-tower. Training data: every eligible April session (no hidden-label
query) in the raw files, with its distinct clicked products in order of first
click. Each prefix of the sequence (its last `history_items` products) is a
"session so far" whose target is the next product clicked. The product tower
embeds name tokens, category, price band and leak-safe popularity (daily click
and purchase rates before the session's day). The session tower combines the
mean and the last of the clicked products' vectors with the session length.
Loss: sampled softmax of the next product against uniform and popularity-
weighted random products. The epoch is chosen on a fixed subset of the
validation week and the model is saved, so a selection run reuses it exactly.

Offline evaluation (a proxy; the website's logs are the real measure). In each
eligible session of one week, every point after at least one click with a later
click on a different product is a case. Recommend 10 products from the whole
catalog, excluding products already clicked; hits are the later clicks.
Methods on identical cases: popularity, trending (last days of the training
window), item-kNN on the session's clicks, and the two-tower; then relevance
only, fixed alpha and 6 + 4 over a pool from each listed base method.
Only the validation and selection weeks can be opened; the selection week needs
a frozen run and policy, written to frozen_choices.json before reading it.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .adaptive_policies import diverse_tail_order
from .core import greedy_rerank
from .diversity import ProductDiversity
from .event_ablation import session_bootstrap
from .item_knn import cosine_similarity, read_knn_settings, session_baskets
from .ranker_phase2 import window_activity
from .study_files import save_table, write_json
from .two_tower import _token_matrix, popularity_features
from .ymal import ALLOWED_WINDOWS, DIVERSITY, RELEVANCE, list_metrics


def read_home_settings(path):
    settings = json.loads(Path(path).read_text(encoding="utf-8"))
    window = settings["window"]
    if ALLOWED_WINDOWS.get(window["name"]) != (window["start"], window["end_exclusive"]):
        raise ValueError(f"window must be one of {ALLOWED_WINDOWS}; confirmation stays sealed")
    for name in ("history_items", "embedding_dim", "hidden_dim", "epochs", "batch_size", "max_cases",
                 "epoch_check_cases", "trending_days", "threads", "max_name_tokens"):
        if not isinstance(settings[name], int) or settings[name] < 1:
            raise ValueError(f"{name} must be a positive integer")
    if settings["uniform_negatives"] + settings["popular_negatives"] < 1:
        raise ValueError("Use at least one negative product per example")
    if not 0 < settings["relevance_slots"] < settings["top_k"] <= settings["pool_size"]:
        raise ValueError("Need 0 < relevance_slots < top_k <= pool_size")
    if set(settings["diversity_bases"]) - {"two_tower", "item_knn"}:
        raise ValueError("diversity_bases may contain two_tower and item_knn")
    if window["name"] == "selection":
        names = {"popularity", "trending", "item_knn", "two_tower"} | {
            f"{kind}_{base}" for base in settings["diversity_bases"]
            for kind in ["six_four"] + [f"fixed_{a:g}" for a in settings["fixed_alphas"]]}
        if settings.get("frozen_policy") not in names or not settings.get("frozen_run"):
            raise ValueError(f"The selection week needs frozen_run and frozen_policy, one of {sorted(names)}")
    return settings


# -------------------------------------------------------------------- data ---

def click_sequences(raw_directory, start, end_exclusive, item_index):
    """Eligible sessions of a window: (session key, day, distinct clicked rows by first click)."""
    raw = Path(raw_directory)
    queries = pd.read_csv(raw / "train-queries.csv", sep=";", usecols=["queryId", "sessionId", "eventdate", "is.test"])
    queries = queries[(queries.eventdate >= start) & (queries.eventdate < end_exclusive)]
    queries = queries[~queries.groupby(["sessionId", "eventdate"])["is.test"].transform("any")]
    clicks = pd.read_csv(raw / "train-clicks.csv", sep=";").merge(queries, on="queryId")
    clicks = clicks[clicks.itemId.isin(item_index.index)]
    first = (clicks.groupby(["sessionId", "eventdate", "itemId"], sort=False).timeframe.min().reset_index()
             .sort_values(["sessionId", "eventdate", "timeframe", "itemId"]))
    first["row"] = item_index.loc[first.itemId].to_numpy()
    return [(f"{sid}|{date}", date, group.row.tolist())
            for (sid, date), group in first.groupby(["sessionId", "eventdate"], sort=False)]


def training_examples(sequences, start, history):
    """(history rows padded with -1, target row, day offset) for every next click."""
    origin = pd.Timestamp(start)
    rows, targets, days = [], [], []
    for _, date, items in sequences:
        day = (pd.Timestamp(date) - origin).days
        for t in range(1, len(items)):
            prefix = items[:t][-history:]
            rows.append(prefix + [-1] * (history - len(prefix)))
            targets.append(items[t])
            days.append(day)
    return np.asarray(rows, dtype=np.int64), np.asarray(targets, dtype=np.int64), np.asarray(days, dtype=np.int64)


def evaluation_cases(sequences, max_cases, seed):
    """(session key, clicked prefix, later distinct clicks) for every evaluable point."""
    cases = []
    for key, _, items in sequences:
        for t in range(1, len(items)):
            cases.append((key, items[:t], set(items[t:])))
    total = len(cases)
    if total > max_cases:
        keep = np.sort(np.random.default_rng(seed).choice(total, max_cases, replace=False))
        cases = [cases[i] for i in keep]
    return cases, total


# ------------------------------------------------------------------- model ---

def build_session_two_tower(names, categories, prices, popularity, settings):
    """Product tower and session tower sharing the product embeddings."""
    import torch
    from torch import nn
    import torch.nn.functional as F

    dim, hidden = settings["embedding_dim"], settings["hidden_dim"]
    names, categories = torch.as_tensor(names), torch.as_tensor(categories)
    prices, popularity = torch.as_tensor(prices), torch.as_tensor(popularity)

    class SessionTwoTower(nn.Module):
        def __init__(self):
            super().__init__()
            self.tokens = nn.Embedding(int(names.max()) + 1, dim, padding_idx=0)
            self.categories = nn.Embedding(int(categories.max()) + 1, dim)
            self.prices = nn.Embedding(int(prices.max()) + 1, dim)
            self.item_mlp = nn.Sequential(nn.Linear(3 * dim + popularity.shape[-1], hidden), nn.ReLU(),
                                          nn.Linear(hidden, dim))
            self.session_mlp = nn.Sequential(nn.Linear(2 * dim + 1, hidden), nn.ReLU(), nn.Linear(hidden, dim))
            self.log_scale = nn.Parameter(torch.tensor(float(np.log(10.0))))

        def items(self, rows, days):
            rows, days = torch.as_tensor(rows), torch.as_tensor(days)
            tokens = names[rows]
            mask = (tokens > 0).float().unsqueeze(-1)
            bag = (self.tokens(tokens) * mask).sum(-2) / mask.sum(-2).clamp(min=1.0)
            x = torch.cat([bag, self.categories(categories[rows]), self.prices(prices[rows]),
                           popularity[days, rows]], dim=-1)
            return F.normalize(self.item_mlp(x), dim=-1)

        def sessions(self, vectors, mask):
            """vectors: [B, H, dim] product vectors; mask: [B, H] of real (not padded) positions."""
            weights = mask.float().unsqueeze(-1)
            mean = (vectors * weights).sum(1) / weights.sum(1).clamp(min=1.0)
            last_index = (mask.long().sum(1) - 1).clamp(min=0)
            last = vectors[torch.arange(len(vectors)), last_index]
            length = torch.log1p(mask.float().sum(1, keepdim=True))
            return F.normalize(self.session_mlp(torch.cat([mean, last, length], dim=-1)), dim=-1)

        def scale(self):
            return self.log_scale.clamp(0.0, 4.0).exp()

    return SessionTwoTower()


class HomeData:
    """Catalog arrays, popularity, similarity and sequences shared by training and evaluation."""

    def __init__(self, settings, knn_settings):
        run = Path(settings["trained_run"])
        products = pd.read_parquet(run / "products.parquet")
        self.products = products
        self.item_index = pd.Series(np.arange(len(products)), index=products.item_id.to_numpy())
        self.catalog_ids = products.item_id.to_numpy()
        self.categories = products.category_id.fillna(-1).to_numpy(np.int64)
        vocab = {t: i for i, t in enumerate(sorted({t for s in products.name_tokens.dropna()
                                                     for t in s.split(",") if t}), start=1)}
        self.names = _token_matrix(products.name_tokens.tolist(), vocab, settings["max_name_tokens"])
        self.category_ids = np.where(self.categories < 0, 0, self.categories + 1)
        self.prices = products.price_log2.clip(lower=0).to_numpy(np.int64)
        window = settings["training_window"]
        start, end = pd.Timestamp(window["start"]), pd.Timestamp(window["end_exclusive"])
        self.days = int((end - start).days)
        clicks, purchases = window_activity(settings["raw_directory"], self.item_index, start, self.days)
        self.popularity_inputs = np.ascontiguousarray(popularity_features(clicks, purchases))
        self.popularity = clicks.sum(0).astype(float)
        recent = clicks[max(0, self.days - settings["trending_days"]):].sum(0).astype(float)
        self.global_order = np.argsort(-self.popularity, kind="stable")
        self.trending_order = np.lexsort((-self.popularity, -recent))
        session, item, _ = session_baskets(settings["raw_directory"], window["start"], window["end_exclusive"],
                                           self.item_index, knn_settings["max_items_per_session"])
        self.similarity, _ = cosine_similarity(session, item, len(products), knn_settings["shrinkage"])
        self.similarity = self.similarity.tocsr()
        self.diversity = ProductDiversity(products)


def train_model(data, settings, sequences, check_cases):
    """Fit the session two-tower; keep the epoch with the best check-set recall@10."""
    import torch
    torch.manual_seed(settings["seed"])
    torch.set_num_threads(settings["threads"])
    rng = np.random.default_rng(settings["seed"])
    history, targets, days = training_examples(sequences, settings["training_window"]["start"],
                                               settings["history_items"])
    model = build_session_two_tower(data.names, data.category_ids, data.prices, data.popularity_inputs, settings)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"])
    weights = np.maximum(data.popularity, 0) ** settings["popular_negative_power"]
    weights = weights / weights.sum()
    products = len(data.catalog_ids)
    log, best = [], (-1.0, None, 0)
    for epoch in range(1, settings["epochs"] + 1):
        model.train()
        order = rng.permutation(len(targets))
        total, steps = 0.0, 0
        for start in range(0, len(order), settings["batch_size"]):
            batch = order[start:start + settings["batch_size"]]
            rows, mask = history[batch], history[batch] >= 0
            day = days[batch]
            negatives = np.concatenate([
                rng.integers(0, products, (len(batch), settings["uniform_negatives"])),
                rng.choice(products, (len(batch), settings["popular_negatives"]), p=weights)], axis=1)
            vectors = model.items(np.where(mask, rows, 0).reshape(-1), np.repeat(day, rows.shape[1])).reshape(
                len(batch), rows.shape[1], -1)
            session = model.sessions(vectors, torch.as_tensor(mask))
            positive = (session * model.items(targets[batch], day)).sum(-1, keepdim=True)
            other = model.items(negatives.reshape(-1), np.repeat(day, negatives.shape[1])).reshape(
                len(batch), negatives.shape[1], -1)
            negative = (session.unsqueeze(1) * other).sum(-1)
            logits = torch.cat([positive, negative], dim=1) * model.scale()
            loss = torch.nn.functional.cross_entropy(logits, torch.zeros(len(batch), dtype=torch.long))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total, steps = total + float(loss), steps + 1
        model.eval()
        recall = np.mean([r["recall"] for r in score_two_tower_cases(model, data, settings, check_cases)])
        log.append(dict(epoch=epoch, loss=total / steps, check_recall_at_10=float(recall)))
        print(f"[home] epoch {epoch}: loss {total / steps:.4f}, check recall@10 {recall:.4f}", flush=True)
        if recall > best[0]:
            best = (recall, {k: v.detach().clone() for k, v in model.state_dict().items()}, epoch)
    model.load_state_dict(best[1])
    model.eval()
    return model, pd.DataFrame(log), dict(examples=int(len(targets)), best_epoch=best[2])


def item_vectors(model, data):
    import torch
    with torch.no_grad():
        rows = np.arange(len(data.catalog_ids))
        return np.concatenate([model.items(rows[i:i + 16384], np.full(len(rows[i:i + 16384]), data.days)).numpy()
                               for i in range(0, len(rows), 16384)])


def session_vector(model, vectors, prefix, history):
    import torch
    rows = prefix[-history:]
    with torch.no_grad():
        stacked = torch.as_tensor(vectors[rows]).unsqueeze(0)
        return model.sessions(stacked, torch.ones(1, len(rows), dtype=torch.bool)).numpy()[0]


def top_excluding(order_or_scores, exclude, count, scores=False):
    """First `count` products of an order (or by descending score) not in `exclude`."""
    if scores:
        values = order_or_scores
        candidates = np.argpartition(-values, count + len(exclude))[:count + len(exclude)]
        ranked = candidates[np.argsort(-values[candidates], kind="stable")]
    else:
        ranked = order_or_scores[:count + len(exclude)]
    return np.array([p for p in ranked if p not in exclude][:count], dtype=int)


def knn_scores(data, prefix):
    """Sum of co-click similarities to the clicked products, with a tiny popularity tie-break."""
    row = np.asarray(data.similarity[prefix].sum(axis=0)).ravel()
    return row + data.popularity / (data.popularity.max() + 1) * 1e-7


def score_two_tower_cases(model, data, settings, cases):
    vectors = item_vectors(model, data)
    rows = []
    for _, prefix, targets in cases:
        scores = vectors @ session_vector(model, vectors, prefix, settings["history_items"])
        top = top_excluding(scores, set(prefix), settings["top_k"], scores=True)
        rows.append(dict(recall=len(targets & set(top.tolist())) / len(targets)))
    return rows


def case_lists(data, model, vectors, settings, prefix):
    """Every method's top-k product rows for one case; products already clicked are excluded."""
    k, slots, pool = settings["top_k"], settings["relevance_slots"], settings["pool_size"]
    exclude = set(prefix)
    tower_scores = vectors @ session_vector(model, vectors, prefix, settings["history_items"])
    neighbour_scores = knn_scores(data, prefix)
    ranked = {"two_tower": (tower_scores, top_excluding(tower_scores, exclude, pool, scores=True)),
              "item_knn": (neighbour_scores, top_excluding(neighbour_scores, exclude, pool, scores=True))}
    lists = {"popularity": top_excluding(data.global_order, exclude, k),
             "trending": top_excluding(data.trending_order, exclude, k),
             "item_knn": ranked["item_knn"][1][:k], "two_tower": ranked["two_tower"][1][:k]}
    for name in settings["diversity_bases"]:
        values, members = ranked[name]
        relevance = values[members]
        ids = data.catalog_ids[members].tolist()
        reward = np.nan_to_num(data.diversity.distances(ids)["combined"], nan=0.0)
        lists[f"six_four_{name}"] = members[diverse_tail_order(relevance, reward, slots, k)]
        for alpha in settings["fixed_alphas"]:
            lists[f"fixed_{alpha:g}_{name}"] = members[greedy_rerank(relevance, reward, alpha, k)]
    return lists


def load_frozen_model(data, settings):
    """The session two-tower saved by the frozen validation run."""
    import torch
    frozen_run = Path(settings["frozen_run"])
    saved = json.loads((frozen_run / "home_settings.json").read_text())["settings"]
    model = build_session_two_tower(data.names, data.category_ids, data.prices, data.popularity_inputs, saved)
    model.load_state_dict(torch.load(frozen_run / "session_two_tower.pt"))
    model.eval()
    return model


def evaluate(data, model, settings, cases):
    """All methods on identical cases, one row per case and method."""
    slots = settings["relevance_slots"]
    vectors = item_vectors(model, data)
    rows = []
    for number, (key, prefix, targets) in enumerate(cases, start=1):
        reference = prefix[-1]
        base = dict(session_key=key, clicked_so_far=len(prefix), targets=len(targets))
        lists = case_lists(data, model, vectors, settings, prefix)
        for method, top in lists.items():
            rows.append(base | dict(method=method, **list_metrics(np.asarray(top), targets, reference,
                                                                  data.catalog_ids, data.categories,
                                                                  data.diversity, slots)))
        if number % 2000 == 0 or number == len(cases):
            print(f"[home] {number:,}/{len(cases):,} cases", flush=True)
    frame = pd.DataFrame(rows)
    frame["case"] = frame.groupby("method").cumcount()
    return frame


def differences(per_case, settings):
    rng = np.random.default_rng(settings["seed"])
    rows = []
    for reference in ("popularity", "two_tower"):
        base = per_case[per_case.method.eq(reference)].set_index("case")
        for method, frame in per_case.groupby("method", sort=False):
            if method == reference:
                continue
            frame = frame.set_index("case").reindex(base.index)
            for metric in RELEVANCE + ["combined_ild", "category_ild", "tail_to_head_distance"]:
                diff = frame[metric] - base[metric]
                low, high = session_bootstrap(diff, base.session_key, settings["bootstrap_replicates"], rng)
                rows.append(dict(method=method, reference=reference, metric=metric, value=float(frame[metric].mean()),
                                 reference_value=float(base[metric].mean()), difference=float(diff.mean()),
                                 ci_low=low, ci_high=high))
    return pd.DataFrame(rows)


def run_home_page(settings_path=Path("config/home_page.json"), output_root=None):
    import torch
    settings = read_home_settings(settings_path)
    knn_settings = read_knn_settings(settings["knn_config"])
    window = settings["window"]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    folder = Path(output_root or settings["output_directory"]) / f"{stamp}_{window['name']}"
    folder.mkdir(parents=True)
    selection = window["name"] == "selection"
    if selection:
        from .heldout import git_state
        frozen_run = Path(settings["frozen_run"])
        write_json(folder / "frozen_choices.json", dict(
            frozen_at_utc=datetime.now(timezone.utc).isoformat(), code=git_state(Path.cwd()), page="home page",
            policy=settings["frozen_policy"], reason=settings.get("frozen_reason"),
            model=str(frozen_run / "session_two_tower.pt"), model_settings=str(frozen_run / "home_settings.json"),
            heldout_week=window))
        print(f"[home] Frozen policy {settings['frozen_policy']} written before opening the week", flush=True)
    print("[home] Catalog, popularity and co-click similarity from the training window", flush=True)
    data = HomeData(settings, knn_settings)
    if selection:
        model = load_frozen_model(data, settings)
        training = dict(reused=str(frozen_run))
        log = pd.DataFrame()
    else:
        print("[home] Training sessions", flush=True)
        train_window = settings["training_window"]
        sequences = click_sequences(settings["raw_directory"], train_window["start"], train_window["end_exclusive"],
                                    data.item_index)
    print(f"[home] {window['name']} week {window['start']} to before {window['end_exclusive']}", flush=True)
    week = click_sequences(settings["raw_directory"], window["start"], window["end_exclusive"], data.item_index)
    cases, total = evaluation_cases(week, settings["max_cases"], settings["seed"])
    if not selection:
        check = cases[:: max(1, len(cases) // settings["epoch_check_cases"])][:settings["epoch_check_cases"]]
        model, log, training = train_model(data, settings, sequences, check)
        torch.save(model.state_dict(), folder / "session_two_tower.pt")
    per_case = evaluate(data, model, settings, cases)
    summary = pd.concat([
        frame.groupby("method", sort=False)[RELEVANCE + DIVERSITY].mean()
        .join(frame.groupby("method", sort=False).size().rename("cases")).reset_index().assign(segment=name)
        for name, frame in (("all_cases", per_case), ("one_click_so_far", per_case[per_case.clicked_so_far.eq(1)]),
                            ("two_plus_clicks_so_far", per_case[per_case.clicked_so_far.ge(2)]))],
        ignore_index=True)
    write_json(folder / "home_settings.json", dict(settings=settings, knn=knn_settings, training=training,
                                                   sessions=len(week), evaluable_cases=total,
                                                   evaluated_cases=len(cases)))
    save_table(log, folder / "training_log.csv")
    save_table(per_case, folder / "per_case.parquet")
    save_table(summary, folder / "summary.csv")
    save_table(differences(per_case, settings), folder / "differences.csv")
    print(f"[home] Saved to {folder}", flush=True)
    return folder
