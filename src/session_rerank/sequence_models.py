"""Neural sequence models for the home page: SASRec and GRU4Rec.

Both read a session's clicks in order and predict the next clicked product.
They are compared with the frozen home-page policy and its alternatives on
exactly the cases of a saved home-page run (same week, same sampled cases),
so every difference is paired.

  * GRU4Rec: a recurrent network (GRU) over the product embeddings of the clicks.
  * SASRec: a causal self-attention (transformer) encoder over the same
    embeddings plus position embeddings; each position attends to earlier clicks.

Data follow the agreed rules: clicks only, every eligible April session from the
raw files, products in order of first click, the last `max_length` clicks.
Each model learns one vector per product clicked at least once in April (its
vocabulary); products outside it cannot be recommended, and the share of cases
whose later clicks are all outside it is reported. Output scores are the dot
product of the session state with the (shared) product vectors.

Training: next-product prediction at every position of every session with two
or more known clicks, sampled softmax against `negatives` shared products per
batch (half uniform, half popularity-weighted; accidental hits are masked). The
epoch with the best recall@10 on a fixed subset of the evaluation cases is kept,
as for the session two-tower. A selection-week run reuses the saved models.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .diversity import ProductDiversity
from .event_ablation import session_bootstrap
from .home_page import click_sequences, evaluation_cases, read_home_settings
from .study_files import save_table, write_json
from .ymal import DIVERSITY, RELEVANCE, list_metrics

MODELS = ("sasrec", "gru4rec")


def read_sequence_settings(path):
    settings = json.loads(Path(path).read_text(encoding="utf-8"))
    if not settings["models"] or set(settings["models"]) - set(MODELS):
        raise ValueError(f"models must come from {MODELS}")
    if not 0 < settings.get("embedding_init_std", 0.02) <= 1:
        raise ValueError("embedding_init_std must be in (0, 1]")
    for name in ("max_length", "embedding_dim", "sasrec_blocks", "sasrec_heads", "epochs", "batch_size",
                 "negatives", "epoch_check_cases", "threads"):
        if not isinstance(settings[name], int) or settings[name] < 1:
            raise ValueError(f"{name} must be a positive integer")
    if settings["embedding_dim"] % settings["sasrec_heads"]:
        raise ValueError("embedding_dim must be divisible by sasrec_heads")
    if settings.get("device", "auto") not in ("auto", "cpu", "cuda"):
        raise ValueError("device must be auto, cpu or cuda")
    if not 0 <= settings["popular_negative_share"] <= 1 or not 0 <= settings["dropout"] < 1:
        raise ValueError("popular_negative_share must be in [0, 1] and dropout in [0, 1)")
    home = read_home_settings(settings["home_config"])
    if home["window"]["name"] == "selection" and not settings.get("frozen_run"):
        raise ValueError("The selection week needs frozen_run: the saved validation-week sequence models")
    # Optional longer training history; it must end before the evaluation weeks
    # (May), like every other model. Defaults to the home page's April window.
    window = settings.setdefault("training_window", home["training_window"])
    if not "2016-01-01" <= window["start"] < window["end_exclusive"] <= "2016-04-30":
        raise ValueError("training_window must lie within 2016-01-01 and 2016-04-30 (before the evaluation weeks)")
    return settings, home


# ------------------------------------------------------------------- data ---

def vocabulary(sequences):
    """Product rows clicked in the training window -> ids 1..V (0 is padding)."""
    rows = np.unique(np.concatenate([np.asarray(items) for _, _, items in sequences]))
    return pd.Series(np.arange(1, len(rows) + 1), index=rows)


def training_batches(sequences, vocab, max_length):
    """Right-padded inputs and next-click targets for sessions with two or more known clicks."""
    inputs, targets = [], []
    for _, _, items in sequences:
        known = [int(vocab[i]) for i in items if i in vocab.index][-(max_length + 1):]
        if len(known) < 2:
            continue
        x, y = known[:-1], known[1:]
        inputs.append(x + [0] * (max_length - len(x)))
        targets.append(y + [0] * (max_length - len(y)))
    return np.asarray(inputs, dtype=np.int64), np.asarray(targets, dtype=np.int64)


# ------------------------------------------------------------------ models ---

def build_model(kind, products, settings):
    import torch
    from torch import nn

    dim, length, dropout = settings["embedding_dim"], settings["max_length"], settings["dropout"]

    class GRU4Rec(nn.Module):
        def __init__(self):
            super().__init__()
            self.items = nn.Embedding(products + 1, dim, padding_idx=0)
            self.dropout = nn.Dropout(dropout)
            self.gru = nn.GRU(dim, dim, batch_first=True)

        def forward(self, sequence):
            hidden, _ = self.gru(self.dropout(self.items(sequence)))
            return hidden

    class SASRec(nn.Module):
        def __init__(self):
            super().__init__()
            self.items = nn.Embedding(products + 1, dim, padding_idx=0)
            self.positions = nn.Embedding(length, dim)
            self.dropout = nn.Dropout(dropout)
            layer = nn.TransformerEncoderLayer(dim, settings["sasrec_heads"], 2 * dim, dropout,
                                               batch_first=True, norm_first=True)
            self.encoder = nn.TransformerEncoder(layer, settings["sasrec_blocks"], enable_nested_tensor=False)
            self.norm = nn.LayerNorm(dim)

        def forward(self, sequence):
            steps = sequence.shape[1]
            # Positions and mask must live on the input's device (GPU or CPU).
            x = self.items(sequence) * dim ** 0.5 + self.positions(torch.arange(steps, device=sequence.device))
            causal = torch.triu(torch.full((steps, steps), float("-inf"), device=sequence.device), diagonal=1)
            return self.norm(self.encoder(self.dropout(x), mask=causal))

    model = {"gru4rec": GRU4Rec, "sasrec": SASRec}[kind]()
    # Small embeddings, as in the reference implementations. PyTorch's default
    # (standard deviation 1) gives huge dot-product scores, which saturated the
    # softmax in the first run (SASRec stayed at chance level; see the log §6j).
    with torch.no_grad():
        nn.init.normal_(model.items.weight, std=settings.get("embedding_init_std", 0.02))
        model.items.weight[0].zero_()
        if kind == "sasrec":
            nn.init.normal_(model.positions.weight, std=settings.get("embedding_init_std", 0.02))
    return model


def choose_device(settings):
    """'auto' (default) uses a GPU when one is visible, otherwise the CPU."""
    import torch
    wanted = settings.get("device", "auto")
    if wanted == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if wanted == "cuda" and not torch.cuda.is_available():
        raise ValueError("device is 'cuda' but no GPU is visible")
    return wanted


def item_weights(model):
    """Product vectors as a CPU array, computed once per scoring pass."""
    return model.items.weight.detach().cpu().numpy()


def last_state(model, prefix_ids):
    """Session state after the last known click (one session)."""
    import torch
    device = next(model.parameters()).device
    with torch.no_grad():
        hidden = model(torch.as_tensor([prefix_ids], device=device))
    return hidden[0, -1].cpu().numpy()


def recommend(model, prefix_rows, vocab, ids_to_rows, fallback_rows, settings, k, weights=None):
    """Top-k product rows for one case; popularity fallback when no click is known."""
    known = [int(vocab[r]) for r in prefix_rows if r in vocab.index][-settings["max_length"]:]
    exclude = set(prefix_rows)
    if not known:
        return np.array([r for r in fallback_rows[:k + len(exclude)] if r not in exclude][:k]), False
    weights = item_weights(model) if weights is None else weights
    scores = weights[1:] @ last_state(model, known)
    order = np.argpartition(-scores, k + len(exclude))[:k + len(exclude)]
    order = order[np.argsort(-scores[order], kind="stable")]
    rows = [int(ids_to_rows[i + 1]) for i in order]
    return np.array([r for r in rows if r not in exclude][:k]), True


def train(kind, sequences, vocab, settings, check_cases, ids_to_rows, fallback_rows, k):
    import torch
    torch.manual_seed(settings["seed"])
    torch.set_num_threads(settings["threads"])
    rng = np.random.default_rng(settings["seed"])
    inputs, targets = training_batches(sequences, vocab, settings["max_length"])
    device = choose_device(settings)
    model = build_model(kind, len(vocab), settings).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"])
    counts = np.bincount(targets[targets > 0], minlength=len(vocab) + 1)[1:].astype(float) + 1
    popular = counts ** settings["popular_negative_power"]
    popular /= popular.sum()
    n_popular = int(round(settings["negatives"] * settings["popular_negative_share"]))
    log, best = [], (-1.0, None, 0)
    for epoch in range(1, settings["epochs"] + 1):
        model.train()
        order = rng.permutation(len(inputs))
        total, steps = 0.0, 0
        for start in range(0, len(order), settings["batch_size"]):
            batch = order[start:start + settings["batch_size"]]
            x = torch.as_tensor(inputs[batch], device=device)
            y = torch.as_tensor(targets[batch], device=device)
            valid = y > 0
            hidden = model(x)[valid]
            target = y[valid]
            negatives = torch.as_tensor(np.concatenate([
                rng.integers(1, len(vocab) + 1, settings["negatives"] - n_popular),
                rng.choice(len(vocab), n_popular, p=popular) + 1]), device=device)
            positive = (hidden * model.items(target)).sum(-1, keepdim=True)
            negative = hidden @ model.items(negatives).T
            negative = negative.masked_fill(negatives.unsqueeze(0) == target.unsqueeze(1), float("-inf"))
            loss = torch.nn.functional.cross_entropy(torch.cat([positive, negative], 1),
                                                     torch.zeros(len(target), dtype=torch.long, device=device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total, steps = total + float(loss), steps + 1
        model.eval()
        weights = item_weights(model)
        recall = np.mean([len(targets_ & set(recommend(model, prefix, vocab, ids_to_rows, fallback_rows,
                                                         settings, k, weights)[0].tolist())) / len(targets_)
                          for _, prefix, targets_ in check_cases])
        log.append(dict(model=kind, epoch=epoch, loss=total / steps, check_recall_at_10=float(recall)))
        print(f"[{kind}] epoch {epoch}: loss {total / steps:.4f}, check recall@10 {recall:.4f}", flush=True)
        if recall > best[0]:
            best = (recall, {key: v.detach().clone() for key, v in model.state_dict().items()}, epoch)
    model.load_state_dict(best[1])
    model.eval()
    return model, log, dict(training_sessions=int(len(inputs)), training_positions=int((targets > 0).sum()),
                            best_epoch=best[2], device=device)


# --------------------------------------------------------------- evaluation ---

def run_sequence_models(settings_path=Path("config/sequence_models.json"), output_root=None):
    import torch
    settings, home = read_sequence_settings(settings_path)
    window = home["window"]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    folder = Path(output_root or settings["output_directory"]) / f"{stamp}_{window['name']}"
    folder.mkdir(parents=True)
    selection = window["name"] == "selection"
    if selection:
        from .heldout import git_state
        frozen = Path(settings["frozen_run"])
        write_json(folder / "frozen_choices.json", dict(
            frozen_at_utc=datetime.now(timezone.utc).isoformat(), code=git_state(Path.cwd()),
            page="home page (comparison models)", models={m: str(frozen / f"{m}.pt") for m in settings["models"]},
            settings=str(frozen / "sequence_settings.json"), heldout_week=window))
        print("[sequence] Frozen models recorded before opening the week", flush=True)
    products = pd.read_parquet(Path(home["trained_run"]) / "products.parquet")
    item_index = pd.Series(np.arange(len(products)), index=products.item_id.to_numpy())
    catalog_ids = products.item_id.to_numpy()
    categories = products.category_id.fillna(-1).to_numpy(np.int64)
    diversity = ProductDiversity(products)
    print(f"[sequence] Click sequences {settings['training_window']}", flush=True)
    train_window = settings["training_window"]
    april = click_sequences(home["raw_directory"], train_window["start"], train_window["end_exclusive"], item_index)
    if selection:
        saved = json.loads((frozen / "sequence_settings.json").read_text())
        vocab_rows = np.load(frozen / "vocabulary_rows.npy")
        vocab = pd.Series(np.arange(1, len(vocab_rows) + 1), index=vocab_rows)
        model_settings = saved["settings"]
    else:
        vocab = vocabulary(april)
        model_settings = settings
        np.save(folder / "vocabulary_rows.npy", vocab.index.to_numpy())
    ids_to_rows = pd.Series(vocab.index.to_numpy(), index=vocab.to_numpy())
    clicks = np.zeros(len(products))
    for _, _, items in april:
        np.add.at(clicks, items, 1)
    fallback_rows = np.argsort(-clicks, kind="stable")
    print(f"[sequence] {window['name']} week cases (same as the home-page run)", flush=True)
    week = click_sequences(home["raw_directory"], window["start"], window["end_exclusive"], item_index)
    cases, total = evaluation_cases(week, home["max_cases"], home["seed"])
    saved_cases = pd.read_parquet(Path(settings["home_run"]) / "per_case.parquet")
    reference = saved_cases[saved_cases.method.eq("popularity")].sort_values("case")
    if len(reference) != len(cases) or not np.array_equal(reference.clicked_so_far.to_numpy(),
                                                          [len(c[1]) for c in cases]):
        raise ValueError("These cases do not match the saved home-page run")
    k, slots = home["top_k"], home["relevance_slots"]
    models, logs, training = {}, [], {}
    device = choose_device(settings)
    print(f"[sequence] device: {device}", flush=True)
    for kind in settings["models"]:
        model = build_model(kind, len(vocab), model_settings)
        if selection:
            model.load_state_dict(torch.load(frozen / f"{kind}.pt", map_location=device))
            model = model.to(device)
            model.eval()
            training[kind] = dict(reused=str(frozen), device=device)
        else:
            check = cases[:: max(1, len(cases) // settings["epoch_check_cases"])][:settings["epoch_check_cases"]]
            model, log, training[kind] = train(kind, april, vocab, settings, check, ids_to_rows, fallback_rows, k)
            logs.extend(log)
            # Saved as CPU tensors so a GPU-trained model also loads on a laptop.
            torch.save({key: value.cpu() for key, value in model.state_dict().items()}, folder / f"{kind}.pt")
        models[kind] = model
    weights = {kind: item_weights(model) for kind, model in models.items()}
    rows = []
    for number, (key, prefix, later) in enumerate(cases):
        for kind, model in models.items():
            top, personal = recommend(model, prefix, vocab, ids_to_rows, fallback_rows, model_settings, k,
                                      weights[kind])
            rows.append(dict(session_key=key, clicked_so_far=len(prefix), targets=len(later), method=kind,
                             case=number, used_session=personal,
                             targets_in_vocabulary=bool(set(later) & set(vocab.index)),
                             **list_metrics(top, later, prefix[-1], catalog_ids, categories, diversity, slots)))
        if (number + 1) % 2000 == 0 or number + 1 == len(cases):
            print(f"[sequence] {number + 1:,}/{len(cases):,} cases", flush=True)
    ours = pd.DataFrame(rows)
    combined = pd.concat([saved_cases[saved_cases.method.isin(["popularity", "item_knn", "two_tower",
                                                               "fixed_0.8_item_knn"])], ours], ignore_index=True)
    combined["clicks_so_far"] = pd.cut(combined.clicked_so_far, [0, 1, 2, 5, np.inf], labels=["1", "2", "3-5", "6+"])
    summary = pd.concat([
        combined.groupby("method", sort=False)[RELEVANCE + DIVERSITY].mean().reset_index().assign(segment="all cases"),
        combined.groupby(["clicks_so_far", "method"], observed=True, sort=False)[RELEVANCE + ["combined_ild"]]
        .mean().reset_index().rename(columns={"clicks_so_far": "segment"})], ignore_index=True)
    rng = np.random.default_rng(settings["seed"])
    differences = []
    for reference_method in ("item_knn", "fixed_0.8_item_knn", "two_tower"):
        base = combined[combined.method.eq(reference_method)].set_index("case")
        for kind in settings["models"]:
            other = combined[combined.method.eq(kind)].set_index("case").reindex(base.index)
            for metric in RELEVANCE + ["combined_ild"]:
                diff = other[metric] - base[metric]
                low, high = session_bootstrap(diff, base.session_key, settings["bootstrap_replicates"], rng)
                differences.append(dict(method=kind, reference=reference_method, metric=metric,
                                        value=float(other[metric].mean()), reference_value=float(base[metric].mean()),
                                        difference=float(diff.mean()), ci_low=low, ci_high=high))
    write_json(folder / "sequence_settings.json", dict(
        settings=model_settings if selection else settings, home=home, training=training,
        vocabulary=int(len(vocab)), evaluable_cases=total, evaluated_cases=len(cases),
        cases_without_known_click=float(1 - ours.used_session.mean()),
        cases_with_a_predictable_target=float(ours.targets_in_vocabulary.mean())))
    save_table(pd.DataFrame(logs), folder / "training_log.csv")
    save_table(ours, folder / "per_case.parquet")
    save_table(summary, folder / "summary.csv")
    save_table(pd.DataFrame(differences), folder / "differences.csv")
    print(f"[sequence] Saved to {folder}", flush=True)
    return folder
