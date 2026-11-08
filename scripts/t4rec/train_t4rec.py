"""Train a Transformers4Rec XLNet session model on exported sessions and predict the cases.

Runs in the separate Transformers4Rec conda environment (see T4REC_GUIDE.md). It
imports nothing from this project: it reads the files written by
`python main.py export-sequences` and writes predictions.parquet for
`python main.py score-external`.

Model (the library's documented next-item setup): a schema with one sequence
column of product ids (`item_id-list`), causal masking, an MLP projection, an
XLNet transformer and a next-item prediction head with tied embeddings. Sessions
are right-padded with 0 to `max_length`.

Training uses the library's loss on every position. After each epoch, recall@10
is measured on the same fixed subset of cases the project uses for its own
neural models (every n-th case, 3,000 cases) and the best epoch is kept.

    python scripts/t4rec/train_t4rec.py --export results/t4rec/RUN_validation
    python scripts/t4rec/train_t4rec.py --export results/t4rec/RUN_validation --smoke

--smoke trains briefly on a small sample and writes predictions_smoke.parquet,
to check the installation and the library calls quickly before a full run.

Written from the library's documentation and not yet run (it cannot be installed
on the development laptop): expect to fix small API differences on first use.
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

CHECK_CASES = 3000
TOP_K = 10


def pad(sequences, length):
    """Right-pad id lists with 0 (the padding id) to a fixed length, keeping the last ids."""
    out = np.zeros((len(sequences), length), dtype=np.int64)
    for row, items in enumerate(sequences):
        items = list(items)[-length:]
        out[row, :len(items)] = items
    return out


def ranked_ids(scores, prefix, count):
    """Ids by descending score, without padding (0) and without the session's own clicks."""
    scores = np.asarray(scores, dtype=float).copy()
    scores[0] = -np.inf
    for item in prefix:
        if 0 <= item < len(scores):
            scores[item] = -np.inf
    top = np.argpartition(-scores, count)[:count]
    return top[np.argsort(-scores[top], kind="stable")].tolist()


def check_subset(cases):
    step = max(1, len(cases) // CHECK_CASES)
    return cases.iloc[::step].head(CHECK_CASES)


def build_model(settings, vocabulary_size):
    import transformers4rec.torch as tr
    from merlin.schema import ColumnSchema, Schema, Tags

    length, d_model = settings["max_length"], settings["d_model"]
    schema = Schema([ColumnSchema(
        "item_id-list", dtype=np.int64, is_list=True, is_ragged=False,
        tags=[Tags.ITEM_ID, Tags.ITEM, Tags.ID, Tags.CATEGORICAL, Tags.LIST],
        properties={"domain": {"min": 0, "max": vocabulary_size, "name": "item_id"},
                    "value_count": {"min": length, "max": length}})])
    inputs = tr.TabularSequenceFeatures.from_schema(schema, max_sequence_length=length, d_output=d_model,
                                                    masking="causal")
    config = tr.XLNetConfig.build(d_model=d_model, n_head=settings["n_head"], n_layer=settings["n_layer"],
                                  total_seq_length=length, dropout=settings["dropout"])
    body = tr.SequentialBlock(inputs, tr.MLPBlock([d_model]), tr.TransformerBlock(config, masking=inputs.masking))
    return tr.Model(tr.Head(body, tr.NextItemPredictionTask(weight_tying=True), inputs=inputs))


def scores_from_output(output):
    """Next-item scores [batch, vocabulary] from the forms different versions return."""
    import torch
    if isinstance(output, dict):
        output = output.get("predictions", output.get("logits"))
    if isinstance(output, (tuple, list)):
        output = output[0]
    if not isinstance(output, torch.Tensor):
        raise TypeError(f"Unexpected model output type: {type(output)}")
    if output.dim() == 3:  # [batch, positions, vocabulary]: keep the last position
        output = output[:, -1]
    return output


def predict(model, cases, settings, device, count):
    import torch
    model.eval()
    rows = []
    batch_size = settings["batch_size"]
    with torch.no_grad():
        for start in range(0, len(cases), batch_size):
            part = cases.iloc[start:start + batch_size]
            known = part[part.prefix_ids.map(len) > 0]
            if len(known):
                batch = {"item_id-list": torch.as_tensor(pad(known.prefix_ids, settings["max_length"]), device=device)}
                scores = scores_from_output(model(batch)).float().cpu().numpy()
                for (case, prefix), row in zip(zip(known.case, known.prefix_ids), scores):
                    rows.append(dict(case=int(case), item_ids=ranked_ids(row, list(prefix), count)))
            rows += [dict(case=int(c), item_ids=[]) for c in part.loc[part.prefix_ids.map(len) == 0, "case"]]
    return pd.DataFrame(rows).sort_values("case")


def recall_at_k(predictions, cases, vocabulary):
    to_id = dict(zip(vocabulary.product_row, vocabulary.vocab_id))
    later = cases.set_index("case").later_rows
    values = []
    for case, ids in zip(predictions.case, predictions.item_ids):
        targets = {to_id[r] for r in later[case] if r in to_id}
        total = len(later[case])
        values.append(len(targets & set(ids[:TOP_K])) / total if total else 0.0)
    return float(np.mean(values))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--export", required=True, help="Folder written by main.py export-sequences")
    parser.add_argument("--smoke", action="store_true", help="Short run on a small sample")
    args = parser.parse_args()

    import torch
    folder = Path(args.export)
    exported = json.loads((folder / "export.json").read_text())
    settings = exported["model"]
    random.seed(settings["seed"])
    np.random.seed(settings["seed"])
    torch.manual_seed(settings["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train = pd.read_parquet(folder / "train_sessions.parquet")
    cases = pd.read_parquet(folder / "cases.parquet")
    vocabulary = pd.read_parquet(folder / "vocabulary.parquet")
    epochs = settings["epochs"]
    if args.smoke:
        train, cases, epochs = train.head(2000), cases.head(500), 1
    check = check_subset(cases)
    print(f"[t4rec] {len(train):,} sessions, {len(cases):,} cases, {len(vocabulary):,} products, device {device}",
          flush=True)

    model = build_model(settings, int(vocabulary.vocab_id.max())).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"])
    sessions = pad(train.item_ids, settings["max_length"])
    rng = np.random.default_rng(settings["seed"])
    log, best = [], (-1.0, None, 0)
    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(len(sessions))
        total, steps = 0.0, 0
        for start in range(0, len(order), settings["batch_size"]):
            batch = {"item_id-list": torch.as_tensor(sessions[order[start:start + settings["batch_size"]]],
                                                     device=device)}
            output = model(batch, training=True)
            loss = output["loss"] if isinstance(output, dict) else output
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total, steps = total + float(loss), steps + 1
        recall = recall_at_k(predict(model, check, settings, device, TOP_K + 20), check, vocabulary)
        log.append(dict(epoch=epoch, loss=total / steps, check_recall_at_10=recall))
        print(f"[t4rec] epoch {epoch}: loss {total / steps:.4f}, check recall@10 {recall:.4f}", flush=True)
        if recall > best[0]:
            best = (recall, {k: v.detach().clone() for k, v in model.state_dict().items()}, epoch)
    model.load_state_dict(best[1])
    predictions = predict(model, cases, settings, device, settings["top_n"])
    suffix = "_smoke" if args.smoke else ""
    predictions.to_parquet(folder / f"predictions{suffix}.parquet", index=False)
    pd.DataFrame(log).to_csv(folder / f"t4rec_training_log{suffix}.csv", index=False)
    torch.save(model.state_dict(), folder / f"t4rec_xlnet{suffix}.pt")
    (folder / f"t4rec_run{suffix}.json").write_text(json.dumps(dict(
        best_epoch=best[2], best_check_recall=best[0], device=device, smoke=args.smoke,
        sessions=len(train), cases=len(cases)), indent=2))
    print(f"[t4rec] best epoch {best[2]}; wrote predictions{suffix}.parquet", flush=True)


if __name__ == "__main__":
    main()
