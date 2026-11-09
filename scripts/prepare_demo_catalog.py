"""Prepare fictional display data and record representative frozen-model paths."""
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
import pandas as pd
from session_rerank.serving import Recommender
from session_rerank.sequence_models import ranked

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / 'results/handoff/20261109T100233Z'
OUT = ROOT / 'web/catalog'
THEMES = {47: 'Electronics', 780: 'Kitchen appliances', 1096: 'Clothing', 842: 'Home furnishings'}
TYPES = [
    [('Over-ear headphones', 79), ('Portable speaker', 59), ('Wireless earbuds', 69), ('Portable charger', 39)],
    [('Electric kettle', 49), ('Two-slot toaster', 55), ('Rice cooker', 89), ('Coffee maker', 75)],
    [('Blue T-shirt', 25), ('Beige jacket', 79), ('Black trousers', 49), ('Grey sweatshirt', 45)],
    [('Cream cushion', 29), ('Woven rug', 89), ('White curtains', 69), ('Grey throw blanket', 45)],
]

def save(name, data):
    (OUT / name).write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    model = Recommender(FOLDER)
    data = pd.read_parquet(FOLDER / 'catalog.parquet').set_index('item_id')
    category = {int(i): None if pd.isna(c) else int(c) for i, c in data.category_id.items()}
    vocabulary = set(np.load(FOLDER / 'home/vocabulary_item_ids.npy').tolist())
    cold = model.home()['items']
    seed = set(cold)
    rankings = {}
    for c in THEMES:
        ids = model.category(c, limit=20)['items']
        rankings[c] = ids
        seed.update(ids)
    visits = [[rankings[47][0], rankings[47][1]], [rankings[47][0], rankings[780][0]], [rankings[1096][0], rankings[842][0]]]
    traces = []
    for visit in visits:
        rows = model.item_index.loc[visit].tolist()
        members, scores, personal = ranked(model.model, rows, model.vocab, model.ids_to_rows, model.fallback_rows, model.model_settings, 20, model.weights)
        pool = model.catalog_ids[members].astype(int).tolist()
        result = model.home(visit)
        seed.update(pool); seed.update(result['items'])
        traces.append(dict(clicks=visit, known_sasrec=[i in vocabulary for i in visit], top20=pool,
                           top20_categories=[category[i] for i in pool], final10=result['items'],
                           final10_categories=[category[i] for i in result['items']], diverse_positions=result['diverse_positions'], personal=bool(personal)))
    targets = set(seed)
    for item in list(seed):
        targets.update(model.item(item)['items'])
    # Prepare the four browse categories as a whole, plus one-hop targets from the inspected paths.
    prepared = set(int(i) for i in data[data.category_id.isin(THEMES)].index) | targets
    links = Counter()
    for item in seed:
        source = category[item]
        if source in THEMES:
            for target in model.item(item)['items']:
                if category[target] not in THEMES and category[target] is not None:
                    links[(category[target], source)] += 1
    support = {}
    for c, source in sorted(links, key=lambda x: (-links[x], x)):
        support.setdefault(c, source)
    # A supporting category gets a single theme, even when reached from different routes.
    records = []
    for item in sorted(prepared):
        c = category[item]
        theme = c if c in THEMES else support.get(c)
        if theme is None:
            continue
        row = list(THEMES).index(theme)
        tokens = str(data.loc[item, 'name_tokens']).split(',')
        key = hashlib.sha256(','.join(sorted(set(tokens))).encode()).digest()[0] % 4
        name, base = TYPES[row][key]
        records.append(dict(item_id=item, category_id=c, name=f'{name} {item}', price=base, currency='SGD', image=row * 4 + key,
                            alt=name))
    categories = [dict(category_id=c, name=name, featured=True, cover=rankings[c][0]) for c, name in THEMES.items()]
    for c, source in sorted(support.items()):
        categories.append(dict(category_id=c, name=f'{THEMES[source]} collection {c}', featured=False))
    fingerprint = hashlib.sha256((FOLDER / 'manifest.json').read_bytes()).hexdigest()
    save('categories.json', dict(manifest_sha256=fingerprint, categories=categories))
    save('products.json', dict(manifest_sha256=fingerprint, products=records))
    missing = sorted(targets - {r['item_id'] for r in records})
    save('coverage.json', dict(cold_home=cold, category_top20=rankings, visits=traces, seed_count=len(seed), target_count=len(targets), prepared_count=len(records), missing_targets=missing,
         note='One-hop targets from sampled paths; future sessions may reach unprepared products. Category IDs are anonymous.'))
    save('asset-sources.json', dict(image='/assets/products/catalog.png', source='Generated with OpenAI imagegen for Clew; fictional unbranded studio product contact sheet', prompt='Four by four product photo grid: headphones, speaker, earbuds, charger; kettle, toaster, rice cooker, coffee maker; T-shirt, jacket, trousers, sweatshirt; cushion, rug, curtains, throw.'))
    print('seed', len(seed), 'targets', len(targets), 'prepared', len(records), 'missing', len(missing))

if __name__ == '__main__':
    main()
