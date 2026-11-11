"""Prepare fictional display data and record representative frozen-model paths."""
import argparse
import lightgbm  # noqa: F401  (must be imported before pandas and torch; on Windows an OpenMP clash crashes LightGBM otherwise)
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
import pandas as pd
from session_rerank.demo import THEMES, display_record
from session_rerank.serving import Recommender
from session_rerank.sequence_models import ranked

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'web/catalog'
ROW = {c: i for i, c in enumerate(THEMES)}
THEME_NAME = list(THEMES.values())

def save(name, data):
    (OUT / name).write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')

def theme_every_category(folder, category, members, support):
    """Give every categorized product's category a theme row: the featured themes and the
    supporting categories first, then link propagation over the saved related products for
    the rest (most frequent theme among themed related products wins; ties to the lower
    row), repeated until nothing changes; whatever is left gets category_id % 4."""
    links = pd.read_parquet(folder / 'item/you_may_also_like.parquet').set_index('item_id')
    themes = dict(ROW)
    for c, source in support.items():
        themes[c] = ROW[source]
    item_theme = {item: themes[c] for item, c in category.items() if c in themes}
    pending = [c for c in sorted(members) if c not in themes]
    by_links = 0
    while True:
        votes = {}
        for c in pending:
            counts = Counter(item_theme[t] for i in members[c] for t in links.loc[i].tolist()
                             if t >= 0 and t in item_theme)
            if counts:
                votes[c] = min(counts, key=lambda t: (-counts[t], t))
        if not votes:
            break
        for c, row in votes.items():
            themes[c] = row
            for i in members[c]:
                item_theme[i] = row
        pending = [c for c in pending if c not in votes]
        by_links += len(votes)
    for c in pending:
        themes[c] = c % 4
    return themes, by_links, len(pending)

def main():
    parser = argparse.ArgumentParser(description="Prepare the Clew display catalog")
    parser.add_argument('--handoff', type=Path, default=Path('results/handoff/20261109T100233Z'),
                        help='Final exported model folder the display data is pinned to')
    args = parser.parse_args()
    FOLDER = args.handoff if args.handoff.is_absolute() else ROOT / args.handoff
    OUT.mkdir(parents=True, exist_ok=True)
    model = Recommender(FOLDER)
    data = pd.read_parquet(FOLDER / 'catalog.parquet').set_index('item_id')
    category = {int(i): None if pd.isna(c) else int(c) for i, c in data.category_id.items()}
    members = defaultdict(list)
    for item, c in category.items():
        if c is not None:
            members[c].append(item)
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
        members_rows, scores, personal = ranked(model.model, rows, model.vocab, model.ids_to_rows, model.fallback_rows, model.model_settings, 20, model.weights)
        pool = model.catalog_ids[members_rows].astype(int).tolist()
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
        records.append(display_record(item, c, data.loc[item, 'name_tokens'], ROW[theme]))
    categories = [dict(category_id=c, name=name, featured=True, cover=rankings[c][0], theme=ROW[c])
                  for c, name in THEMES.items()]
    for c, source in sorted(support.items()):
        categories.append(dict(category_id=c, name=f'{THEMES[source]} collection {c}', featured=False, theme=ROW[source]))
    # Every remaining categorized product's category gets a theme from its related-product links.
    themes, by_links, by_remainder = theme_every_category(FOLDER, category, members, support)
    listed = {c['category_id'] for c in categories}
    for c in sorted(set(members) - listed):
        categories.append(dict(category_id=c, name=f'{THEME_NAME[themes[c]]} collection {c}', featured=False, theme=themes[c]))
    fingerprint = hashlib.sha256((FOLDER / 'manifest.json').read_bytes()).hexdigest()
    save('categories.json', dict(manifest_sha256=fingerprint, categories=categories))
    save('products.json', dict(manifest_sha256=fingerprint, products=records))
    missing = sorted(targets - {r['item_id'] for r in records})
    save('coverage.json', dict(cold_home=cold, category_top20=rankings, visits=traces, seed_count=len(seed), target_count=len(targets), prepared_count=len(records), missing_targets=missing,
         all_products=dict(products_with_display_data=sum(1 for c in category.values() if c is not None),
                           products_without_category=sum(1 for c in category.values() if c is None),
                           categories_themed_by_links=by_links, categories_themed_by_id_modulo=by_remainder),
         note='One-hop targets from sampled paths; future sessions may reach unprepared products. Category IDs are anonymous.'))
    save('asset-sources.json', dict(image='/assets/products/catalog.png', source='Generated with OpenAI imagegen for Clew; fictional unbranded studio product contact sheet', prompt='Four by four product photo grid: headphones, speaker, earbuds, charger; kettle, toaster, rice cooker, coffee maker; T-shirt, jacket, trousers, sweatshirt; cushion, rug, curtains, throw.'))
    print('seed', len(seed), 'targets', len(targets), 'prepared', len(records), 'missing', len(missing),
          'categories', len(categories), 'themed by links', by_links, 'by id % 4', by_remainder)

if __name__ == '__main__':
    main()
