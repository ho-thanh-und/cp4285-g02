"""Check the prepared full-scale arrays and declared sampling/split contracts."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .benchmark_data import ROOT, OUT, load, write_json, NAMES


def audit():
    populations=[];episodes={};all_ids=set()
    protocol=json.loads(Path('config/session_benchmark_protocol.json').read_text())
    for split in ['train','validation','test']:
        d=load(split);m=d['meta'];o=d['offsets']
        assert len(o)==len(m)+1 and o[0]==0 and o[-1]==len(d['items'])
        assert d['X'].shape==(o[-1],len(NAMES))
        assert np.array_equal(np.diff(o),m.candidate_count)
        assert m.query_id.is_unique and not (set(m.query_id)&all_ids)
        all_ids.update(m.query_id)
        episodes[split]=set(m.episode)
        assert m.fraction_rank.between(0,1,inclusive='neither').all()
        assert (m.groupby('episode').fraction_rank.nunique()==1).all()
        start,end=protocol['confirmation' if split=='test' else split]
        assert m.date.between(start,end).all()
        # Blockwise validation keeps peak memory bounded at full scale.
        for a in range(0,len(d['labels']),1_000_000):
            b=min(a+1_000_000,len(d['labels']))
            assert np.isfinite(d['X'][a:b]).all()
            assert np.isin(d['labels'][a:b],[0,1]).all()
            assert np.isin(d['grades'][a:b],[0,1,2,3]).all()
            assert ((d['grades'][a:b]>=2)==(d['labels'][a:b]>0)).all()
        totals=np.add.reduceat(np.asarray(d['labels']),o[:-1])
        assert np.array_equal(totals,m.positives)
        previous=set()
        for f in [.1,.25,.5,1.]:
            take=m.fraction_rank<f
            chosen=set(m.loc[take,'episode'])
            assert previous<=chosen;previous=chosen
            populations.append(dict(split=split,fraction=f,queries=int(take.sum()),episodes=len(chosen),
                candidates=int(m.loc[take,'candidate_count'].sum()),known_fraction=float(m.loc[take,'known'].mean()),
                positive_fraction=float((m.loc[take,'positives']>0).mean()),
                history_available_fraction=float((m.loc[take,'history_count']>0).mean()),
                anomalous_queries=int(m.loc[take,'anomalous'].sum())))
    for a,b in [('train','validation'),('train','test'),('validation','test')]:
        assert not episodes[a]&episodes[b]
    pd.DataFrame(populations).to_csv(OUT/'dataset_populations.csv',index=False)
    write_json(OUT/'data_contract_checks.json',dict(status='passed',query_ids_disjoint=True,
        episodes_disjoint=True,nested_whole_episode_samples=True,arrays_finite=True,
        candidate_counts_and_label_totals_match=True,
        note='Runtime array contracts complement the synthetic causal/floor/metric tests; they do not establish timestamp units or causal relevance.'))
    print('Full-scale data/split/sampling/label contracts passed.',flush=True)


if __name__=='__main__':audit()
