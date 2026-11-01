"""Independent array, cohort, hash and metric checks for the second dataset."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify():
    root=Path('data/retailrocket/replication_v2');out=Path('results/submission/replication_v2')
    frozen=json.loads((out/'frozen.json').read_text());receipt=json.loads((out/'confirmation_receipt.json').read_text())
    assert digest('config/submission_replication_v2_protocol.json')==frozen['protocol_sha256']
    assert digest(root/'complete.json')==frozen['data_sha256']
    assert digest(out/'frozen.json')==receipt['frozen_sha256']
    assert digest(out/'confirmation_metrics.npz')==receipt['metrics_sha256']
    for model in frozen['models']:assert digest(model['path'])==model['sha256']
    previous=pd.read_parquet('results/submission/replication/confirmation_metadata.parquet')
    episodes=[];rows={}
    for split in ['train','validation','confirmation']:
        p=root/split;m=pd.read_parquet(p/'meta.parquet');offsets=np.load(p/'offsets.npy')
        items=np.load(p/'items.npy',mmap_mode='r');labels=np.load(p/'labels.npy',mmap_mode='r')
        assert (m.request_time<m.target_time).all()
        assert np.array_equal(np.diff(offsets),m.candidate_count)
        for j,(a,b) in enumerate(zip(offsets[:-1],offsets[1:])):
            assert np.array_equal(labels[a:b],items[a:b]==m.target.iloc[j])
            assert len(np.unique(items[a:b]))==b-a
        assert np.array_equal(np.add.reduceat(labels,offsets[:-1]),m.positives)
        episodes.append(set(m.episode));rows[split]=dict(queries=len(m),episodes=m.episode.nunique(),retrieved=int(m.positives.sum()))
    assert all(not episodes[a]&episodes[b] for a,b in [(0,1),(0,2),(1,2)])
    m=pd.read_parquet(root/'confirmation/meta.parquet')
    assert not set(m.session_id)&set(previous.session_id)
    # Popularity score is feature 2; compare a separate sklearn NDCG formula.
    X=np.load(root/'confirmation/features.npy',mmap_mode='r');y=np.load(root/'confirmation/labels.npy',mmap_mode='r')
    offsets=np.load(root/'confirmation/offsets.npy');z=np.load(out/'confirmation_metrics.npz')
    index=z['names'].tolist().index('popularity');errors=[]
    for j in np.random.default_rng(4285).choice(len(m),min(1000,len(m)),replace=False):
        a,b=offsets[j:j+2];order=np.argsort(-X[a:b,2],kind='stable')
        scores=np.empty(b-a);scores[order]=np.arange(b-a,0,-1)
        exact=ndcg_score(y[a:b][None,:],scores[None,:],k=10) if b-a>1 else float(y[a])
        errors.append(abs(exact-float(z['values'][j,index,0])))
    assert max(errors)<1e-6
    result=dict(status='passed',splits=rows,episodes_disjoint=True,confirmation_visitors_disjoint_from_v1=True,
        strict_request_before_target=True,all_target_labels_match_candidate_membership=True,hashes_match=True,
        sklearn_metric_checks=len(errors),maximum_ndcg_error=max(errors),
        limitation='These checks verify stored arrays and cohort rules. They do not establish human usefulness or identify exposure propensities.')
    (out/'independent_checks.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':verify()
