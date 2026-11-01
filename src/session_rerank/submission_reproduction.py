"""Reproduce the frozen primary tree through LightGBM's native training API.

The feature transformation is reconstructed here from the documented schema,
without importing the original transform or fit function. No model selection.
"""
import json
from pathlib import Path
import time
import hashlib
import lightgbm as lgb
import numpy as np
import pandas as pd

OUT=Path('results/submission/reproduction')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reconstruct(X):
    columns=list(range(17))+[21]  # delta=1, one selected history summary
    result=np.array(X[:,columns],dtype=np.float32,copy=True)
    confidence=np.minimum(np.float32(1),result[:,10]/np.float32(5))
    result[:,7]=result[:,7]*confidence
    for col in [8,9,17]:result[:,col]=confidence*result[:,col]+(1-confidence)*result[:,5]
    result[:,11]=result[:,11]*confidence
    result[:,16]=result[:,16]*confidence
    return result


def reproduce():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'receipt.json').exists():
        receipt=json.loads((OUT/'receipt.json').read_text())
        if not receipt['exact_prediction_replication']:
            raise RuntimeError('The cached reproduction failed; investigate before reusing it')
        if 'frozen_sha256' in receipt:
            assert digest('results/session_benchmark/frozen.json')==receipt['frozen_sha256']
            assert digest(OUT/'native_api_primary.txt')==receipt['model_sha256']
            assert digest('results/session_benchmark/predictions/scale_1.npy')==receipt['reference_predictions_sha256']
            return
    original=json.loads(Path('results/session_benchmark/frozen.json').read_text())['selected']
    assert original['tau']==5 and original['delta']==1 and original['binary'] and original['fraction']==1
    path=OUT/'native_api_primary.txt';started=time.perf_counter()
    if path.exists():model=lgb.Booster(model_file=str(path))
    else:
        X=np.load('data/session_benchmark/train/features.npy',mmap_mode='r');y=np.load('data/session_benchmark/train/labels.npy',mmap_mode='r')
        offsets=np.load('data/session_benchmark/train/offsets.npy')
        model=lgb.train(dict(objective='lambdarank',metric='ndcg',label_gain=[0,1,2,4],learning_rate=.05,num_leaves=31,
            min_data_in_leaf=150,lambda_l2=2,max_depth=-1,num_threads=4,seed=4285,deterministic=True,force_col_wise=True,verbosity=-1),
            lgb.Dataset(reconstruct(X),label=y,group=np.diff(offsets),weight=np.ones(len(y),dtype=np.float32)),
            num_boost_round=123)
        model.save_model(str(path));del X,y,offsets
    meta=pd.read_parquet('data/session_benchmark/validation/meta.parquet');offsets=np.load('data/session_benchmark/validation/offsets.npy')
    selected=np.flatnonzero(meta.fraction_rank.to_numpy()<.1)
    rows=np.concatenate([np.arange(offsets[j],offsets[j+1]) for j in selected])
    X=np.load('data/session_benchmark/validation/features.npy',mmap_mode='r')
    prediction=model.predict(reconstruct(X[rows]),num_threads=4)
    reference=np.load('results/session_benchmark/predictions/scale_1.npy')
    maximum=float(np.max(np.abs(prediction-reference)))
    result=dict(training_api='LightGBM native lgb.train; original used sklearn LGBMRanker',
        feature_transform='Reconstructed independently from the declared column schema and shrinkage equation',
        training_queries=int(len(np.load('data/session_benchmark/train/offsets.npy'))-1),
        validation_queries=len(selected),validation_candidates=len(rows),trees=123,max_prediction_difference=maximum,
        exact_prediction_replication=maximum<1e-10,seconds=time.perf_counter()-started,
        frozen_sha256=digest('results/session_benchmark/frozen.json'),model_sha256=digest(path),
        reference_predictions_sha256=digest('results/session_benchmark/predictions/scale_1.npy'),
        scope='Independent implementation and training API in the same machine/environment; not a claim of an independent human replication')
    (OUT/'receipt.json').write_text(json.dumps(result,indent=2)+'\n')
    if maximum>=1e-10:raise RuntimeError('Native reimplementation differs from the frozen primary tree; investigate before claiming reproduction')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':reproduce()
