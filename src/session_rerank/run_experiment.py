"""Reproduce the RecBole-backed submission experiment from the repository root."""
import argparse
from pathlib import Path
import subprocess
import sys


def run(*arguments):
    print('Running:',sys.executable,*arguments,flush=True)
    subprocess.run([sys.executable,*arguments],check=True)


def prepare():
    if not Path('data/raw/train-queries.csv').exists():run('scripts/download_data.py')
    if not Path('data/processed/queries_metadata.parquet').exists():
        run('-m','session_rerank.prepare')
    if not Path('results/model.txt').exists():
        run('-m','session_rerank.benchmark','--stage','development')
    if not Path('data/processed_v2/daily_counts.npz').exists():
        run('-m','session_rerank.optimization_data','prepare')
    for split in ['validation','train']:
        if not Path(f'data/processed_v2/{split}/manifest.json').exists():
            run('-m','session_rerank.optimization_data','features','--split',split)
        if not Path(f'data/processed_v2/{split}/features_views.npy').exists():
            run('-m','session_rerank.view_features','--split',split)


def train():
    experiments=[
        ('baseline_20k',['--queries','20000','--features','baseline']),
        ('enhanced_20k',['--queries','20000']),('enhanced_60k',[]),
        ('views_20k',['--queries','20000','--features','views']),
        ('views_60k',['--features','views']),
        ('enhanced_60k_63',['--leaves','63']),
        ('views_60k_63',['--features','views','--leaves','63'])]
    for name,arguments in experiments:
        if not Path(f'results/optimization/{name}.json').exists():
            run('-m','session_rerank.optimize_ranker','--name',name,*arguments)
    if not Path('results/optimization/recbole_gru4rec.pth').exists():
        run('-m','session_rerank.recbole_model','train')


def confirm():
    if not Path('results/optimization/frozen_selection.json').exists():
        raise RuntimeError('Run the freeze stage before opening confirmation outcomes')
    if Path('results/optimization/confirmation_per_query.npz').exists():
        raise RuntimeError('Confirmation already evaluated; preserve and read the recorded results')
    if not Path('data/processed_v2/confirmation/manifest.json').exists():
        run('-m','session_rerank.optimization_data','features','--split','confirmation')
    if not Path('data/processed_v2/confirmation/features_views.npy').exists():
        run('-m','session_rerank.view_features','--split','confirmation')
    run('-m','session_rerank.optimization_report','confirmation')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['prepare','train','freeze','confirm','all'])
    stage=parser.parse_args().stage
    if stage in ['prepare','all']:prepare()
    if stage in ['train','all']:train()
    if stage in ['freeze','all']:run('-m','session_rerank.optimization_report','freeze')
    if stage in ['confirm','all']:confirm()


if __name__=='__main__':main()
