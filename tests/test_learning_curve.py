"""Verify manual controls using tiny synthetic data, never the real course dataset."""

import json
from pathlib import Path

import pandas as pd
import pytest

import main
from test_samples import raw_experiment
from session_rerank.samples import prepare_samples, file_hash
from session_rerank.expanded_data import reserve_expanded
from session_rerank.expanded_training import training_settings
from session_rerank.learning_curve import open_learning_curve_run, query_target


@pytest.fixture
def learning_config(raw_experiment):
    _, protocol = raw_experiment
    pilot = prepare_samples(protocol)
    config = json.loads((Path(__file__).parents[1] / 'config/learning_curve.json').read_text())
    config.update(pilot_directory=str(pilot), output_directory=str(protocol.parent / 'curve'),
                  query_batch_size=2, purchase_gain_sensitivity=[], validation_only=True, top_k=2)
    config['reranking']['relevance_slots'] = 1
    config['reranking']['k1'] = 2
    config['training'].update(num_leaves=[3], min_data_in_leaf=[1], num_boost_round=3,
                              early_stopping_rounds=2, log_evaluation_period=1,
                              max_bin=31, learning_rate=.125, lambda_l2=.5)
    path = protocol.parent / 'manual.json'
    path.write_text(json.dumps(config))
    return path, config


@pytest.mark.parametrize('target', ['0', '-1', 'oops', '2.5', True])
def test_invalid_manual_targets_are_rejected(target):
    with pytest.raises(ValueError, match='positive integer or all'):
        query_target(target)


def test_manual_samples_are_nested_and_open_only_train_validation(learning_config):
    _, config = learning_config
    sets, validations = [], []
    for target in (1, 4, None):
        _, sessions, metadata, _ = reserve_expanded(config | {'training_target_queries': target})
        sets.append(set(metadata.loc[metadata.split.eq('train'), 'queryId']))
        validations.append(set(metadata.loc[metadata.split.eq('validation'), 'queryId']))
        assert set(sessions.split) == {'train', 'validation'}
        counts = metadata.groupby('session_key').size()
        assert counts.equals(sessions.set_index('session_key').query_count.reindex(counts.index).rename(None))
    assert sets[0] <= sets[1] <= sets[2] and len(sets[2]) == 5
    assert validations[0] == validations[1] == validations[2]
    with pytest.raises(ValueError, match='only 5 are eligible'):
        reserve_expanded(config | {'training_target_queries': 6})


def test_hyperparameters_are_validated_before_training(learning_config):
    _, config = learning_config
    assert training_settings(config)['learning_rate'] == .125
    for update in ({'learning_rate': 0}, {'num_leaves': 15}, {'max_bin': 1}, {'typo': 5}):
        with pytest.raises(ValueError):
            training_settings(config | {'training': config['training'] | update})


def test_manual_training_compares_sizes_resumes_and_forwards_json_settings(learning_config, monkeypatch, capsys):
    path, config = learning_config
    from session_rerank import expanded_training
    calls = []
    original = expanded_training.lgb.train
    def recorded_train(params, *args, **kwargs):
        calls.append((params.copy(), kwargs['num_boost_round']))
        return original(params, *args, **kwargs)
    monkeypatch.setattr(expanded_training.lgb, 'train', recorded_train)
    small = main.manual_training_main(path, queries='1')
    large = main.manual_training_main(path, queries='all')
    assert len(calls) == 8  # Four nominal model variants at each size.
    assert all(params['learning_rate'] == .125 and params['num_leaves'] == 3 and
               params['min_data_in_leaf'] == 1 and params['max_bin'] == 31 and
               params['lambda_l2'] == .5 and rounds == 3 for params, rounds in calls)
    for folder in (small, large):
        assert not (folder / 'selection_features').exists()
        assert not (folder / 'selection_choice.json').exists()
        assert not (folder / 'confirmation_opened.json').exists()
        assert not (folder / 'confirmation.sqlite').exists()
        assert pd.read_parquet(folder / 'confirmation_ids.parquet').empty
        assert json.loads((folder / 'objective.json').read_text())['primary'] == 'click_ndcg'
    measured = pd.read_parquet(small / 'validation_reranking.parquet')
    assert measured.top_k.eq(2).all()
    assert measured.topk_item_ids.map(lambda value: len(json.loads(value))).eq(2).all()
    assert set(measured.policy_id) == {'relevance', 'configured'}
    assert measured[measured.policy_id.eq('configured')].candidate_pool_size.eq(2).all()
    assert measured[measured.policy_id.eq('relevance')].candidate_pool_size.eq(3).all()
    assert measured[measured.policy_id.eq('configured')].alpha.isna().all()
    assert {'head_click_ndcg', 'head_gain_ndcg', 'tail_diversity_ild'} <= set(measured.columns)
    for _, group in measured.groupby(['model', 'query_id']):
        lists = group.set_index('policy_id').topk_item_ids.map(json.loads)
        assert lists['relevance'][:1] == lists['configured'][:1]
    comparison = pd.read_csv(Path(config['output_directory']) / 'comparison.csv')
    assert comparison.training_queries.nunique() == 2
    assert comparison.top_k.eq(2).all()
    assert comparison.reranked_click_ndcg.notna().all()
    assert comparison.validation_labels_sha256.nunique() == 1
    assert (Path(config['output_directory']) / 'learning_curve.png').exists()
    before = file_hash(small / 'click_session.txt')
    main.manual_training_main(path, queries='1')
    assert len(calls) == 8 and file_hash(small / 'click_session.txt') == before
    assert 'trial 1/1' in capsys.readouterr().out
    # Reproduce native Windows receipts on Linux. Only the comparison's path
    # spelling changes; model files, query metadata and their hashes stay intact.
    receipt_path = small / 'features_complete.json'
    receipt = json.loads(receipt_path.read_text())
    receipt['files'] = {name.replace('/', '\\'): digest for name, digest in receipt['files'].items()}
    receipt_path.write_text(json.dumps(receipt))
    from session_rerank.learning_curve import compare_learning_sizes
    compare_learning_sizes(Path(config['output_directory']))
    assert len(calls) == 8 and file_hash(small / 'click_session.txt') == before



def test_changing_hyperparameters_needs_a_separate_experiment(learning_config):
    path, config = learning_config
    _, folder, _ = open_learning_curve_run(path, queries='1')
    config['training']['learning_rate'] = .02
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match='different|differ'):
        open_learning_curve_run(path, queries='1')
    _, other, _ = open_learning_curve_run(path, queries='1', curve_root=path.parent / 'other-curve')
    assert other != folder


def test_cli_dispatches_manual_training_without_starting_it(monkeypatch):
    calls = []
    monkeypatch.setattr(main, 'manual_training_main', lambda *args: calls.append(args))
    main.cli(['train', '--queries', '2000'])
    assert calls == [(Path('config/learning_curve.json'), '2000', None)]
    calls.clear()
    main.cli(['train', '--queries', '10000', '--training-config', 'config/custom.json'])
    assert calls == [(Path('config/custom.json'), '10000', None)]
    with pytest.raises(SystemExit):
        main.cli(['train'])
    with pytest.raises(SystemExit):
        main.cli(['train', '--training-config', 'config/custom.json'])
    with pytest.raises(SystemExit):
        main.cli(['train', '--queries', '0'])
    with pytest.raises(SystemExit):
        main.cli(['expand', '--queries', '2000'])


def test_manual_entry_requires_an_explicit_size_before_touching_files(tmp_path):
    with pytest.raises(ValueError, match='--queries'):
        main.manual_training_main(tmp_path/'does-not-exist.json')
    assert not list(tmp_path.iterdir())


def test_explicit_pilot_command_preserves_legacy_training(monkeypatch):
    calls = []
    monkeypatch.setattr(main, 'main', lambda *args, **kwargs: calls.append((args, kwargs)))
    main.cli(['pilot-train'])
    assert calls[0][1]['train'] is True
