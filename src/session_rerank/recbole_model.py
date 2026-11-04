"""Train RecBole GRU4Rec, validating on the project's complete query candidates.

RecBole handles atomic data, sequence augmentation, negative sampling, model
loss, optimizer and checkpointing. Only the validation adapter is overridden:
default next-item/full-catalog scores would change our query-ranking task.
"""
import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.trainer import Trainer
from recbole.utils import get_model, init_seed

from .optimization_data import load_features

OUT=Path('results/optimization')


def ndcg_by_query(scores, labels, offsets, k=10):
    """Return one NDCG@k relevance score for each query's candidate list.

    scores and binary labels are flattened across queries. offsets gives
    each query's start/end rows. Sort within those boundaries so candidates
    from different searches are never mixed. Queries must have positives.
    """
    result=np.empty(len(offsets)-1,dtype=float)
    discount=1/np.log2(np.arange(k)+2)
    for j,(start,end) in enumerate(zip(offsets[:-1],offsets[1:])):
        order=np.argsort(-scores[start:end],kind='stable')[:k]
        n=int(labels[start:end].sum())
        result[j]=np.dot(labels[start:end][order],discount[:len(order)])/discount[:min(n,k)].sum()
    return result


def make_sequences(raw_prefixes, mapping):
    """Translate original product-history IDs into the neural model's IDs.

    raw_prefixes has one oldest-first, zero-padded history per query. mapping
    comes from the trained RecBole dataset. Drop unknown products; return
    padded sequences, safe lengths, and a flag for histories with known items.
    Empty histories use length 1 for tensor compatibility but remain flagged.
    """
    sequence=np.zeros(raw_prefixes.shape,dtype=np.int64)
    lengths=np.ones(len(sequence),dtype=np.int64)
    usable=np.zeros(len(sequence),dtype=bool)
    for i,prefix in enumerate(raw_prefixes):
        known=[mapping.get(str(int(item)),0) for item in prefix if item]
        known=[item for item in known if item]
        if known:
            sequence[i,:len(known)]=known
            lengths[i]=len(known)
            usable[i]=True
    return sequence,lengths,usable


def candidate_scores(model, mapping, data, batch_size=256):
    """Return one neural relevance score per supplied candidate row.

    model and mapping come from training; data supplies histories, candidates,
    query offsets, and fallback features. Encode each history once, then take
    its dot product with each candidate's learned item embedding.
    Empty/entirely unknown histories fall back to past-week click popularity.
    For usable histories, unknown candidates score below known candidates;
    if all candidates are unknown, use popularity. Keep every candidate.
    """
    model.eval()
    device=next(model.parameters()).device
    seq,lengths,usable=make_sequences(data['prefixes'],mapping)
    offsets=data['offsets']
    scores=np.empty(int(offsets[-1]),dtype=np.float32)
    with torch.no_grad():
        for begin in range(0,len(seq),batch_size):
            end=min(begin+batch_size,len(seq))
            encoded=model.forward(torch.as_tensor(seq[begin:end],device=device),
                                  torch.as_tensor(lengths[begin:end],device=device))
            max_candidates=int(np.diff(offsets[begin:end+1]).max())
            candidate_tokens=np.zeros((end-begin,max_candidates),dtype=np.int64)
            for local,j in enumerate(range(begin,end)):
                items=data['items'][offsets[j]:offsets[j+1]]
                candidate_tokens[local,:len(items)]=[mapping.get(str(int(x)),0) for x in items]
            tensor=torch.as_tensor(candidate_tokens,device=device)
            values=(model.item_embedding(tensor)*encoded[:,None,:]).sum(dim=-1).cpu().numpy()
            for local,j in enumerate(range(begin,end)):
                start,stop=offsets[j:j+2]
                n=stop-start
                if not usable[j]:
                    scores[start:stop]=data['X'][start:stop,13]
                else:
                    v=values[local,:n].copy()
                    known=candidate_tokens[local,:n]>0
                    if known.any():
                        v[~known]=v[known].min()-max(1,float(np.ptp(v[known])))
                    else:
                        v=data['X'][start:stop,13].copy()
                    scores[start:stop]=v
    return scores


class CandidateValidationTrainer(Trainer):
    """RecBole trainer that validates on our query-ranking task."""
    def __init__(self,config,model,mapping):
        """Keep RecBole training behavior and remember item IDs and validation history."""
        super().__init__(config,model)
        self.mapping=mapping
        self.history=[]

    def _valid_epoch(self,valid_data,show_progress=False):
        """Score our full query candidate lists after an epoch and return mean NDCG.

        Replaces RecBole's usual next-item validation, which would measure a
        different task. The trainer uses this value to choose its checkpoint.
        """
        scores=candidate_scores(self.model,self.mapping,valid_data)
        value=float(ndcg_by_query(scores,valid_data['y'],valid_data['offsets']).mean())
        self.history.append(value)
        print(f'RecBole epoch {len(self.history)} validation full-candidate NDCG@10={value:.6f}',flush=True)
        return value,{'ndcg@10':value}


def train():
    """Train the original GRU4Rec model using RecBole's BPR loss configuration.

    Load daily click sequences and validate against the project's candidate
    lists after each epoch. Save the best checkpoint, item-ID mapping,
    validation scores, and training summary under results/optimization.
    BPR learns to prefer a clicked item over a sampled negative item.
    """
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'recbole_gru4rec.pth').exists():
        raise RuntimeError('RecBole model already exists; preserve the recorded run')
    torch.set_num_threads(4)
    config=Config(model='GRU4Rec',dataset='diginetica_session',config_file_list=['config/recbole_gru4rec.yaml'])
    init_seed(config['seed'],config['reproducibility'])
    logging.basicConfig(level=logging.INFO,format='%(message)s')
    dataset=create_dataset(config)
    train_data,_,_=data_preparation(config,dataset)
    model=get_model(config['model'])(config,train_data.dataset).to(config['device'])
    mapping=train_data.dataset.field2token_id[config['ITEM_ID_FIELD']]
    trainer=CandidateValidationTrainer(config,model,mapping)
    validation=load_features('validation')
    trainer.fit(train_data,valid_data=validation,saved=True,show_progress=False)
    # This checkpoint was just produced locally by RecBole's Trainer.
    checkpoint=torch.load(trainer.saved_model_file,map_location='cpu',weights_only=False)
    model.load_state_dict(checkpoint['state_dict'])
    payload=dict(state_dict=model.state_dict(),mapping=dict(mapping),config_file='config/recbole_gru4rec.yaml',
                 best_epoch=int(checkpoint['epoch']),validation_history=trainer.history,
                 n_items=int(model.n_items),dataset_items=int(train_data.dataset.item_num))
    torch.save(payload,OUT/'recbole_gru4rec.pth')
    np.save(OUT/'gru4rec_validation.npy',candidate_scores(model,mapping,validation))
    summary={key:value for key,value in payload.items() if key not in ['state_dict','mapping']}
    summary.update(model_class=type(model).__module__+'.'+type(model).__name__,
                   trainer_class='RecBole Trainer with same-task validation adapter',
                   training_examples=len(train_data.dataset))
    (OUT/'recbole_training.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


def load_model():
    """Restore the saved GRU4Rec weights on CPU and return (model, ID mapping).

    Only the item count and saved configuration are needed to rebuild the
    model shape; the full training dataset does not need to be loaded.
    """
    torch.set_num_threads(4)
    checkpoint=torch.load(OUT/'recbole_gru4rec.pth',map_location='cpu',weights_only=False)
    config=Config(model='GRU4Rec',dataset='diginetica_session',config_file_list=[checkpoint['config_file']])
    class DatasetShape:
        """Minimal dataset interface needed to rebuild the saved model."""
        def num(self,field):
            """Return the saved item-vocabulary size, or 1 for other requested fields."""
            return checkpoint['n_items'] if field==config['ITEM_ID_FIELD'] else 1
    model=get_model('GRU4Rec')(config,DatasetShape()).to('cpu')
    model.load_state_dict(checkpoint['state_dict'])
    return model,checkpoint['mapping']


def main():
    """Run the original GRU4Rec training/scoring command with confirmation guards."""
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['train','score'])
    parser.add_argument('--split',choices=['validation','confirmation'],default='validation')
    args=parser.parse_args()
    if args.action=='train':train()
    else:
        if args.split=='confirmation' and not (OUT/'frozen_selection.json').exists():
            raise RuntimeError('Freeze selection before confirmation scoring')
        if args.split=='confirmation':
            from .policies import file_hash
            frozen=json.loads((OUT/'frozen_selection.json').read_text())
            if file_hash(OUT/'recbole_gru4rec.pth')!=frozen['recbole_checkpoint_sha256']:
                raise RuntimeError('RecBole checkpoint differs from the frozen selection')
        output=OUT/f'gru4rec_{args.split}.npy'
        if output.exists():raise RuntimeError('Predictions already cached; reuse the existing file')
        model,mapping=load_model()
        np.save(output,candidate_scores(model,mapping,load_features(args.split)))


if __name__=='__main__':main()
