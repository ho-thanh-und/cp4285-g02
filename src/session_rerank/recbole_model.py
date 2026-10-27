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
    result=np.empty(len(offsets)-1,dtype=float)
    discount=1/np.log2(np.arange(k)+2)
    for j,(start,end) in enumerate(zip(offsets[:-1],offsets[1:])):
        order=np.argsort(-scores[start:end],kind='stable')[:k]
        n=int(labels[start:end].sum())
        result[j]=np.dot(labels[start:end][order],discount[:len(order)])/discount[:min(n,k)].sum()
    return result


def make_sequences(raw_prefixes, mapping):
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
    """Equivalent to RecBole GRU4Rec.predict, encoding each prefix only once.

    Empty/entirely-OOV histories fall back to past-week click popularity.
    Unknown candidates receive a score below known candidates for a usable prefix.
    Every supplied candidate remains in evaluation, including cold products.
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
    def __init__(self,config,model,mapping):
        super().__init__(config,model)
        self.mapping=mapping
        self.history=[]

    def _valid_epoch(self,valid_data,show_progress=False):
        scores=candidate_scores(self.model,self.mapping,valid_data)
        value=float(ndcg_by_query(scores,valid_data['y'],valid_data['offsets']).mean())
        self.history.append(value)
        print(f'RecBole epoch {len(self.history)} validation full-candidate NDCG@10={value:.6f}',flush=True)
        return value,{'ndcg@10':value}


def train():
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
    torch.set_num_threads(4)
    checkpoint=torch.load(OUT/'recbole_gru4rec.pth',map_location='cpu',weights_only=False)
    config=Config(model='GRU4Rec',dataset='diginetica_session',config_file_list=[checkpoint['config_file']])
    class DatasetShape:
        def num(self,field):
            return checkpoint['n_items'] if field==config['ITEM_ID_FIELD'] else 1
    model=get_model('GRU4Rec')(config,DatasetShape()).to('cpu')
    model.load_state_dict(checkpoint['state_dict'])
    return model,checkpoint['mapping']


def main():
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
