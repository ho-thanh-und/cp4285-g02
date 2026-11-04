"""Additional real RecBole models; never access new confirmation during fitting."""
import argparse
import json
import logging
from pathlib import Path
import time
import numpy as np
import torch
from recbole.config import Config
from recbole.data import create_dataset,data_preparation
from recbole.utils import get_model,init_seed
from .recbole_model import CandidateValidationTrainer,candidate_scores
from .optimization_data import load_features

OUT=Path('results/iteration3')
VARIANTS={'gru_ce':('GRU4Rec','config/recbole_gru_ce.yaml'),
          'sas_ce':('SASRec','config/recbole_sas_ce.yaml')}


def configuration(variant,device='auto'):
    """Build a RecBole configuration for gru_ce or sas_ce and choose a device.

    Merge the common sequence settings with the selected model's YAML file.
    device='auto' uses CUDA, then Apple MPS, then CPU when available.
    Returns the configuration passed to dataset/model/trainer creation.
    """
    model,path=VARIANTS[variant]
    config=Config(model=model,dataset='diginetica_session',
                  config_file_list=['config/recbole_gru4rec.yaml',path])
    if device=='auto':
        chosen='cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    else:chosen=device
    config['device']=torch.device(chosen)
    return config


def train(variant,device='auto',smoke=False):
    """Train one GRU4Rec or SASRec model with cross-entropy (CE) loss.

    variant selects the model/configuration; device chooses the hardware.
    Validate on complete query candidate lists and save the best checkpoint,
    ID mapping, scores, and summary under results/iteration3. smoke=True
    instead runs five minibatches and reports loss without saving a model.
    CE learns the clicked target over the item vocabulary.
    """
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'frozen_selection.json').exists() or (OUT/'selection_summary.csv').exists():raise RuntimeError('Iteration 3 model selection is frozen')
    path=OUT/f'{variant}.pth'
    if path.exists():raise RuntimeError('Preserve the existing model run')
    torch.set_num_threads(4)
    config=configuration(variant,device)
    init_seed(config['seed'],config['reproducibility'])
    logging.basicConfig(level=logging.INFO,format='%(message)s')
    started=time.perf_counter()
    dataset=create_dataset(config)
    training,_,_=data_preparation(config,dataset)
    model=get_model(config['model'])(config,training.dataset).to(config['device'])
    mapping=training.dataset.field2token_id[config['ITEM_ID_FIELD']]
    if smoke:
        optimizer=torch.optim.Adam(model.parameters(),lr=config['learning_rate'])
        for j,batch in enumerate(training):
            optimizer.zero_grad()
            loss=model.calculate_loss(batch.to(config['device']))
            loss.backward();optimizer.step()
            if j==4:break
        if config['device'].type=='mps':torch.mps.synchronize()
        print(json.dumps(dict(smoke_batches=5,loss=float(loss.detach().cpu()),device=str(config['device']),seconds=time.perf_counter()-started)),flush=True)
        return
    trainer=CandidateValidationTrainer(config,model,mapping)
    validation=load_features('validation')
    trainer.fit(training,valid_data=validation,saved=True,show_progress=False)
    checkpoint=torch.load(trainer.saved_model_file,map_location='cpu',weights_only=False)
    model.load_state_dict(checkpoint['state_dict'])
    payload=dict(state_dict={key:value.cpu() for key,value in model.state_dict().items()},
                 mapping=dict(mapping),variant=variant,model_name=config['model'],n_items=model.n_items)
    torch.save(payload,path)
    scores=candidate_scores(model,mapping,validation)
    np.save(OUT/f'{variant}_validation.npy',scores)
    summary=dict(variant=variant,model=config['model'],loss='CE',device=str(config['device']),
        best_epoch=int(checkpoint['epoch'])+1,validation_history=trainer.history,
        best_validation_ndcg10=max(trainer.history),training_prefix_examples=len(training.dataset),
        seconds=time.perf_counter()-started)
    (OUT/f'{variant}.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


def score(variant,split):
    """Restore one neural variant and save scores for the requested split.

    variant='bpr' uses the earlier GRU4Rec; gru_ce/sas_ce use the newer
    checkpoints. Load candidate histories and score all supplied products.
    Confirmation is blocked until choices are frozen. Returns nothing;
    the score array is saved under results/iteration3.
    """
    if split=='confirmation' and not (OUT/'frozen_selection.json').exists():raise RuntimeError('Freeze before confirmation')
    from .iteration3 import load_data
    if variant=='bpr':
        from .recbole_model import load_model
        model,mapping=load_model()
        np.save(OUT/f'bpr_{split}.npy',candidate_scores(model,mapping,load_data(split)))
        return
    checkpoint=torch.load(OUT/f'{variant}.pth',map_location='cpu',weights_only=False)
    config=configuration(variant,'cpu')
    torch.set_num_threads(4)
    class Shape:
        """Minimal dataset interface for reconstructing a CE model from its checkpoint."""
        def num(self,field):
            """Return the item-vocabulary size recorded in the saved checkpoint."""
            return checkpoint['n_items']
    model=get_model(checkpoint['model_name'])(config,Shape())
    model.load_state_dict(checkpoint['state_dict'])
    np.save(OUT/f'{variant}_{split}.npy',candidate_scores(model,checkpoint['mapping'],load_data(split)))


def main():
    """Dispatch the neural train, five-batch smoke, or saved-model scoring command."""
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['train','score','smoke'])
    parser.add_argument('--variant',choices=list(VARIANTS)+['bpr'],required=True)
    parser.add_argument('--device',choices=['auto','cpu','mps','cuda'],default='auto')
    parser.add_argument('--split',choices=['validation','selection','confirmation'],default='confirmation')
    args=parser.parse_args()
    if args.action=='score':score(args.variant,args.split)
    else:
        if args.variant=='bpr':raise ValueError('The original BPR run is immutable; use score only')
        train(args.variant,args.device,args.action=='smoke')


if __name__=='__main__':main()
