import json
import numpy as np
import pandas as pd
import os

def load_metrics(model_name: str) -> dict:
    # load the metrics json
    model_folder = f"models/{model_name}"

    with open(f"{model_folder}/metrics.json", "r") as f:
        metrics = json.load(f)
        
    return metrics

def reformat_rouge(metrics: dict, rouge_type: str) -> dict:
    # reformat the rouge metrics to a dictionary with precision, recall, and f1 values
    rouge = metrics['relevance']['rouge']
    
    result = np.array([metric[rouge_type] for metric in metrics['relevance']['rouge']])
    result = {'precision': result[:,0], 'recall': result[:,1], 'f1': result[:,2]}
    
    return result

def rouge_averages(rouge: dict) -> pd.DataFrame:
    rouge_1 = rouge['rouge_1']
    rouge_2 = rouge['rouge_2']
    rouge_l = rouge['rouge_l']
    
    averages_df = pd.DataFrame({

        'Rouge_1': [np.mean(values) for values in rouge_1.values()],

        'Rouge_2': [np.mean(values) for values in rouge_2.values()],

        'Rouge_L': [np.mean(values) for values in rouge_l.values()]

    }, index=['Precision', 'Recall', 'F1'])
    
    return averages_df

def get_trainer_states(model_name: str) -> list:
    # load the trainer states json
    model_folder = f"models/{model_name}"
    
    trainer_states = {}
    for checkpoint_folder in os.listdir(os.path.join(model_folder, "training")):
        if not checkpoint_folder.startswith("checkpoint-"):
            continue
        
        checkpoint_path = os.path.join(model_folder, "training", checkpoint_folder)
        
        #load the trainer state
        trainer_state = json.load(open(os.path.join(checkpoint_path, "trainer_state.json")))
        
        trainer_states[checkpoint_folder] = trainer_state
        
    return trainer_states

def get_trainer_state_of_latest_checkpoint(model_name: str) -> dict:
    # get the latest checkpoint folder
    model_folder = f"models/{model_name}"
    
    checkpoint_folders = os.listdir(os.path.join(model_folder, "training"))
    checkpoint_folders = [folder for folder in checkpoint_folders if folder.startswith("checkpoint-")]
    
    if len(checkpoint_folders) == 0:
        raise ValueError(f"No checkpoints found in {model_folder}/training")
        
    latest_checkpoint_folder = max(checkpoint_folders, key=lambda x: int(x.split("-")[1]))
    
    # load the trainer state
    trainer_state = json.load(open(os.path.join(model_folder, "training", latest_checkpoint_folder, "trainer_state.json")))
    
    return trainer_state

def get_loss_per_epoch(trainer_state: dict) -> pd.DataFrame:
    # get the loss per epoch for a trainer state
    
    loss = {'epoch': [], 'loss': []}

    for history in trainer_state['log_history']:
        if 'loss' in history.keys():
            loss['epoch'].append(history['epoch'])
            loss['loss'].append(history['loss'])
            
    return pd.DataFrame(loss)
