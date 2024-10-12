#imports
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import torch
from torch.utils.data import Dataset
from transformers import BartForConditionalGeneration, BartTokenizer
from tqdm import tqdm

# import the metrics
from rouge_score import rouge_scorer
from bert_score import BERTScorer
#from evaluation.readability import flesch_kincaid_grade_level, dale_chall_readability_score, coleman_liau_index, lens

import json
import numpy as np

def _load_dataset() -> Dataset:
    from datasets import load_dataset
    dataset = load_dataset("Samsung/samsum", "samsum", trust_remote_code=True)
    
    #shuffle the train and validation sets
    dataset['train'] = dataset['train'].shuffle()
    dataset['validation'] = dataset['validation'].shuffle()
    
    return dataset

def generate_summaries(model, tokenizer, dataset, limit: int = None, batch_size: int = 8) -> dict:
    limit = len(dataset) if limit is None else min(limit, len(dataset))
    
    summaries = np.empty(limit, dtype=object)

    # Calculate number of batches
    num_batches = (limit + batch_size - 1) // batch_size
    
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient calculations
        for batch_idx in tqdm(range(num_batches), desc="Generating summaries"):
            # Get batch indices
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, limit)
            dialogues = [dataset[i]["dialogue"] for i in range(start_idx, end_idx)]
            
            # Tokenize batch
            inputs = tokenizer(dialogues, return_tensors="pt", max_length=512, truncation=True, padding=True).to(model.device)
            
            # Generate summaries for the batch
            with torch.amp.autocast("cuda"):
                predicted_summaries = model.generate(
                    inputs.input_ids, 
                    max_length=150, 
                    num_beams=4, 
                    early_stopping=True
                )
            
            # Decode summaries and store them
            decoded_summaries = [tokenizer.decode(summary, skip_special_tokens=True) for summary in predicted_summaries]
            summaries[start_idx:end_idx] = decoded_summaries

    return {
        'generated_summaries': list(summaries),
        'dialogue': [entry['dialogue'] for entry in dataset],
        'summary': [entry['summary'] for entry in dataset]
    }

def all_summaries_of_sufficient_length(generated_summaries: np.ndarray, length: int = 100) -> bool:
    all_good = True
    
    for summary in generated_summaries:
        #get the number of words in the summary
        summary_length = len(summary.split())
        
        if summary_length < length:
            all_good = False
            print(f"{summary_length} words => Summary {summary} is too short.")
            
    return all_good
    
    
#compute all metrics for the test set and save them to a json file
#the dataset is a list of dicts with 'article', 'summary', 'section_headings', 'keywords', 'year', 'title'
#add a progress bar

def compute_metrics(summaries: list[str], dataset):
    metrics = {"relevance": {"rouge": [], "bert": []}, "readability": {"fkgl": [], "dcrs": [], "cli": []}}
    
    bert = BERTScorer(lang="en", model_type="bert-base-uncased")
    rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    for i in tqdm(range(len(summaries)), desc="Computing metrics"):
        #compute the predicted summary from the article
        expert_summary = dataset[i]["summary"]
        predicted_summary = summaries[i]
        
        #compute the relevance scores
        metrics["relevance"]["rouge"].append(rouge.score(expert_summary, predicted_summary))
        metrics["relevance"]["bert"].append(bert.score([predicted_summary], [expert_summary])[0].item())
        
        # #compute the readability scores
        # r = Readability(predicted_summary)
        # try:
        #     #metrics["readability"]["fkgl"].append(r.flesch_kincaid().score)
        #     metrics["readability"]["dcrs"].append(r.dale_chall().score)
        #     metrics["readability"]["cli"].append(r.coleman_liau().score)
        # except ReadabilityException:
        #     logging.warning(f"Readability exception for (len: {len(predicted_summary.split())}) {predicted_summary}")
        #     metrics["readability"]["fkgl"].append(0)
        #     metrics["readability"]["dcrs"].append(0)
        #     metrics["readability"]["cli"].append(0)
            
        
    return metrics

if __name__ == "__main__":
    #set up logging
    import logging
    logging.basicConfig(level=logging.INFO)
    
    #set up argparse
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="basic", help="Dataset to use")
    parser.add_argument("--model", type=str, help="Name of the model to evaluate", required=True)
    parser.add_argument("--model-folder", type=str, help="Folder where the models are stored", default="./models")
    parser.add_argument("--device", type=str, help="Device to use for evaluation", default="cpu")
    parser.add_argument("--batch-size", type=int, help="Batch size for evaluation", default=16)
    parser.add_argument("--tokenizer", type=str, help="Path to the tokenizer directory", required=False)
    args = parser.parse_args()
    
    dataset = _load_dataset()
    if args.dataset == "basic":
        pass
    elif args.dataset == "cleaned":
        from samsum_dataset.preprocessing import clean_dataset
        dataset = clean_dataset(dataset)
    elif args.dataset == "oov":
        from samsum_dataset.preprocessing import remove_oov_words, clean_dataset
        dataset = clean_dataset(dataset)
        dataset = remove_oov_words(dataset)
    elif args.dataset == "wsd":
        from samsum_dataset.preprocessing import wsd
        dataset = wsd(dataset, splits=['test'])
    
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is not available. Please use a different device.")
    
    model = BartForConditionalGeneration.from_pretrained(f"{args.model_folder}/{args.model}")
    if args.tokenizer is None:
        tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
    else:
        tokenizer = BartTokenizer.from_pretrained(args.tokenizer)
        model.resize_token_embeddings(len(tokenizer))
    # Resize the model's embeddings
    model.to(args.device)
    
    generated_summaries = generate_summaries(model, tokenizer, dataset['test'], limit=None, batch_size=args.batch_size)
    
    with open(f"{args.model_folder}/{args.model}/summaries.json", "w") as f:
        json.dump(generated_summaries, f)