#imports
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BartForConditionalGeneration, BartTokenizer
from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq
import pandas as pd
from tqdm.notebook import tqdm
import logging

from samsum_dataset.dataset import SAMSumDataset

def _load_dataset() -> Dataset:
    from datasets import load_dataset
    dataset = load_dataset("Samsung/samsum", "samsum", trust_remote_code=True)
    
    #shuffle the train and validation sets
    dataset['train'] = dataset['train'].shuffle()
    dataset['validation'] = dataset['validation'].shuffle()
    
    return dataset

def load_base_model_and_tokenizer(device: str = "cpu") -> tuple[BartForConditionalGeneration, BartTokenizer]:
    model_name = "facebook/bart-base"
    model = BartForConditionalGeneration.from_pretrained(model_name)
    tokenizer = BartTokenizer.from_pretrained(model_name)

    model.to(device)
    return model, tokenizer

def load_base_model_and_custom_tokenizer(device: str = "cpu", tokenizer_path: str = "tokenizer") -> tuple[BartForConditionalGeneration, BartTokenizer]:
    # Load the custom tokenizer
    tokenizer = BartTokenizer.from_pretrained(tokenizer_path)
    
    # Load the BART model
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
    
    # Resize the model's embeddings
    model.resize_token_embeddings(len(tokenizer))
    
    model.to(device)
    return model, tokenizer

def get_trainer(model: BartForConditionalGeneration, tokenizer: BartTokenizer, train_dataset: Dataset, eval_dataset: Dataset, output_dir: str, epochs: int = 3) -> Trainer:
    # Create a data collator specifically for sequence-to-sequence models
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,  # This ensures dynamic padding within each batch
        label_pad_token_id=-100  # Ensures labels are padded with -100 for loss calculation
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        logging_steps=10,
        fp16=True,
        save_strategy="epoch",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        
    )
    
    return trainer
    
if __name__ == "__main__":
    #set up logging
    logging.basicConfig(level=logging.INFO)
    
    #set up argparse for selecting which dataset to use
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="basic", help="Dataset to use")
    parser.add_argument("--training-output", type=str, default="./training", help="Output directory for training")
    parser.add_argument("--model-output", type=str, default="./model", help="Output directory for model")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs to train for")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for training. (cuda or cpu)")
    parser.add_argument("-y", action="store_true", help="save the model after training without asking")
    parser.add_argument("-n", action="store_true", help="don't save the model after training")
    parser.add_argument("--tokenizer", type=str, help="Path to the tokenizer directory", required=False)
    
    args = parser.parse_args()
    
    #load dataset
    logging.info("Loading raw dataset...")
    dataset = _load_dataset()
    logging.info(f"Loaded dataset with {len(dataset['train'])} training examples and {len(dataset['validation'])} validation examples.")
    
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is not available. Please use a different device.")
        
    
    #load model and tokenizer
    logging.info("Loading model and tokenizer...")
    #check if tokenizer arg is provided
    if args.tokenizer is None:
        model, tokenizer = load_base_model_and_tokenizer(args.device)
    else:
        model, tokenizer = load_base_model_and_custom_tokenizer(args.device, args.tokenizer)
    logging.info("Loaded model and tokenizer.")
    
    #load dataset
    if args.dataset == "basic":
        logging.info("Creating basic dataset...")
        from samsum_dataset.dataset import get_basic_datasets
        train_dataset, eval_dataset = get_basic_datasets(dataset, tokenizer)
        logging.info("Finished creating basic dataset.")
    elif args.dataset == "cleaned":
        logging.info("Creating cleaned dataset...")
        from samsum_dataset.dataset import get_cleaned_datasets
        train_dataset, eval_dataset = get_cleaned_datasets(dataset, tokenizer)
        logging.info("Finished creating cleaned dataset.")
    elif args.dataset == "oov":
        logging.info("Creating dataset with OOV words removed...")
        from samsum_dataset.dataset import get_dataset_with_oov_words_removed
        train_dataset, eval_dataset = get_dataset_with_oov_words_removed(dataset, tokenizer)
        logging.info("Finished creating dataset with OOV words removed.")
    elif args.dataset == "wsd":
        logging.info("Creating dataset with WSD disambiguation...")
        from samsum_dataset.dataset import get_wsd_dataset
        train_dataset, eval_dataset = get_wsd_dataset(dataset, tokenizer, splits=['train', 'validation'])
        logging.info("Finished creating dataset with WSD disambiguation.")
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    #train model
    logging.info("Training model...")
    trainer = get_trainer(model, tokenizer, train_dataset, eval_dataset, args.training_output, args.epochs)
    trainer.train()
    
    if args.y:
        model.save_pretrained(args.model_output) #save the model
        model = None
    elif args.n:
        model = None #don't save the model
    else:
        #ask the user if they want to save the model
        print("Do you want to save the model?")
        if input("y/n: ") == "y":
            model.save_pretrained(args.model_output)
            
        model = None
    