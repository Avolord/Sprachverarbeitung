#imports
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import logging
import os

from tokenizers import ByteLevelBPETokenizer

def get_training_corpus(dataset):
    for split in ['train', 'validation', 'test']:
        for item in dataset[split]:
            yield item['dialogue']
            yield item['summary']

def train_custom_tokenizer(dataset):
    from tokenizers import ByteLevelBPETokenizer

    if not os.path.isdir("tokenizer"):
        # Initialize the tokenizer
        tokenizer = ByteLevelBPETokenizer()
        
        # Train the tokenizer
        tokenizer.train_from_iterator(
            get_training_corpus(dataset),
            vocab_size=52000,
            min_frequency=2,
            special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
        )
        
        # Save the tokenizer files
        tokenizer.save_model("tokenizer")
        print("Custom tokenizer trained and saved.")
    else:
        print("Custom tokenizer already exists.")
        
if __name__ == "__main__":
    #set up logging
    logging.basicConfig(level=logging.INFO)
    
    from datasets import load_dataset
    dataset = load_dataset("Samsung/samsum", "samsum", trust_remote_code=True)
    from samsum_dataset.preprocessing import clean_dataset, remove_oov_words
    dataset = clean_dataset(dataset)
    dataset = remove_oov_words(dataset)
    
    # Initialize the tokenizer
    tokenizer = ByteLevelBPETokenizer()

    # Train the tokenizer
    tokenizer.train_from_iterator(
        get_training_corpus(dataset),
        vocab_size=10000,
        min_frequency=2,
        special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
    )
    
    # Save the tokenizer files
    tokenizer.save_model("samsum_oov_tokenizer")