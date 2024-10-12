import contractions
import re
from tqdm import tqdm

def expand_contractions(text):
    return contractions.fix(text)

def remove_symbols(text):
    pattern = r'[\$%^&#@*]'
    return re.sub(pattern, '', text)

def clean_dataset(dataset):
    for split in ['train', 'test', 'validation']:
        for item in dataset[split]:
            # Expand contractions
            item['article'] = expand_contractions(item['article'])
            item['summary'] = expand_contractions(item['summary'])
            
            # Remove specified symbols
            item['article'] = remove_symbols(item['article'])
            item['summary'] = remove_symbols(item['summary'])
            
    return dataset


def tokenize(text):
    # Convert text to lowercase and find all word characters
    return re.findall(r'\b\w+\b', text.lower())

def remove_oov_words(dataset):
    vocab = set()
    for item in tqdm(dataset['train'], desc='Building vocabulary'):
        tokens_article = tokenize(item['article'])
        tokens_summary = tokenize(item['summary'])
        vocab.update(tokens_article)
        vocab.update(tokens_summary)
        
    def replace_oov_words(text, vocab):
        tokens = tokenize(text)
        tokens = [token if token in vocab else 'unk' for token in tokens]
        return ' '.join(tokens)
    
    for split in tqdm(['validation', 'test'], desc=f'Replacing OOV words'):
        for item in tqdm(dataset[split], desc=f'Replacing OOV words in {split}'):
            item['article'] = replace_oov_words(item['article'], vocab)
            item['summary'] = replace_oov_words(item['summary'], vocab)
            
    return dataset