import contractions
import re
from tqdm import tqdm

def expand_contractions(text):
    return contractions.fix(text)

def remove_symbols(text):
    pattern = r'[\$%^&#@*]'
    return re.sub(pattern, '', text)

def clean_dataset(dataset):
    def preprocess(entry):
        # Expand contractions
        entry['dialogue'] = expand_contractions(entry['dialogue'])
        entry['summary'] = expand_contractions(entry['summary'])
        
        # Remove specified symbols
        entry['dialogue'] = remove_symbols(entry['dialogue'])
        entry['summary'] = remove_symbols(entry['summary'])
        return entry
    
    dataset['train'] = dataset['train'].map(preprocess)
    dataset['validation'] = dataset['validation'].map(preprocess)
    dataset['test'] = dataset['test'].map(preprocess)
    
    return dataset


def tokenize(text):
    # Convert text to lowercase and find all word characters
    return re.findall(r'\b\w+\b', text.lower())

def remove_oov_words(dataset):
    vocab = set()
    for item in tqdm(dataset['train'], desc='Building vocabulary'):
        tokens_dialogue = tokenize(item['dialogue'])
        tokens_summary = tokenize(item['summary'])
        vocab.update(tokens_dialogue)
        vocab.update(tokens_summary)
        
    def replace_oov_words(text, vocab):
        tokens = tokenize(text)
        tokens = [token if token in vocab else '<unk>' for token in tokens]
        return ' '.join(tokens)
    
    def preprocess(entry):
        entry['dialogue'] = replace_oov_words(entry['dialogue'], vocab)
        entry['summary'] = replace_oov_words(entry['summary'], vocab)
        return entry
    
    dataset['train'] = dataset['train'].map(preprocess)
    dataset['validation'] = dataset['validation'].map(preprocess)
    dataset['test'] = dataset['test'].map(preprocess)
    
    return dataset

def wsd(dataset, splits: list[str] = ['train', 'validation', 'test']):
    from pywsd import disambiguate
    def preprocess(entry):
        t = disambiguate(entry['dialogue'])
        entry['dialogue'] = ' '.join([(word if not synset else synset.name()) for word, synset in t])
        return entry
        
    for split in splits:
        dataset[split] = dataset[split].map(preprocess)
    
    return dataset