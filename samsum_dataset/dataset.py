from torch.utils.data import Dataset

# class ElifeDataset(Dataset):
#     def __init__(self, data, tokenizer, prompt_template: str):
#         self.data = data
#         self.tokenizer = tokenizer
#         self.promp_template = prompt_template

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         item = self.data[idx]
#         article = self.promp_template.format(article=item['article'])
#         summary = item['summary']
#         inputs = self.tokenizer(article, return_tensors='pt', max_length=512, truncation=True)
#         targets = self.tokenizer(summary, return_tensors='pt', max_length=150, truncation=True)
#         return {
#             'input_ids': inputs.input_ids.flatten(),
#             'attention_mask': inputs.attention_mask.flatten(),
#             'labels': targets.input_ids.flatten()
#         }

class SAMSumDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        dialogue = item['dialogue']
        summary = item['summary']
        inputs = self.tokenizer(dialogue, return_tensors='pt', max_length=512, truncation=True)
        targets = self.tokenizer(summary, return_tensors='pt', max_length=128, truncation=True)
        return {
            'input_ids': inputs.input_ids.flatten(),
            'attention_mask': inputs.attention_mask.flatten(),
            'labels': targets.input_ids.flatten()
        }
        
def get_basic_datasets(dataset, tokenizer) -> tuple[SAMSumDataset, SAMSumDataset]:
    train_dataset = SAMSumDataset(dataset['train'], tokenizer)
    eval_dataset = SAMSumDataset(dataset['validation'], tokenizer)
    
    return train_dataset, eval_dataset
    
def get_cleaned_datasets(dataset, tokenizer) -> tuple[SAMSumDataset, SAMSumDataset]:
    from samsum_dataset.preprocessing import clean_dataset
    dataset = clean_dataset(dataset)
    
    train_dataset = SAMSumDataset(dataset['train'], tokenizer)
    eval_dataset = SAMSumDataset(dataset['validation'], tokenizer)
    
    return train_dataset, eval_dataset
    
def get_dataset_with_oov_words_removed(dataset, tokenizer) -> tuple[SAMSumDataset, SAMSumDataset]:
    from samsum_dataset.preprocessing import remove_oov_words, clean_dataset
    dataset = clean_dataset(dataset)
    dataset = remove_oov_words(dataset)
    
    train_dataset = SAMSumDataset(dataset['train'], tokenizer)
    eval_dataset = SAMSumDataset(dataset['validation'], tokenizer)
    
    return train_dataset, eval_dataset

def get_wsd_dataset(dataset, tokenizer, splits: list[str] = ['train', 'validation', 'test']) -> tuple[SAMSumDataset, SAMSumDataset]:
    from samsum_dataset.preprocessing import wsd
    dataset = wsd(dataset, splits)
    
    train_dataset = SAMSumDataset(dataset['train'], tokenizer)
    eval_dataset = SAMSumDataset(dataset['validation'], tokenizer) 
    
    return train_dataset, eval_dataset