from torch.utils.data import Dataset
import numpy as np


class CustomDataset(Dataset):
    def __init__(self, df, tokenizer, train_stage=True):
        super().__init__()
        self.df = df
        self.tokenizer = tokenizer
        self.train_stage = train_stage

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        instance = self.df.iloc[idx]
        code1 = instance['code1']
        code2 = instance['code2']

        tokenized = self.tokenizer(code1,
                                   code2,
                                   return_tensors="np",
                                   max_length=512,
                                   padding='max_length',
                                   truncation=True,
                                   add_special_tokens=True,
                                   return_token_type_ids=True
                                   )
        if self.train_stage:
            label = instance['similar']

            return {'input_ids': tokenized['input_ids'],
                    'attention_mask': tokenized['attention_mask'],
                    'token_type_ids': tokenized['token_type_ids'],
                    'labels': np.array(label)}
        else:
            return {'input_ids': tokenized['input_ids'],
                    'attention_mask': tokenized['attention_mask'],
                    'token_type_ids': tokenized['token_type_ids']}


def __test():
    import pandas as pd
    from utils.common.project_paths import GetPaths
    from custom_tokenizer.custom_tokenizer import get_tokenizer
    from torch.utils.data import DataLoader
    df = pd.read_csv(GetPaths.get_data_folder('train.tsv'), delimiter='\t')
    tokenizer = get_tokenizer()
    w = 'ひ'
    a = tokenizer.encode(w)
    b = tokenizer.decode(a)
    d_set = DataLoader(CustomDataset(df, tokenizer))
    for i in d_set:
        print(i)


if __name__ == '__main__':
    __test()
