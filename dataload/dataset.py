from torch.utils.data import Dataset
import numpy as np


class CustomDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=256, ignore_index=-100, train_stage=True, translator=False):
        super().__init__()
        self.df = df
        self.tokenizer = tokenizer
        self.train_stage = train_stage
        self.max_len = max_len
        self.ignore_index = ignore_index
        self.translator = translator

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        instance = self.df.iloc[idx]

        input_ids = [self.tokenizer.bos_token_id]
        input_ids += self.tokenizer.encode(instance['원문'])
        input_ids.append(self.tokenizer.eos_token_id)
        input_ids = self.token_masking(input_ids)
        input_ids = self.add_padding_data(input_ids)

        label_ids = [self.tokenizer.bos_token_id]
        if self.translator:  # 번역기 학습
            label_ids += self.tokenizer.encode(instance['번역문'])
        else:  # base 학습
            label_ids += self.tokenizer.encode(instance['원문'])
        label_ids.append(self.tokenizer.eos_token_id)

        dec_input_ids = [self.tokenizer.eos_token_id]
        dec_input_ids += label_ids[:-1]
        dec_input_ids = self.add_padding_data(dec_input_ids)

        label_ids = self.add_padding_data(label_ids)

        return {'input_ids': np.array(input_ids),
                'decoder_input_ids': np.array(dec_input_ids),
                'labels': np.array(label_ids)}

    def token_masking(self, inputs):
        n_mask = np.random.randint(2)
        if n_mask != 0:
            idx = [i for i in range(len(inputs))]
            mask_position = np.random.choice(idx, n_mask)

            for i in mask_position:
                inputs[i] = self.tokenizer.mask_token_id
        return inputs

    def add_padding_data(self, inputs):
        if len(inputs) < self.max_len:
            pad = np.array([self.tokenizer.pad_token_id] * (self.max_len - len(inputs)))
            inputs = np.concatenate([inputs, pad])
        else:
            inputs = inputs[:self.max_len]
        return inputs

    def add_ignored_data(self, inputs):
        if len(inputs) < self.max_len:
            pad = np.array([self.ignore_index] * (self.max_len - len(inputs)))
            inputs = np.concatenate([inputs, pad])
        else:
            inputs = inputs[:self.max_len]
        return inputs


def test():
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
    test()
