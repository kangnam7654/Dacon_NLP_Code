from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from dataload.dataset import CustomDataset


class LitDataLoader(LightningDataModule):
    def __init__(self, train_df=None, valid_df=None, test_df=None, tokenizer=None):
        super().__init__()
        self.train_df = train_df
        self.valid_df = valid_df
        self.test_df = test_df
        self.tokenizer = tokenizer

    def __create_dataset(self, mode='train'):
        if mode == 'train':
            return CustomDataset(df=self.train_df, tokenizer=self.tokenizer, train_stage=True)
        elif mode == 'valid':
            return CustomDataset(df=self.valid_df, tokenizer=self.tokenizer, train_stage=True)
        elif mode == 'test':
            return CustomDataset(df=self.test_df, tokenizer=self.tokenizer, train_stage=False)

    def train_dataloader(self, batch_size=8, shuffle=True, drop_last=False, num_workers=4):
        dataset = self.__create_dataset('train')
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)

    def val_dataloader(self, batch_size=8, shuffle=False, drop_last=False, num_workers=4):
        dataset = self.__create_dataset('valid')
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)

    def predict_dataloader(self, batch_size=8, num_workers=4):
        dataset = self.__create_dataset(mode='test')
        return DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers)

    def test_dataloader(self, batch_size=8, num_workers=4):
        dataset = self.__create_dataset(mode='test')
        return DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers)


if __name__ == '__main__':
    pass