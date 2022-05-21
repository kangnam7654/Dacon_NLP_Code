from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from dataload.dataset import CustomDataset


class LitDataLoader(LightningDataModule):
    def __init__(self, train_df=None, valid_df=None, test_df=None, tokenizer=None, translator=False):
        super().__init__()
        self.train_df = train_df
        self.valid_df = valid_df
        self.test_df = test_df
        self.tokenizer = tokenizer
        self.translator = translator  # 번역기 학습 여부

    def __create_dataset(self, mode='train'):
        if mode == 'train':
            return CustomDataset(df=self.train_df, tokenizer=self.tokenizer, train_stage=True, translator=self.translator)
        elif mode == 'valid':
            return CustomDataset(df=self.valid_df, tokenizer=self.tokenizer, train_stage=True, translator=self.translator)
        elif mode == 'test':
            return CustomDataset(df=self.test_df, tokenizer=self.tokenizer, train_stage=False, translator=self.translator)

    def train_dataloader(self, batch_size=6, shuffle=True, drop_last=False, num_workers=4):
        dataset = self.__create_dataset('train')
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)

    def val_dataloader(self, batch_size=6, shuffle=False, drop_last=False, num_workers=4):
        dataset = self.__create_dataset('valid')
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)

    def predict_dataloader(self):
        raise Exception('predict_dataloader는 구현되어 있지 않습니다. test_dataloader를 이용해주세요.')

    def test_dataloader(self):  # 이번 프로젝트에선 사용 안하므로 삭제 대기
        dataset = self.__create_dataset('test')
        # 수정 요망
        return DataLoader(dataset, batch_size=self.cfg['TEST']['BATCH_SIZE'], shuffle=False, drop_last=False, num_workers=8, pin_memory=True)


if __name__ == '__main__':
    pass