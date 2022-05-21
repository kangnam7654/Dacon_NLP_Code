import pandas as pd

from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from model.bart_model import Bart
from utils.common.project_paths import GetPaths
from dataload.dataloader import LitDataLoader
from sklearn.model_selection import train_test_split


def main():
    # 모델 및 토크나이저

    # config, device = global_setting('cfg.yaml')
    model = Bart()  # 모델
    model.apply_ckpt('./ckpt/model.ckpt')
    tokenizer = model.tokenizer  # 토크나이저

    # 데이터 프레임
    csv = pd.read_csv(GetPaths.get_data_folder('train.tsv'), delimiter='\t')  # tsv 읽어오기
    csv = csv[:30000]
    train_df, valid_df = train_test_split(csv, test_size=0.1)  # train, valid 나누기
    train_df.reset_index(drop=True, inplace=True)  # Index 리셋
    valid_df.reset_index(drop=True, inplace=True)  # Index 리셋

    # 데이터 로더
    lit_loaders = LitDataLoader(train_df=train_df,
                                valid_df=valid_df,
                                tokenizer=tokenizer,
                                translator=False)

    train_loader = lit_loaders.train_dataloader(num_workers=6)  # 학습 Data Loader
    valid_loader = lit_loaders.val_dataloader(num_workers=6)  # 검증 Data Loader

    # 콜백
    csv_logger = pl_loggers.CSVLogger(save_dir='./logs/', name='')
    ckpt_callback = ModelCheckpoint(dirpath='./ckpt',
                                    filename='./model',
                                    monitor='valid_loss',
                                    save_top_k=1,
                                    save_weights_only=True,
                                    mode='min',
                                    save_last=False,
                                    verbose=True)

    early_stop = EarlyStopping(monitor='valid_loss', verbose=True, patience=10, mode='min')

    # 학습
    trainer = Trainer(max_epochs=100,
                      accelerator='gpu',
                      gpus=1,
                      logger=csv_logger,
                      callbacks=[ckpt_callback, early_stop]
                      )

    trainer.fit(model, train_loader, valid_loader)


if __name__ == '__main__':
    main()