import pandas as pd

from utils.config_reader import cfg_load
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor
from model.code_model import CodeModel
from utils.common.project_paths import GetPaths
from dataload.dataloader import LitDataLoader


def main():
    # configs
    config = cfg_load()

    # sets seeds for numpy, torch and python.random.
    seed_everything(config['train']['seed'], workers=True)

    # 모델 및 토크나이저
    model = CodeModel(**config['train']['model'])  # 모델
    model.apply_ckpt(GetPaths.get_project_root(*config['train']['model_check_point']))
    tokenizer = model.tokenizer  # 토크나이저

    # 데이터 프레임
    train_df = pd.read_csv(GetPaths.get_data_folder(config['train']['data']['name1'])) # train file load
    valid_df = pd.read_csv(GetPaths.get_data_folder(config['train']['data']['name2'])) # validation file load

    # 데이터 로더
    lit_loaders = LitDataLoader(train_df=train_df,
                                valid_df=valid_df,
                                tokenizer=tokenizer)

    train_loader = lit_loaders.train_dataloader(**config['train']['train_dataloader'])  # 학습 Data Loader
    valid_loader = lit_loaders.val_dataloader(**config['train']['valid_dataloader'])  # 검증 Data Loader

    # 콜백
    wandb_logger = WandbLogger(project='Dacon_NLP_Code', log_model='all')
    lr_monitor = LearningRateMonitor(logging_interval='step')
    ckpt_callback = ModelCheckpoint(**config['train']['model_checkpoint'])

    early_stop = EarlyStopping(monitor='valid_avg_acc', verbose=True, patience=10, mode='max')

    # 학습
    trainer = Trainer(max_epochs=config['train']['trainer']['max_epochs'],
                      accelerator=config['train']['trainer']['accelerator'],
                      gpus=config['train']['trainer']['gpus'],
                      logger=wandb_logger,
                      callbacks=[lr_monitor, ckpt_callback, early_stop],
                      precision=config['train']['trainer']['precision'],
                      deterministic=config['train']['trainer']['deterministic'],
                      val_check_interval=config['train']['trainer']['val_check_interval'],
                      num_sanity_val_steps=config['train']['trainer']['num_sanity_val_steps'],
                      )

    wandb_logger.watch(model)  # WandB
    trainer.fit(model, train_loader, valid_loader)


if __name__ == '__main__':
    main()