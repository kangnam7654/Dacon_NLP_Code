import torch
from utils.config_reader import cfg_load
from utils.common.project_paths import GetPaths
from model.code_model import CodeModel
from dataload.dataloader import LitDataLoader
from pytorch_lightning import Trainer
from tqdm import tqdm
import pandas as pd


def main():
    # configs
    config = cfg_load()

    # 테스트 csv
    test_csv = pd.read_csv(GetPaths.get_data_folder('pre_processed_test.csv'))  # csv 읽어오기

    # 제출 df
    submit = pd.DataFrame(columns=['pair_id', 'similar'])

    # 모델 및 토크나이저
    model = CodeModel(**config['inference']['model'])  # 모델
    model.apply_ckpt(GetPaths.get_project_root('ckpt', config['inference']['ckpt']['ckpt_name']))
    tokenizer = model.tokenizer  # 토크나이저

    # 데이터 로더
    lit_loaders = LitDataLoader(test_df=test_csv,
                                tokenizer=tokenizer)
    test_loader = lit_loaders.test_dataloader(batch_size=128)

    # 학습
    trainer = Trainer(accelerator='gpu',
                      gpus=1,
                      precision=16,
                      num_sanity_val_steps=0,
                      max_epochs=1
                      )

    all_predicts = trainer.predict(model=model, dataloaders=test_loader)
    all_predicts_ = torch.cat(all_predicts, dim=0)
    for p_id, sim in tqdm(enumerate(all_predicts_)):
        insert = [p_id+1, sim.item()]
        submit.loc[p_id] = insert
    submit.to_csv(GetPaths.get_data_folder('submit.csv'), encoding='utf-8', index=False)


def __test():
    csv = pd.read_csv(GetPaths.get_data_folder('submit.csv'), encoding='utf-8')
    csv_ = csv.drop(columns=[csv.columns[0]])
    csv_.to_csv(GetPaths.get_data_folder('submit_.csv'), encoding='utf-8', index=False)


if __name__ == '__main__':
    main()

