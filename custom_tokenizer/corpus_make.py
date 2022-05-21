from utils.common.project_paths import GetPaths
import pandas as pd
from tqdm import tqdm


class CorpusMake:
    def __init__(self, df_path):
        super().__init__()
        self.df_path = df_path
        self.df = self.load_csv()

    def load_csv(self):
        df = pd.read_csv(self.df_path, delimiter='\t')
        return df

    def write(self):
        with open(GetPaths.get_data_folder('corpus_kor.txt'), 'w', encoding='utf-8') as f:
            for i in tqdm(range(len(self.df)), desc='kor'):
                f.write(self.df.loc[i, '원문'] + '\n')

        with open(GetPaths.get_data_folder('corpus_eng.txt'), 'w', encoding='utf-8') as f:
            for i in tqdm(range(len(self.df)), desc='eng'):
                f.write(self.df.loc[i, '번역문'] + '\n')


def main():
    df_path = GetPaths.get_data_folder('train.tsv')
    maker = CorpusMake(df_path)
    maker.write()


if __name__ == '__main__':
    main()