import numpy as np
import pandas as pd
from tqdm import tqdm
from utils.common.project_paths import GetPaths
from transformers import AutoTokenizer
from rank_bm25 import BM25Okapi
from itertools import combinations
import os


class CodePreProcess:
    def __init__(self, train_csv=None, test_csv=None):
        super().__init__()
        self.train_csv = train_csv
        self.test_csv = test_csv

    @staticmethod
    def code_preprocess(lines):
        '''
        간단한 전처리 함수
        주석 -> 삭제
        '    '-> tab 변환
        다중 개행 -> 한 번으로 변환
        '''
        preprocess_lines = []
        for line in lines:
            if line.lstrip().startswith('#'):
                continue
            line = line.rstrip()
            if '#' in line:
                line = line[:line.index('#')]
            line = line.replace('\n', '')
            line = line.replace('    ', '\t')
            if line == '':
                continue
            preprocess_lines.append(line)
        preprocessed_script = '\n'.join(preprocess_lines)
        return preprocessed_script

    def py_preprocess(self, py_file):
        '''
        간단한 전처리 함수
        주석 -> 삭제
        '    '-> tab 변환
        다중 개행 -> 한 번으로 변환
        '''
        with open(py_file, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            preprocessed_script = self.code_preprocess(lines)
        return preprocessed_script

    def train_csv_preprocess(self, save_path=None):
        if not self.train_csv:
            raise Exception('train_csv 경로 입력 요망')
        raw_df = pd.read_csv(self.train_csv)
        code1 = []
        code2 = []
        similar = []

        for row_num in tqdm(range(len(raw_df))):
            row = raw_df.iloc[row_num]
            code1_ = row['code1']
            code2_ = row['code2']
            similar_ = row['similar']
            code1_lines = code1_.split('\n')
            code2_lines = code2_.split('\n')
            code1_processed = self.code_preprocess(code1_lines)
            code2_processed = self.code_preprocess(code2_lines)
            code1.append(code1_processed)
            code2.append(code2_processed)
            similar.append(similar_)
        df = pd.DataFrame(data={'code1': code1, 'code2': code2, 'similar': similar})
        if save_path:
            df.to_csv(save_path, encoding='utf-8', index=False)

    def test_csv_preprocess(self, save_path=None):
        if not self.test_csv:
            raise Exception('train_csv 경로 입력 요망')
        raw_df = pd.read_csv(self.test_csv)
        pair_id = []
        code1 = []
        code2 = []

        for row_num in tqdm(range(len(raw_df))):
            row = raw_df.iloc[row_num]
            pair_id_ = row['pair_id']
            code1_ = row['code1']
            code2_ = row['code2']
            code1_lines = code1_.split('\n')
            code2_lines = code2_.split('\n')
            code1_processed = self.code_preprocess(code1_lines)
            code2_processed = self.code_preprocess(code2_lines)
            code1.append(code1_processed)
            code2.append(code2_processed)
            pair_id.append(pair_id_)
        df = pd.DataFrame(data={'pair_id': pair_id, 'code1': code1, 'code2': code2})
        if save_path:
            df.to_csv(save_path, encoding='utf-8', index=False)

    def make_codes(self):
        preprocessed_scripts = []
        problem_nums = []

        code_folder = GetPaths.get_data_folder('code')
        problem_folders = os.listdir(code_folder)

        for problem_folder in tqdm(problem_folders):
            scripts = os.listdir(os.path.join(code_folder, problem_folder))
            problem_num = scripts[0].split('_')[0]  # 문제 번호
            for script in scripts:
                script_file = os.path.join(code_folder, problem_folder, script)
                preprocessed_script = self.py_preprocess(script_file)

                preprocessed_scripts.append(preprocessed_script)
            problem_nums.extend([problem_num] * len(scripts))
        df = pd.DataFrame(data={'code': preprocessed_scripts, 'problem_num': problem_nums})

        tokenizer = AutoTokenizer.from_pretrained("microsoft/graphcodebert-base")
        df['tokens'] = df['code'].apply(tokenizer.tokenize)
        df['len'] = df['tokens'].apply(len)
        ndf = df[df['len'] <= 512].reset_index(drop=True)  # 길이 512가 넘는 Input 삭제

        ndf.to_csv(GetPaths.get_data_folder('codes.tsv'), sep='\t', index=False)

    def make_bm25_scores(self):
        tsv = GetPaths.get_data_folder('codes.tsv')
        df = pd.read_csv(tsv, delimiter='\t')
        tokenizer = AutoTokenizer.from_pretrained('microsoft/graphcodebert-base')

        codes = df['code'].to_list()
        problems = df['problem_num'].to_list()

        data_map = {}

        for idx, [code, problem] in enumerate(zip(codes, problems)):
            problem_num = int(problem[-3:])
            tokenized_code = tokenizer.tokenize(code)
            data_map[idx] = {'code': code,
                             'tokenized_code': tokenized_code,
                             'problem_num': problem_num,
                             'positive': [],
                             'negative': []
                             }

        tokenized_corpus = [tokenizer.tokenize(code) for code in codes]
        bm25 = BM25Okapi(tokenized_corpus)

        for idx, V in tqdm(data_map.items()):
            bm25_score = bm25.get_scores(V['tokenized_code'])
            problem = V['problem_num']
            ranks = bm25_score.argsort()[::-1]  # 내림차순

            for rank in ranks:
                if data_map[rank]['problem_num'] == problem:  # positive
                    V['positive'].append(rank)
                else:
                    V['negative'].append(rank)
            V['negative'] = V['negative'][:150]
            if idx % 100 == 99:
                tmp = pd.DataFrame(data_map).T
                tmp = tmp.loc[:, ['positive', 'negative']]
                tmp.to_csv('./data_map.csv')

        tmp = pd.DataFrame(data_map).T
        tmp = tmp.loc[:, ['positive', 'negative']]
        tmp.to_csv('./data_map.csv')

    def noname_(self):
        code_csv = pd.read_csv(GetPaths.get_data_folder('codes.tsv'), delimiter='\t')
        bm_scores = pd.read_csv(GetPaths.get_data_folder('data_map.csv'))

        pairs = {}
        for i in range(len(bm_scores)):
            pairs[i] = {}
            pairs[i]['positive'] = []
            pairs[i]['negative'] = []

        for idx in range(len(bm_scores)):
            row = bm_scores.iloc[idx]
            positives = row['positive']
            negatives = row['negative']

            # string 형식으로 저장된 리스트 변환
            positives = positives.replace('[', '').replace(']', '')
            positives = positives.split(',')
            positives = [int(i) for i in positives]

            negatives = negatives.replace('[', '').replace(']', '')
            negatives = negatives.split(',')
            negatives = [int(i) for i in negatives]

            if idx in positives:
                positives.pop(positives.index(idx))  # 자기 자신 제거

            for positive in positives:
                if idx not in pairs[positive]['positive']:
                    pairs[idx]['positive'].append(positive)
                if len(pairs[idx]['positive']) >= 5:
                    break

            for positive in positives[::-1]:
                if idx not in pairs[positive]['positive']:
                    pairs[idx]['positive'].append(positive)
                if len(pairs[idx]['positive']) >= 10:
                    break

            for negative in negatives:
                if idx not in pairs[negative]['negative']:
                    pairs[idx]['negative'].append(negative)
                if len(pairs[idx]['negative']) >= 10:
                    break

        tmp = []
        for code1, V in tqdm(pairs.items()):
            for pos_code2 in V['positive']:
                pos = pd.DataFrame(data={'code1': code_csv.loc[code1, 'code'], 'code2': code_csv.loc[pos_code2, 'code'], 'similar': [1]})
                tmp.append(pos)
            for neg_code2 in V['negative']:
                neg = pd.DataFrame(data={'code1': code_csv.loc[code1, 'code'], 'code2': code_csv.loc[neg_code2, 'code'], 'similar': [0]})
                tmp.append(neg)

        df = pd.concat(tmp, ignore_index=True)
        df.to_csv(GetPaths.get_data_folder('train_data_.csv'), index=False)


if __name__ == '__main__':
    preprocess = CodePreProcess()
    preprocess.noname_()