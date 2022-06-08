import pandas as pd
from tqdm import tqdm
from utils.common.project_paths import GetPaths
from transformers import AutoTokenizer
import os


def make_codes():
    preproc_scripts = []
    problem_nums = []

    code_folder = GetPaths.get_data_folder('code')
    problem_folders = os.listdir(code_folder)

    for problem_folder in tqdm(problem_folders):
        scripts = os.listdir(os.path.join(code_folder, problem_folder))
        problem_num = scripts[0].split('_')[0]  # 문제 번호
        for script in scripts:
            script_file = os.path.join(code_folder, problem_folder, script)
            preprocessed_script = preprocess_script(script_file)

            preproc_scripts.append(preprocessed_script)
        problem_nums.extend([problem_num]*len(scripts))
    df = pd.DataFrame(data={'code': preproc_scripts, 'problem_num': problem_nums})

    tokenizer = AutoTokenizer.from_pretrained("microsoft/graphcodebert-base")
    df['tokens'] = df['code'].apply(tokenizer.tokenize)
    df['len'] = df['tokens'].apply(len)
    ndf = df[df['len'] <= 512].reset_index(drop=True)  # 길이 512가 넘는 Input 삭제

    ndf.to_csv(GetPaths.get_data_folder('codes.tsv'), sep='\t', index=False)


if __name__ == '__main__':
    make_codes()