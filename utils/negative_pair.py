from rank_bm25 import BM25Okapi
from itertools import combinations
from tqdm import tqdm
from transformers import AutoTokenizer
import pandas as pd


df = pd.read_csv('../data/codes.tsv', delimiter='\t')
tokenizer = AutoTokenizer.from_pretrained('microsoft/graphcodebert-base')

codes = df['code'].to_list()
problems = df['problem_num'].unique().tolist()
problems.sort()

tokenized_corpus = [tokenizer.tokenize(code) for code in codes]
bm25 = BM25Okapi(tokenized_corpus)

total_positive_pairs = []
total_negative_pairs = []

for problem in tqdm(problems):
    solution_codes = df[df['problem_num'] == problem]['code']
    positive_pairs = list(combinations(solution_codes.to_list(), 2))

    solution_codes_indices = solution_codes.index.to_list()
    negative_pairs = []

    first_tokenized_code = tokenizer.tokenize(positive_pairs[0][0])
    negative_code_scores = bm25.get_scores(first_tokenized_code)
    negative_code_ranking = negative_code_scores.argsort()[::-1]  # 내림차순
    ranking_idx = 0

    for solution_code in solution_codes:
        negative_solutions = []
        while len(negative_solutions) < len(positive_pairs) // len(solution_codes):
            high_score_idx = negative_code_ranking[ranking_idx]

            if high_score_idx not in solution_codes_indices:
                negative_solutions.append(df['code'].iloc[high_score_idx])
            ranking_idx += 1

        for negative_solution in negative_solutions:
            negative_pairs.append((solution_code, negative_solution))

    total_positive_pairs.extend(positive_pairs)
    total_negative_pairs.extend(negative_pairs)

pos_code1 = list(map(lambda x: x[0], total_positive_pairs))
pos_code2 = list(map(lambda x: x[1], total_positive_pairs))

neg_code1 = list(map(lambda x: x[0], total_negative_pairs))
neg_code2 = list(map(lambda x: x[1], total_negative_pairs))

pos_label = [1] * len(pos_code1)
neg_label = [0] * len(neg_code1)

pos_code1.extend(neg_code1)
total_code1 = pos_code1
pos_code2.extend(neg_code2)
total_code2 = pos_code2
pos_label.extend(neg_label)
total_label = pos_label
pair_data = pd.DataFrame(data={
    'code1': total_code1,
    'code2': total_code2,
    'similar': total_label
})
pair_data = pair_data.sample(frac=1).reset_index(drop=True)

pair_data.to_csv('data/train_data.csv', index=False)

'''
valid_df에도 동일하게...
'''