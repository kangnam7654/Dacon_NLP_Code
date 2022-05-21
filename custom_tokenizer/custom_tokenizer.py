from transformers import PreTrainedTokenizerFast
from utils.common.project_paths import GetPaths

def get_tokenizer():
    path = GetPaths.get_project_root('custom_tokenizer', 'vocab', 'vocab.json')
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=path,
                                        bos_token="<s>",
                                        eos_token="</s>",
                                        unk_token="<unk>",
                                        pad_token="<pad>",
                                        mask_token="<mask>",
                                        do_basic_tokenize=False
                                        )

    return tokenizer


if __name__ == '__main__':
    tokenizer = get_tokenizer()
