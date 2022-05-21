from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
from transformers import PreTrainedTokenizerFast
from utils.common.project_paths import GetPaths
from glob import glob


def main():
    """
    bpe 로 Tokenizer를 만드는 함수입니다.
    :return:
    """

    # Initialize a tokenizer
    tokenizer = Tokenizer(models.BPE())

    # Customize pre-tokenization and decoding
    tokenizer.pre_tokenizer = pre_tokenizers.Metaspace()
    tokenizer.decoder = decoders.Metaspace()

    # Special Tokens
    special_token = ['<pad>', '<s>', '</s>', '<unk>', '<mask>']

    # unused token add
    for i in range(100):
        unused = f'<unused{i}>'
        special_token.append(unused)

    # And then train
    trainer = trainers.BpeTrainer(
        vocab_size=32000,
        min_frequency=5,
        limit_alphabet=6000,
        special_tokens=special_token
    )
    corp_path = glob(GetPaths.get_data_folder('corpus_kor.txt'), recursive=True)
    tokenizer.train(corp_path, trainer=trainer)

    # And Save it
    tokenizer.save("./vocab/vocab.json", pretty=True)


def test():
    tokenizer = PreTrainedTokenizerFast(tokenizer_file='./vocab/vocab.json',
                                        bos_token="<s>",
                                        eos_token="</s>",
                                        unk_token="<unk>",
                                        pad_token="<pad>",
                                        mask_token="<mask>",
                                        )

    sentence = '안녕하세요, 저는 김강남 입니다.'
    encode = tokenizer.encode(sentence)
    decode = tokenizer.decode(encode)
    tokenized = tokenizer(sentence)
    print(f'\nencode: {encode} \ndecode: {decode} \ntokenized: {tokenized}')


if __name__ == '__main__':
    main()
    # test()