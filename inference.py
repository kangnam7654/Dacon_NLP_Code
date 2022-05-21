from transformers import BartForConditionalGeneration
from custom_tokenizer.custom_tokenizer import get_tokenizer
import torch


def main():
    model = BartForConditionalGeneration.from_pretrained('./binary/kang_bart2')
    tokenizer = get_tokenizer()

    # kor = input('원문 입력: ')
    kor = '<s><mask>가 좋은 하루입니다.</s>'

    input_ids = tokenizer(kor, return_tensors='pt')['input_ids']

    # out = list(torch.argmax(model(input_ids).logits, dim=2))
    out = model.generate(input_ids, bos_token_id=1, eos_token_id=2, max_length=256)
    translated = tokenizer.batch_decode(out, skip_special_tokens=True)
    # translated = tokenizer.decode(out[0])

    print(translated)

if __name__ == '__main__':
    main()
