from model.bart_model import Bart


def main():
    inf = Bart(mode='inference')  # 모델 불러오기
    inf.apply_ckpt('./ckpt/model-v1.ckpt')  # CKPT 적용
    inf.model.save_pretrained('./binary/kang_bart2/')  # Binary 추출


if __name__ == '__main__':
    main()
