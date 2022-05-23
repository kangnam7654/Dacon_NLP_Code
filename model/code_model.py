import numpy as np
import torch
from pytorch_lightning import LightningModule
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class CodeModel(LightningModule):
    def __init__(self, model_name=None, tokenizer_name=None):
        super().__init__()
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name
        self.tokenizer = self.build_tokenizer()
        self.model = self.build_model()
        self.lr = 0.01
        self.save_hyperparameters()

    def forward(self, inputs):
        out = self.model(input_ids=inputs['input_ids'].squeeze(1),
                         attention_mask=inputs['attention_mask'].squeeze(1),
                         token_type_ids=inputs['token_type_ids'].squeeze(1),
                         labels=inputs['labels'],
                         )
        return out

    def training_step(self, batch, batch_idx):
        train_loss, train_acc = self.__share_step(batch)
        results = {'loss': train_loss, 'acc': train_acc}
        self.log_dict({'train_loss': train_loss, 'train_acc': train_acc})
        return results

    def validation_step(self, batch, batch_idx):
        valid_loss, valid_acc = self.__share_step(batch)
        results = {'loss': valid_loss, 'acc': valid_acc}
        self.log_dict({'valid_loss': valid_loss, 'valid_acc': valid_acc})
        return results

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        prediction = torch.argmax(self(batch), dim=1)
        return prediction

    def training_epoch_end(self, outputs):
        self.__share_epoch_end(outputs, 'train')

    def validation_epoch_end(self, outputs):
        self.__share_epoch_end(outputs, 'valid')

    def configure_optimizers(self):
        opt = torch.optim.SGD(params=self.parameters(),
                              lr=self.lr,
                              momentum=0.9,
                              nesterov=True)
        return [opt]
        # sch = {'scheduler': torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=opt, T_0=50, T_mult=2, eta_min=1e-5),
        #        'interval': 'step',
        #        'frequency': 1}
        # returns = {'optimizer': opt, 'lr_scheduler': sch}
        # return returns

    def __share_step(self, batch):
        out = self(batch)
        loss = out.loss
        acc = self.compute_accuracy(out, batch['labels'])
        return loss, acc

    def __share_epoch_end(self, outputs, mode):
        all_loss = []
        all_acc = []
        for out in outputs:
            loss, acc = out['loss'], out['acc']
            all_loss.append(loss)
            all_acc.append(acc)
        avg_loss = torch.mean(torch.stack(all_loss))
        avg_acc = np.mean(all_acc)
        self.log_dict({f'{mode}_avg_loss': avg_loss, f'{mode}_avg_acc': avg_acc})

    def build_tokenizer(self):
        if self.tokenizer_name:
            tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        else:
            print('모델 입력이 없어 "graphcodebert-base"토크나이저를 불러옵니다.')
            tokenizer = AutoTokenizer.from_pretrained("microsoft/graphcodebert-base")
        return tokenizer

    def build_model(self):
        if self.model_name:
            graph_code_bert = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        else:
            print('모델 입력이 없어 "roberta-base"모델을 불러옵니다.')
            graph_code_bert = AutoModelForSequenceClassification.from_pretrained('roberta-base')
        return graph_code_bert

    def apply_ckpt(self, checkpoint_path):
        ckpt = torch.load(checkpoint_path)
        if 'state_dict' in ckpt.keys():
            state_dict = {}
            for k, v in ckpt['state_dict'].items():
                k = k[6:]
                state_dict[k] = v
            self.model.load_state_dict(state_dict)
        else:
            self.model.load_state_dict(ckpt)
        print(f'모델을 성공적으로 불러왔습니다.')

    @staticmethod
    def compute_accuracy(out, labels):  # for classification
        max_indices = torch.argmax(out.logits, dim=-1)
        acc = (max_indices == labels).to(torch.float).mean().item()
        return acc


if __name__ == '__main__':
    model = CodeModel()