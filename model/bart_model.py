import torch
from pytorch_lightning import LightningModule
from transformers import AutoModelForSeq2SeqLM, BartForConditionalGeneration
from custom_tokenizer.custom_tokenizer import get_tokenizer


class Bart(LightningModule):
    def __init__(self, mode='train'):
        super().__init__()
        self.tokenizer = self.build_tokenizer()
        self.bart_config = self.bart_configs()
        self.model = self.build_model(self.bart_config)
        self.bos_token = '<s>'
        self.eos_token = '</s>'
        self.pad_token_id = self.tokenizer.pad_token_id
        self.mode = mode

    def forward(self, inputs):
        attention_mask = inputs['input_ids'].ne(self.pad_token_id).float()
        decoder_attention_mask = inputs['decoder_input_ids'].ne(self.pad_token_id).float()

        out = self.model(input_ids=inputs['input_ids'],
                         attention_mask=attention_mask,
                         decoder_input_ids=inputs['decoder_input_ids'],
                         decoder_attention_mask=decoder_attention_mask,
                         labels=inputs['labels'].long(),
                         return_dict=True
                         )
        return out

    def training_step(self, batch, batch_idx):
        train_loss, train_acc = self.__share_step(batch)
        # train_acc.requires_grad = True
        results = {'loss': train_loss, 'acc': train_acc}
        return results

    def validation_step(self, batch, batch_idx):
        valid_loss, valid_acc = self.__share_step(batch)
        results = {'loss': valid_loss, 'acc': valid_acc}
        return results

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        out = torch.softmax(self(batch), dim=1)
        pred = torch.argmax(out, dim=1)
        return pred

    def training_epoch_end(self, outputs):
        self.__share_epoch_end(outputs, 'train')

    def validation_epoch_end(self, outputs):
        self.__share_epoch_end(outputs, 'valid')

    def configure_optimizers(self):
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

        # translator 용
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(
                nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        opt = torch.optim.SGD(
            self.parameters(),
            lr=0.001,
            weight_decay=0.0001,
            momentum=0.9,
            nesterov=True
        )
        # sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=opt,)
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=opt,
                                                         mode='min',
                                                         factor=0.1,
                                                         patience=2,
                                                         min_lr=1e-8,
                                                         verbose=True)
        monitor = 'valid_loss'
        returns = {'optimizer': opt, 'lr_scheduler': sch, 'monitor': monitor}
        return returns

    def __share_step(self, batch):
        out = self(batch)
        loss = out.loss
        acc = self.compute_accuracy(out, batch['labels']).unsqueeze(dim=0)
        return loss, acc

    def __share_epoch_end(self, outputs, mode):
        all_loss = []
        all_acc = []
        for out in outputs:
            loss, acc = out['loss'], out['acc']
            all_loss.append(loss.unsqueeze(0))
            all_acc.append(acc.unsqueeze(0))
        avg_loss = torch.mean(torch.cat(all_loss))
        avg_acc = torch.mean(torch.cat(all_acc))
        self.log_dict({f'{mode}_loss': avg_loss, f'{mode}_acc': avg_acc})

    def bart_configs(self):
        bart_config = AutoModelForSeq2SeqLM.from_pretrained('facebook/bart-base').config
        bart_config.vocab_size = self.tokenizer.vocab_size  # default : 50265
        bart_config.pad_token_id = self.tokenizer.pad_token_id
        bart_config.bos_token_id = self.tokenizer.bos_token_id
        bart_config.forced_bos_id = self.tokenizer.bos_token_id
        bart_config.eos_token_id = self.tokenizer.eos_token_id
        bart_config.forced_eos_id = self.tokenizer.eos_token_id
        return bart_config

    @staticmethod
    def build_tokenizer():
        tokenizer = get_tokenizer()
        return tokenizer

    @staticmethod
    def build_model(bart_config):
        bart_model = BartForConditionalGeneration(bart_config)
        return bart_model

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
        max_indices = torch.argmax(torch.softmax(out.logits, dim=2), dim=2)
        acc = (max_indices == labels).to(torch.float).mean() * 100
        return acc


if __name__ == '__main__':
    model = Bart()