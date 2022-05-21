import numpy as np
import pandas as pd
import torch
from torch.cuda import amp
import tqdm


def print_result(result):
    """
    결과를 print하는 함수 입니다.
    :param result: list를 input으로 받아 print합니다.
    :return:
    """
    epoch, train_loss, valid_loss, train_acc, valid_acc = result
    print(
        f"[epoch{epoch}] train_loss: {round(train_loss, 3)}, valid_loss: {round(valid_loss, 3)}, train_acc: {train_acc}%, valid_acc: {valid_acc}%"
    )


def split_dataset(data_path, target_column=None, valid_size=0.2, seed=None):
    from sklearn.model_selection import train_test_split
    """
    학습 데이터셋과 검증 데이터셋으로 나누는 함수입니다.
    :param data_path:
    :param test_size:
    :param seed:
    :param target_column:
    :return:
    """
    df = pd.read_csv(data_path)
    train_df, valid_df = train_test_split(
        df,
        test_size=valid_size,
        random_state=seed,
        stratify=df[target_column]
    )
    train_df.reset_index(drop=True, inplace=True)
    valid_df.reset_index(drop=True, inplace=True)
    return train_df, valid_df


def share_loop(epoch=10,
               model=None,
               data_loader=None,
               criterion=None,
               optimizer=None,
               mode="train",
               fp16=True):
    """
    학습과 검증에서 사용하는 loop 입니다. mode를 이용하여 조정합니다.
    :param epoch:
    :param model:
    :param data_loader:
    :param criterion:
    :param optimizer:
    :param mode: 'train', 'valid' 중 하나의 값을 받아 loop를 진행합니다.
    :return: average_loss(float64), total_losses(list), accuracy(float)
    """
    count = 0
    correct = 0
    total_losses = []
    scaler = amp.GradScaler()

    mode = mode.lower()
    if mode == "train":
        model.train()
        for batch in tqdm.tqdm(data_loader, desc=f"{mode} {epoch}"):
            data, label = batch
            # gradient 초기화
            optimizer.zero_grad()

            if fp16:
                with amp.autocast():
                    out = torch.softmax(model(input_ids=data).logits, dim=1)
                    loss = criterion(out, label)
                    assert out.dtype is torch.float16
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                out = torch.softmax(model(input_ids=data).logits)
                loss = criterion(out, label)
                loss.backward()
                model.optimizer.step()



            # accuracy 계산
            predicted = torch.argmax(out, dim=1)
            label_ = torch.argmax(label, dim=1)
            count += label.size(0)
            correct += (predicted == label_).sum().item()
            total_losses.append(loss.item())

    elif mode == 'valid':
        model.eval()
        with torch.no_grad():
            for batch in tqdm.tqdm(data_loader, desc=f"{mode} {epoch}"):
                data, label = batch

                if fp16:
                    with amp.autocast():
                        out = torch.softmax(model(input_ids=data).logits, dim=1)
                        loss = criterion(out, label)
                        assert out.dtype is torch.float16
                else:
                    out = torch.softmax(model(input_ids=data).logits, dim=1)
                    loss = criterion(out, label)


                # accuracy 계산
                predicted = torch.argmax(out, dim=1)
                label_ = torch.argmax(label, dim=1)
                count += label.size(0)
                correct += (predicted == label_).sum().item()
                total_losses.append(loss.item())

    elif mode == 'test':
        model.eval()
        with torch.no_grad():
            all_preds = []
            for batch in tqdm.tqdm(data_loader, desc=f"{mode}"):
                data = batch
                if fp16:
                    with amp.autocast():
                        out = torch.softmax(model(input_ids=data).logits, dim=1)
                else:
                    out = torch.softmax(model(input_ids=data).logits, dim=1)
                predicted = torch.argmax(out, dim=1)
                all_preds.append(predicted.detach())
        return all_preds
    else:
        raise Exception(f'mode는 train, valid 중 하나여야 합니다. 현재 mode값 -> {mode}')

    avg_loss = np.average(total_losses)
    accuracy = 100 * correct / count  # Accuracy 계산
    return avg_loss, total_losses, accuracy


if __name__ == '__main__':
    pass
