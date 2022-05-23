from pytorch_lightning import Trainer


def custom_lr_finder(model, train_loader, num_training=10000):
    trainer = Trainer(accelerator='gpu',
                      gpus=1,
                      precision=16,
                      num_sanity_val_steps=0)

    lr_finder = trainer.tuner.lr_find(model, train_dataloaders=train_loader, num_training=num_training)
    new_lr = lr_finder.suggestion()
    return new_lr
