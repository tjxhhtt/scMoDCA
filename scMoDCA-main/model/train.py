import math

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from model.util import scMoDCA_CMDL_LossManager, save_checkpoint

def adjust_lr_delayed_cosine(optimizer, epoch, max_epochs, initial_lr, delay_epochs=20):
    if epoch <= delay_epochs:
        lr = initial_lr
    else:
        progress = (epoch - delay_epochs) / (max_epochs - delay_epochs)
        lr = 0.5 * initial_lr * (1 + math.cos(math.pi * progress))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def adjust_critic_lr(optimizer, epoch, initial_lr, drop_epoch=20):
    lr = initial_lr if epoch <= drop_epoch else initial_lr * 0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def train_model(args, model, train_dl, test_dl, lr, epochs, dataset, classify_dim=1, save_path=""):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # loss_config = {
    #     'num_classes': classify_dim,
    #     'mi_w': args.get('mi_loss_weight', 0.01),
    #     'train_loss_weight': 0.1
    # }
    loss_manager = scMoDCA_CMDL_LossManager(args)

    optimizer_main = optim.Adam(
        list(model.encoder.parameters()) +
        list(model.decoder.parameters()) +
        list(model.infonce_mod1_shared.parameters()) +
        list(model.infonce_mod2_shared.parameters()) +
        list(model.classify.parameters()),
        lr=lr,
    )

    optimizer_mi_estimator = optim.Adam(
        list(model.club1.parameters()) +
        list(model.club2.parameters()) +
        list(model.club_cond_1.parameters()) +
        list(model.club_cond_2.parameters()),
        lr=lr
    )

    for epoch in range(1, epochs + 1):
        model.train()
        current_lr = adjust_lr_delayed_cosine(optimizer_main, epoch, epochs, lr, delay_epochs=5)
        _ = adjust_critic_lr(optimizer_mi_estimator, epoch, lr, drop_epoch=15)

        for i, batch_sample in enumerate(train_dl):
            x = batch_sample['data'].to(device).float().view(batch_sample['data'].size(0), -1)
            y_true = batch_sample['label'].to(device).long()

            # STEP 1: 更新 MI 估计器
            results = model(x)
            l_mi_critic, mi_val = loss_manager.learn_losses(model, results)
            optimizer_mi_estimator.zero_grad()
            l_mi_critic.backward()
            optimizer_mi_estimator.step()

            # STEP 2: 更新主模型
            optimizer_main.zero_grad()
            l_main, (l_recon, l_mi, l_cls) = loss_manager.main_loss(
                model, x, results, y_true, epoch, dataset
            )

            total_loss = l_main
            total_loss.backward()

            main_params = (list(model.encoder.parameters()) +
                           list(model.decoder.parameters()) +
                           list(model.infonce_mod1_shared.parameters()) +
                           list(model.infonce_mod2_shared.parameters()) +
                           list(model.classify.parameters()))
            torch.nn.utils.clip_grad_norm_(main_params, max_norm=1.0)
            optimizer_main.step()

            if i % 10 == 0:
                print(f"Epoch [{epoch}/{epochs}] Batch [{i}] "
                      f"Loss: {total_loss.item():.3f} | Recon: {l_recon.item():.3f} | MI: {l_mi.item():.3f}  | CLS:{l_cls.item():.3f}")

        if epoch % 10 == 0 or epoch == epochs:
            save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
            }, save_path)

    return model
