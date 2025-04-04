import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

# 1 эпоха обучения
def train_epoch(model, optimizer, scheduler, loss_func, data_loader, device):
    model = model.train()

    losses = []
    correct_predictions = 0

    train_tqdm = tqdm(data_loader, leave=True)

    for d in train_tqdm:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        targets = d["targets"].to(device)


        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        _, preds = torch.max(outputs, dim=1)
        loss = loss_func(outputs, targets)


        correct_predictions += torch.sum(preds == torch.argmax(targets, dim=1))
        losses.append(loss.item())

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()


    return correct_predictions.double() / len(data_loader), np.mean(losses)

# Валидация модели
def valid_model(model, dataloader, loss_func, device):
    model = model.eval()
    losses = []
    correct_predictions = 0

    with torch.no_grad():
        for d in dataloader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            _, preds = torch.max(outputs, dim=1)

            loss = loss_func(outputs, targets)

            correct_predictions += torch.sum(preds == torch.argmax(targets, dim=1))
            losses.append(loss.item())

    val_acc = correct_predictions.double() / len(dataloader)
    val_loss = np.mean(losses)

    return val_acc, val_loss