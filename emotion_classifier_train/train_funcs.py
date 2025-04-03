import torch
import torch.nn as nn
import numpy as np

# 1 эпоха обучения
def train_epoch(model, optimizer, scheduler, loss_func, data_loader, device, x, y, lm_count, loss_mean):
    '''
    model = model.train()
    pred = torch.argmax(model(x).to(device))
    loss = loss_func(pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
    lm_count += 1
    loss_mean = 1 / lm_count * loss.item() + (1 - 1 / lm_count) * loss_mean

    return loss_mean
    '''

    model = model.train()

    losses = []
    correct_predictions = 0

    for d in data_loader:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        targets = d["targets"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        _, preds = torch.max(outputs, dim=1)
        loss = loss_func(outputs, targets)

        correct_predictions += torch.sum(preds == targets)
        losses.append(loss.item())

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return correct_predictions.double() / n_examples, np.mean(losses)

# Валидация модели
def valid_model(model, dataloader, loss_func, device):
    model = model.eval()
    losses = []
    correct_predictions = 0

    with torch.no_grad():
        for x, y in dataloader:

            pred = model(x).to(device)
            pred_item = torch.argmax(pred).item()
            label = torch.argmax(y).item()
            loss = loss_func(pred, y)

            correct_predictions += torch.sum(pred_item == label)
            losses.append(loss.item())

    val_acc = correct_predictions.double() / len(dataloader)
    val_loss = np.mean(losses)

    return val_acc, val_loss