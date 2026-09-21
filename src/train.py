import torch
from tqdm.auto import tqdm
from sklearn.metrics import f1_score


def train_epoch(model, loader, optimizer, device):
    model.train()

    total_loss = 0
    predictions = []
    labels = []

    for batch in tqdm(loader, desc="Training"):
        batch = {
            key: value.to(device)
            for key, value in batch.items()
        }

        optimizer.zero_grad()

        outputs = model(**batch)
        loss = outputs.loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        preds = outputs.logits.argmax(dim=1)

        predictions.extend(preds.detach().cpu().numpy())
        labels.extend(batch["labels"].detach().cpu().numpy())

    f1 = f1_score(labels, predictions)

    return total_loss / len(loader), f1


def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    epochs=3
):
    best_f1 = 0.0
    best_state = None

    for epoch in range(epochs):
        train_loss, train_f1 = train_epoch(
            model,
            train_loader,
            optimizer,
            device
        )

        val_f1 = evaluate_f1(
            model,
            val_loader,
            device
        )

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Loss: {train_loss:.4f} | "
            f"Train F1: {train_f1:.4f} | "
            f"Val F1: {val_f1:.4f}"
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_f1


def evaluate_f1(model, loader, device):
    model.eval()

    predictions = []
    labels = []

    with torch.no_grad():
        for batch in loader:
            batch = {
                key: value.to(device)
                for key, value in batch.items()
            }

            outputs = model(**batch)

            preds = outputs.logits.argmax(dim=1)

            predictions.extend(preds.cpu().numpy())
            labels.extend(batch["labels"].cpu().numpy())

    return f1_score(labels, predictions)