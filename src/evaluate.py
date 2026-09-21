import torch

from tqdm.auto import tqdm
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)

from .dataset import ParaphraseDataset
from torch.utils.data import DataLoader


def evaluate_model(
    model,
    df,
    tokenizer,
    device,
    batch_size=16,
    max_length=128
):
    dataset = ParaphraseDataset(
        df,
        tokenizer,
        max_length=max_length
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False
    )

    model.eval()

    predictions = []
    labels = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluation"):
            batch = {
                key: value.to(device)
                for key, value in batch.items()
            }

            outputs = model(**batch)
            preds = outputs.logits.argmax(dim=1)

            predictions.extend(
                preds.cpu().numpy()
            )
            labels.extend(
                batch["labels"].cpu().numpy()
            )

    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1:       {f1:.4f}")

    print(
        classification_report(
            labels,
            predictions,
            digits=4
        )
    )

    print("Confusion matrix:")
    print(confusion_matrix(labels, predictions))

    return {
        "accuracy": accuracy,
        "f1": f1,
        "predictions": predictions,
        "labels": labels
    }