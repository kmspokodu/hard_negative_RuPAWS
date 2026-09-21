import re
import pandas as pd


def tokenize_words(text):
    return set(
        re.findall(
            r"[а-яёa-z0-9]+",
            str(text).lower()
        )
    )


def lexical_overlap(sentence1, sentence2):
    tokens1 = tokenize_words(sentence1)
    tokens2 = tokenize_words(sentence2)

    if not tokens1 and not tokens2:
        return 1.0

    if not tokens1 or not tokens2:
        return 0.0

    return len(tokens1 & tokens2) / len(tokens1 | tokens2)


def find_hard_negatives(
    df,
    predictions,
    overlap_threshold=0.5
):
    result = df.copy()

    result["prediction"] = predictions

    hard_negatives = result[
        (result["label"] == 0) &
        (result["prediction"] == 1)
    ].copy()

    hard_negatives["overlap"] = hard_negatives.apply(
        lambda row: lexical_overlap(
            row["sentence1"],
            row["sentence2"]
        ),
        axis=1
    )

    hard_negatives = hard_negatives[
        hard_negatives["overlap"] >= overlap_threshold
    ].copy()

    return hard_negatives


def prepare_hard_training_data(
    train_df,
    hard_negatives
):
    hard_train = hard_negatives[
        ["sentence1", "sentence2", "label"]
    ].copy()

    return pd.concat(
        [train_df, hard_train],
        ignore_index=True
    )