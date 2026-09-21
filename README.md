# Robust Russian Paraphrase Identification with Hard-Negative Augmentation

Binary Russian paraphrase identification using `ai-forever/ruBert-base` and model-derived hard-negative augmentation.

## Approach

1. Train a RuBERT baseline on ParaPhraser + RuPAWS.
2. Find false-positive non-paraphrases on the validation set.
3. Select examples with Jaccard lexical overlap >= 0.5.
4. Add 121 hard negatives to the training set.
5. Train the same RuBERT model on the augmented data.
6. Evaluate on untouched ParaPhraser and RuPAWS test sets.

## Results

| Model | ParaPhraser F1 | RuPAWS F1 |
|---|---:|---:|
| RuBERT baseline | 0.8626 | 0.6877 |
| **RuBERT + hard negatives** | **0.8760** | **0.7191** |

The hard-negative augmentation improves F1 by **1.34 pp** on ParaPhraser and **3.14 pp** on RuPAWS.

## Dataset

- **ParaPhraser** — Russian paraphrase corpus.
- **RuPAWS** — adversarial Russian paraphrase dataset.

After preprocessing, the combined training data contains **11,772** sentence pairs.

## Repository Structure

```text
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   └── experiments.ipynb
├── src/
├── results/
├── report/
├── requirements.txt
└── README.md