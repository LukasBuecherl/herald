"""
negative_samples.py — Negative peptide sample generation for HERALD.

Provides utilities for generating random non-AMP peptide sequences that are
not present in the curated AMP database. These negative samples can be used
to build simple baseline classification datasets for comparing AMP and
non-AMP-like sequences.

The current implementation generates random amino-acid sequences with lengths
sampled from the observed AMP sequence length range. These sequences are
synthetic negatives and should be treated as a simple baseline rather than
experimentally confirmed non-antimicrobial peptides.
"""

import os
import random

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR_PROCESSED = os.path.join(BASE_DIR, "data", "processed")


def generate_negative_samples(amp_df, n_samples):
    """
    Generate random peptide sequences not present in the AMP database.
    Args:
        amp_df (pandas.DataFrame): AMP database containing a "sequence" column.
        n_samples (int): Number of synthetic negative sequences to generate.
    Returns:
        pandas.DataFrame: DataFrame containing generated negative samples with
        "sequence" and "source" columns.
    """
    AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
    min_len = amp_df["sequence"].str.len().min()
    max_len = amp_df["sequence"].str.len().max()

    existing_sequences = set(amp_df["sequence"])
    records = []
    while len(records) < n_samples:
        n = random.randint(min_len, max_len)
        madeup_sequence = "".join(random.choices(AMINO_ACIDS, k=n))
        if madeup_sequence not in existing_sequences:
            records.append(madeup_sequence)
            existing_sequences.add(madeup_sequence)

    df_neg = pd.DataFrame({"sequence": records, "source": "negative"})
    return df_neg


amp_df = pd.read_csv(os.path.join(DATA_DIR_PROCESSED, "amp_database.csv"))
print(generate_negative_samples(amp_df, 5))


def prepare_dataset(amp_df, n_samples):
    """
    Prepare a dataset by generating negative samples and combining with AMP data.
    Args:
        amp_df (pandas.DataFrame): AMP database containing a "sequence" column.
        n_samples (int): Number of synthetic negative sequences to generate.
    Returns:
        tuple: x_train, x_test, y_train, y_test as pandas Series,
            where x contains sequences and y contains binary labels (1=AMP, 0=negative).
    """
    amp_df = amp_df.copy()
    amp_df["label"] = 1
    df_neg = generate_negative_samples(amp_df, n_samples)
    df_neg["label"] = 0
    df_all = pd.concat([amp_df, df_neg])

    x = df_all["sequence"]
    y = df_all["label"]
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        shuffle=True,
    )

    return x_train, x_test, y_train, y_test


def compute_esm2_embeddings(sequences, model, alphabet, batch_size=32):
    """
    Compute ESM-2 sequence embeddings for a list of peptide sequences.

    Each sequence is tokenized and passed through the ESM-2 transformer model.
    Per-token representations from the final layer are mean-pooled across the
    sequence length to produce a single fixed-size embedding per peptide.

    Args:
        sequences (list[str]): List of amino acid sequences to embed.
        model: Pretrained ESM-2 model loaded via esm.pretrained.
        alphabet: ESM-2 alphabet object used for tokenization.
        batch_size (int): Number of sequences to process per batch.

    Returns:
        numpy.ndarray: Array of shape (n_sequences, 320) containing
        one embedding vector per input sequence.
    """
    # use Apple Silicon GPU if available, otherwise fall back to CPU
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # batch converter tokenizes sequences into numerical tokens for ESM-2
    batch_converter = alphabet.get_batch_converter()

    all_embeddings = []

    # process sequences in batches to avoid running out of memory
    for start in range(0, len(sequences), batch_size):
        batch = sequences[start : start + batch_size]

        # ESM-2 expects (label, sequence) tuples — use index as label
        data = [(str(j), seq) for j, seq in enumerate(batch)]

        # tokenize batch — discard labels and strings, keep tokens only
        _, _, batch_tokens = batch_converter(data)
        batch_tokens = batch_tokens.to(device)

        # run through ESM-2 without computing gradients (inference only)
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[6])

        # extract layer 6 representations — shape: (batch_size, seq_len+2, 320)
        token_embeddings = results["representations"][6]

        for i, (_, seq) in enumerate(data):
            L = len(seq)
            # slice 1:L+1 to skip START and END special tokens
            # mean(0) averages across sequence length → shape (320,)
            embedding = token_embeddings[i, 1 : L + 1].mean(0)
            # move to CPU and convert to numpy for scikit-learn compatibility
            all_embeddings.append(embedding.cpu().numpy())

    # stack list of (320,) arrays into a single (n_sequences, 320) matrix
    return np.array(all_embeddings)


def train_classifier(embeddings, labels):
    """
    Train a logistic regression classifier on ESM-2 peptide embeddings.
    Args:
        embeddings (numpy.ndarray): Array of shape (n_sequences, 320)
            containing ESM-2 embeddings for each peptide.
        labels (numpy.ndarray): Binary labels where 1 = AMP and 0 = negative.
    Returns:
        sklearn.linear_model.LogisticRegression: Fitted classifier saved
        to data/processed/amp_classifier.pkl.
    """
    clf = LogisticRegression(max_iter=1000)
    clf.fit(embeddings, labels)
    joblib.dump(clf, os.path.join(DATA_DIR_PROCESSED, "amp_classifier.pkl"), compress=3)
    return clf


def predict_amp(sequence, model, alphabet, clf):
    """
    Predict whether a peptide sequence is antimicrobial.

    Args:
        sequence (str): Amino acid sequence to evaluate.
        model: Pretrained ESM-2 model.
        alphabet: ESM-2 alphabet object used for tokenization.
        clf: Fitted logistic regression classifier.

    Returns:
        tuple: (prediction, probability) where prediction is 1 (AMP) or 0 (not AMP)
        and probability is the confidence score between 0 and 1.
    """
    embedding = compute_esm2_embeddings([sequence], model, alphabet)
    prediction = clf.predict(embedding)[0]
    probability = float(clf.predict_proba(embedding)[0, 1])
    return prediction, probability


# Test function
if __name__ == "__main__":
    import esm
    from sklearn.metrics import classification_report, roc_auc_score

    # load data
    amp_df = pd.read_csv(os.path.join(DATA_DIR_PROCESSED, "amp_database.csv"))

    # prepare dataset
    X_train, X_test, y_train, y_test = prepare_dataset(amp_df, n_samples=18000)

    # load ESM-2
    model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()

    # compute embeddings
    print("Computing training embeddings...")
    X_train_emb = compute_esm2_embeddings(X_train.tolist(), model, alphabet)
    print("Computing test embeddings...")
    X_test_emb = compute_esm2_embeddings(X_test.tolist(), model, alphabet)

    # train
    print("Training classifier...")
    clf = train_classifier(X_train_emb, y_train)

    # test a known AMP
    pred, prob = predict_amp("GIGKFLKKAKKFGKAFVKILKK", model, alphabet, clf)
    print(f"Prediction: {pred}, Probability: {prob:.3f}")

    y_pred = clf.predict(X_test_emb)
    y_prob = clf.predict_proba(X_test_emb)[:, 1]

    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.3f}")
