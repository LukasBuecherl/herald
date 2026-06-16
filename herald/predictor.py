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

import pandas as pd

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
