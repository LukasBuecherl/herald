"""
screening.py — Protein-enzyme screening workflow for HERALD.

Combines protein sequence retrieval, in silico digestion, peptide feature
calculation, and simple AMP-like scoring into a reusable screening workflow.

This module is used to compare protein-enzyme combinations and summarize which
combinations produce peptide profiles with higher exploratory AMP-like scores.
The scoring is currently rule-based and intended as a transparent baseline for
later comparison with curated AMP databases, machine-learning predictors, and
experimental validation.
"""

import pandas as pd

from herald.digestion import digest_sequence
from herald.enzymes import ENZYME_RULES
from herald.proteins import WHEY_PROTEINS, protein_sequence
from herald.scoring import peptide_features


def screen_combinations(
    proteins=None,
    enzymes=None,
    min_length=4,
    max_length=50,
):
    """
    Screen protein-enzyme combinations using the HERALD workflow.
    For each protein-enzyme combination, this function retrieves the protein sequence,
    simulates enzymatic digestion, computes peptide features, and summarizes AMP-like
    scoring metrics.

    Args:
        proteins (dict, optional): Mapping of protein names to UniProt accession IDs.
            If None, uses WHEY_PROTEINS.
        enzymes (dict, optional): Mapping of enzyme names to enzyme rule dictionaries.
            If None, uses ENZYME_RULES.
        min_length (int): Minimum peptide length retained after digestion.
        max_length (int): Maximum peptide length retained after digestion.

    Returns:
        pandas.DataFrame: Summary table of protein-enzyme screening results.
    """
    if proteins is None:
        proteins = WHEY_PROTEINS

    if enzymes is None:
        enzymes = ENZYME_RULES

    results = []

    for protein_name, accession_id in proteins.items():
        sequence = protein_sequence(accession_id)

        for enzyme_name in enzymes.keys():
            peptides = digest_sequence(
                sequence, enzyme_name, min_length=min_length, max_length=max_length
            )

            features = [peptide_features(peptide) for peptide in peptides]
            scores = [feature["simple_amp_score"] for feature in features]

            if len(scores) == 0:
                continue

            top_score = max(scores)
            top_peptide = peptides[scores.index(top_score)]
            num_score_ge_3 = sum(1 for score in scores if score >= 3)
            num_score_ge_4 = sum(1 for score in scores if score >= 4)
            avg_score = sum(scores) / len(scores)

            results.append(
                {
                    "protein": protein_name,
                    "accession_id": accession_id,
                    "enzyme": enzyme_name,
                    "min_length": min_length,
                    "max_length": max_length,
                    "num_peptides": len(peptides),
                    "max_score": top_score,
                    "top_peptide": top_peptide,
                    "num_score_ge_3": num_score_ge_3,
                    "num_score_ge_4": num_score_ge_4,
                    "avg_score": avg_score,
                }
            )

    return pd.DataFrame(results)
