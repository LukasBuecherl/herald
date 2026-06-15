"""
scoring.py — Simple peptide feature and baseline scoring calculations for HERALD.

This module computes first-pass peptide properties relevant for screening
hydrolysis products for AMP-like behavior. These features provide a transparent,
rule-based starting point before adding more advanced AMP predictors or
machine-learning models.

Current outputs include:
    - Peptide length
    - Approximate net charge
    - Hydrophobic residue fraction
    - Aromatic residue fraction
    - Simple rule-based AMP-like score

The calculations use simple residue-count approximations based on standard
one-letter amino-acid codes. The simple AMP-like score is an exploratory
baseline that rewards broad properties commonly associated with antimicrobial
peptides, including moderate length, positive charge, moderate hydrophobicity,
and aromatic content.

These values are useful for initial ranking and exploratory analysis, but they
are not validated predictors of antimicrobial activity. In later versions, these
functions may be replaced or supplemented with pH-dependent charge calculations,
hydrophobicity scales, curated AMP database models, external peptide-property
prediction tools, and experimental validation.
"""

POSITIVE_RESIDUES = {"K", "R", "H"}
NEGATIVE_RESIDUES = {"D", "E"}
HYDROPHOBIC_RESIDUES = {"A", "V", "I", "L", "M", "F", "W", "Y"}
AROMATIC_RESIDUES = {"F", "W", "Y"}


def net_charge(peptide):
    """
    Estimate peptide net charge using a simple residue-count approximation.

    Positive residues: K, R, H
    Negative residues: D, E
    """
    charge = 0

    for aa in peptide.upper():
        if aa in POSITIVE_RESIDUES:
            charge += 1
        elif aa in NEGATIVE_RESIDUES:
            charge -= 1

    return charge


def hydrophobic_fraction(peptide):
    """
    Compute the fraction of residues that are hydrophobic.
    """

    peptide = peptide.upper()

    if len(peptide) == 0:
        return 0

    count = 0

    for aa in peptide:
        if aa in HYDROPHOBIC_RESIDUES:
            count += 1

    return count / len(peptide)


def aromatic_fraction(peptide):
    """
    Compute the fraction of residues that are aromatic.
    """
    peptide = peptide.upper()

    if len(peptide) == 0:
        return 0

    count = 0

    for aa in peptide:
        if aa in AROMATIC_RESIDUES:
            count += 1

    return count / len(peptide)


def simple_amp_score(peptide):
    """
    Compute a simple rule-based AMP-like score.

    This score is an exploratory heuristic, not a trained AMP predictor.
    The thresholds reward broad AMP-like properties reported in the peptide
    design literature, including moderate length, positive charge, moderate
    hydrophobicity, and aromatic content. These thresholds should be refined
    later using AMP databases and experimental validation.
    """
    peptide = peptide.upper()
    score = 0

    length = len(peptide)
    charge = net_charge(peptide)
    hydrophobicity = hydrophobic_fraction(peptide)
    aromaticity = aromatic_fraction(peptide)

    if 8 <= length <= 30:
        score += 1

    if charge >= 1:
        score += 1
    if charge >= 2:
        score += 1

    if 0.3 <= hydrophobicity <= 0.7:
        score += 1

    if aromaticity > 0:
        score += 1

    return score


def peptide_features(peptide):
    """
    Compute simple AMP-relevant features for a peptide.
    """
    peptide = peptide.upper()
    return {
        "sequence": peptide,
        "length": len(peptide),
        "net_charge": net_charge(peptide),
        "hydrophobic_fraction": hydrophobic_fraction(peptide),
        "aromatic_fraction": aromatic_fraction(peptide),
        "simple_amp_score": simple_amp_score(peptide),
    }
