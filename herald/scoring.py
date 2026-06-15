"""
scoring.py — Simple peptide feature calculations for HERALD.

This module computes first-pass peptide properties that are relevant for
screening hydrolysis products for AMP-like behavior. These features are
intended to provide a transparent, rule-based starting point before adding
more advanced AMP predictors or machine-learning models.

Current features include:
    - Peptide length
    - Approximate net charge
    - Hydrophobic residue fraction
    - Aromatic residue fraction

The calculations use simple residue-count approximations based on standard
one-letter amino-acid codes. These values are useful for initial ranking and
exploratory analysis, but they are not full physicochemical models. In later
versions, these functions may be replaced or supplemented with pH-dependent
charge calculations, hydrophobicity scales, or external peptide-property
prediction tools.
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
    }
