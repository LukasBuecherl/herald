"""
digestion.py — In silico enzymatic digestion for HERALD.

Uses enzyme cleavage rules from enzymes.py to simulate how a protein
sequence would be cut into peptide fragments by a selected protease.
"""

from herald.enzymes import ENZYME_RULES


def digest_sequence(sequence, enzyme_name, min_length=4, max_length=50):
    """
    Simulate enzymatic digestion of a protein sequence.

    Args:
        sequence (str): Protein amino acid sequence.
        enzyme_name (str): Name of enzyme in ENZYME_RULES.
        min_length (int): Minimum peptide length to keep.
        max_length (int): Maximum peptide length to keep.

    Returns:
        list: Peptide fragments produced by the simulated digestion.
    """
    sequence = sequence.upper()
    enzyme_name = enzyme_name.lower()

    if enzyme_name not in ENZYME_RULES:
        raise ValueError(f"Unknown enzyme: {enzyme_name}")

    rules = ENZYME_RULES[enzyme_name]
    cut_sites = []
    for i in range(len(sequence) - 1):
        current_aa = sequence[i]
        next_aa = sequence[i + 1]

        if current_aa in rules["cleaves_after"] and next_aa not in rules["not_before"]:
            cut_sites.append(i + 1)

    cut_sites = [0] + cut_sites + [len(sequence)]

    peptides = []
    for i in range(len(cut_sites) - 1):
        peptide = sequence[cut_sites[i] : cut_sites[i + 1]]
        if min_length <= len(peptide) <= max_length:
            peptides.append(peptide)
    return peptides
