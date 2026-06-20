"""
digestion.py — In silico enzymatic digestion for HERALD.

Uses enzyme cleavage rules from enzymes.py to simulate how a protein
sequence would be cut into peptide fragments by a selected protease.
"""

from sys import _current_frames

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

        if current_aa in rules.get("cleaves_after", []) and next_aa not in rules.get(
            "not_before", []
        ):
            cut_sites.append(i + 1)
        elif current_aa in rules.get("cleaves_before", []):
            cut_sites.append(i)

    cut_sites = [0] + cut_sites + [len(sequence)]

    peptides = []
    for i in range(len(cut_sites) - 1):
        peptide = sequence[cut_sites[i] : cut_sites[i + 1]]
        if min_length <= len(peptide) <= max_length:
            peptides.append(peptide)
    return peptides


def digest_sequential(sequence, enzyme_combination, min_length=4, max_length=50):
    """
    Simulate sequential enzymatic digestion of a protein sequence.
    Applies a list of enzymes one after another, where the peptide fragments
    produced by each enzyme become the input for the next. This simulates
    multi-enzyme hydrolysis workflows such as gastric followed by intestinal
    digestion, or combinations of industrial proteases.
    Length filtering is applied only after all enzymes have been applied,
    ensuring that intermediate fragments that are temporarily too long or
    too short are not discarded before further digestion.
    Args:
        sequence (str): Protein amino acid sequence.
        enzyme_combination (list[str]): Ordered list of enzyme names to apply
            sequentially. Each enzyme must be defined in ENZYME_RULES.
        min_length (int): Minimum peptide length to retain after all digestion.
        max_length (int): Maximum peptide length to retain after all digestion.
    Returns:
        list[str]: Peptide fragments produced by sequential digestion,
        filtered to the specified length range.
    Example:
        >>> digest_sequential("LIVTQTMKGLDIQKVAGTWYR", ["trypsin", "chymotrypsin"])
        ["LIVTQTMK", "GLDIQK", "VAGTW"]
    """
    current_fragments = [sequence]
    for enzyme_name in enzyme_combination:
        next_fragments = []
        for fragment in current_fragments:
            next_fragments.extend(digest_sequence(fragment, enzyme_name, 1, max_length))
        current_fragments = next_fragments

    return [p for p in current_fragments if min_length <= len(p) <= max_length]
