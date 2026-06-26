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

try:
    from herald.digestion import digest_sequence
    from herald.enzymes import ENZYME_RULES
    from herald.proteins import WHEY_PROTEINS, protein_sequence
    from herald.scoring import peptide_features, peptide_features_ml
except ModuleNotFoundError:
    from digestion import digest_sequence
    from enzymes import ENZYME_RULES
    from proteins import WHEY_PROTEINS, protein_sequence
    from scoring import peptide_features, peptide_features_ml


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


def screen_combinations_ml(
    proteins=None,
    enzymes=None,
    min_length=4,
    max_length=50,
    model=None,
    alphabet=None,
    clf=None,
):
    """
    Screen protein-enzyme combinations using ESM-2 AMP probability scoring.
    Extends screen_combinations() by replacing the rule-based simple_amp_score
    with ML-derived AMP probabilities from a pretrained ESM-2 classifier.
    For each protein-enzyme combination, peptides are digested, embedded using
    ESM-2, and scored using a logistic regression classifier trained on curated
    AMP databases.
    Args:
        proteins (dict, optional): Mapping of protein names to UniProt accession
            IDs. If None, uses WHEY_PROTEINS.
        enzymes (dict, optional): Mapping of enzyme names to cleavage rule
            dictionaries. If None, uses ENZYME_RULES.
        min_length (int): Minimum peptide length retained after digestion.
        max_length (int): Maximum peptide length retained after digestion.
        model: Pretrained ESM-2 model for sequence embedding.
        alphabet: ESM-2 alphabet object used for tokenization.
        clf: Fitted logistic regression classifier trained on AMP database.
    Returns:
        pandas.DataFrame: Summary table with one row per protein-enzyme
        combination, ranked by ML-derived AMP probability scores. Columns
        include num_predicted_amp, avg_probability, top_probability,
        and top_peptide.
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

            if len(peptides) == 0:
                continue

            features = [
                peptide_features_ml(peptide, model, alphabet, clf)
                for peptide in peptides
            ]
            probabilities = [feature["amp_probability"] for feature in features]
            predictions = [feature["amp_prediction"] for feature in features]

            num_predicted_amp = sum(predictions)
            avg_probability = sum(probabilities) / len(probabilities)
            top_probability = max(probabilities)
            top_peptide = peptides[probabilities.index(top_probability)]

            results.append(
                {
                    "protein": protein_name,
                    "accession_id": accession_id,
                    "enzyme": enzyme_name,
                    "min_length": min_length,
                    "max_length": max_length,
                    "num_peptides": len(peptides),
                    "num_predicted_amp": num_predicted_amp,
                    "avg_probability": avg_probability,
                    "top_probability": top_probability,
                    "top_peptide": top_peptide,
                }
            )

    return pd.DataFrame(results)


if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    try:
        from herald.enzymes import ENZYME_RULES
        from herald.proteins import WHEY_PROTEINS
    except ModuleNotFoundError:
        from enzymes import ENZYME_RULES
        from proteins import WHEY_PROTEINS

    print("=" * 60)
    print("HERALD — screening.py self-test (rule-based only)")
    print("=" * 60)

    df = screen_combinations()

    print(f"\nTotal combinations screened: {len(df)}")
    print(f"Expected: 24 (4 proteins x 6 enzymes)\n")

    print(
        f"{'Protein':<22} {'Enzyme':<14} {'Peptides':>8} {'MaxScore':>9} {'>=4':>5} {'>=3':>5} {'Avg':>6}"
    )
    print("-" * 70)
    for _, row in df.iterrows():
        print(
            f"{row['protein']:<22} {row['enzyme']:<14} "
            f"{row['num_peptides']:>8} {row['max_score']:>9} "
            f"{row['num_score_ge_4']:>5} {row['num_score_ge_3']:>5} "
            f"{row['avg_score']:>6.2f}"
        )

    print("\n" + "=" * 60)
    print("Cross-check against manuscript (Table 2):")
    print(
        "  Lactoferrin + trypsin      : expected 58 peptides, max 5, 9 >=4, 34 >=3, avg 2.47"
    )
    print(
        "  Lactoferrin + chymotrypsin : expected 67 peptides, max 5, 7 >=4, 26 >=3, avg 2.13"
    )
    print(
        "  Alpha-lactalbumin + trypsin: expected 10 peptides, max 5, 2 >=4,  5 >=3, avg 2.50"
    )
    print("=" * 60)

    checks = [
        ("lactoferrin", "trypsin", 58, 5, 9, 34, 2.47),
        ("lactoferrin", "chymotrypsin", 67, 5, 7, 26, 2.13),
        ("alpha-lactalbumin", "trypsin", 10, 5, 2, 5, 2.50),
    ]

    all_passed = True
    for protein, enzyme, exp_pep, exp_max, exp_ge4, exp_ge3, exp_avg in checks:
        row = df[(df["protein"] == protein) & (df["enzyme"] == enzyme)].iloc[0]
        ok = (
            row["num_peptides"] == exp_pep
            and row["max_score"] == exp_max
            and row["num_score_ge_4"] == exp_ge4
            and row["num_score_ge_3"] == exp_ge3
            and abs(row["avg_score"] - exp_avg) < 0.01
        )
        status = "✅" if ok else "❌"
        print(
            f"  {status} {protein} + {enzyme}: "
            f"got {row['num_peptides']} peptides, "
            f"max {row['max_score']}, "
            f"{row['num_score_ge_4']} >=4, "
            f"{row['num_score_ge_3']} >=3, "
            f"avg {row['avg_score']:.2f}"
        )
        if not ok:
            all_passed = False

    print()
    if all_passed:
        print("ALL CHECKS PASSED")
    else:
        print("MISMATCH DETECTED — review output above")
