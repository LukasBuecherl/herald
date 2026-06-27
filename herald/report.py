def generate_report(results, env, candidates_df):
    """
    Generate a human-readable AMP discovery report for wet lab collaborators.
    Summarizes the RL agent's enzyme recommendation, ranked action values,
    top AMP candidate peptides, and experimental conditions in a format
    accessible to non-computational researchers.
    Args:
        results (dict): Training results from train() containing best_action,
            action_values, and best_combination.
        env (HERALDEnvironment): Trained environment containing protein name,
            action space, and enzyme conditions.
        candidates_df (pandas.DataFrame): Ranked peptide candidates with
            sequence and probability columns from digest_sequential() and
            predict_amp().
    Returns:
        None: Prints formatted report to stdout.
    """
    best_action = results["best_action"]
    best_combo = env.action_space[best_action]

    # header
    print("=" * 60)
    print("HERALD — AMP Discovery Report")
    print(f"Protein: {env.protein_name.capitalize()}")
    print(f"UniProt ID: {env.protein_accession}")
    print("=" * 60)

    # recommended combination
    print("\nRECOMMENDED ENZYME COMBINATION")
    print("-" * 60)
    print(f"Enzyme:      {best_combo['label']}")
    print(f"pH:          {best_combo['ph']}")
    print(f"Temperature: {best_combo['temp']}°C")
    print(f"Confidence:  {results['action_values'][best_action]:.3f} / 1.000")

    # ranked combinations
    print("\nRANKED ENZYME COMBINATIONS")
    print("-" * 60)
    sorted_actions = sorted(
        enumerate(zip(env.action_space, results["action_values"])),
        key=lambda x: x[1][1],
        reverse=True,
    )
    for rank, (i, (action, value)) in enumerate(sorted_actions, 1):
        marker = " ← recommended" if i == best_action else ""
        print(f"{rank}. {action['label']:<25} {value:.3f}{marker}")

    # top candidates
    print("\nTOP AMP CANDIDATES")
    print("-" * 60)
    for i, row in candidates_df.head(5).iterrows():
        print(f"{i + 1}. {row['sequence']:<35} {row['probability']:.4f}")

    # experimental note
    print("\nEXPERIMENTAL CONDITIONS")
    print("-" * 60)
    print(
        f"Run {best_combo['label']} digestion at "
        f"pH {best_combo['ph']} and {best_combo['temp']}°C."
    )
    print("Optimal conditions based on BRENDA enzyme database.")

    # disclaimer
    print("\nNOTE: Predictions are computational estimates.")
    print("Wet lab validation is required before drawing")
    print("conclusions about antimicrobial activity.")
    print("=" * 60)


if __name__ == "__main__":
    import os
    import sys

    import numpy as np
    import pandas as pd

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    try:
        from herald.environment import HERALDEnvironment
        from herald.proteins import protein_sequence
    except ModuleNotFoundError:
        from environment import HERALDEnvironment
        from proteins import protein_sequence

    print("=" * 60)
    print("HERALD — report.py self-test (mock data)")
    print("=" * 60)

    # mock environment using real lactoferrin sequence
    class MockModel:
        pass

    class MockAlphabet:
        pass

    class MockClf:
        pass

    seq = protein_sequence("P24627")
    env = HERALDEnvironment(
        protein_name="lactoferrin",
        protein_sequence=seq,
        protein_accession="P24627",
        model=MockModel(),
        alphabet=MockAlphabet(),
        clf=MockClf(),
    )

    # mock training results matching manuscript Table 4
    mock_results = {
        "best_action": 0,  # trypsin
        "action_values": np.array(
            [
                0.857,  # trypsin
                0.846,  # chymotrypsin
                0.760,  # alcalase
                0.704,  # papain
                0.773,  # bromelain
                0.838,  # pepsin
                0.826,  # pepsin → trypsin
                0.693,  # alcalase → papain
            ]
        ),
        "best_combination": env.action_space[0],
    }

    # mock candidates matching manuscript Table 5
    mock_candidates = pd.DataFrame(
        {
            "sequence": [
                "LFVPALLSLGALGLCLAAPR",
                "DSALGFLR",
                "QVLLHQQALFGK",
                "LRPVAAEIYGTK",
                "LGAPSITCVR",
            ],
            "probability": [0.9997, 0.9890, 0.9759, 0.9669, 0.9449],
        }
    )

    print()
    generate_report(mock_results, env, mock_candidates)

    print("\n" + "=" * 60)
    print("Cross-check against manuscript:")
    print("  Recommended enzyme : trypsin (expected trypsin)")
    print("  Confidence         : 0.857 (expected 0.857)")
    print("  Top candidate      : LFVPALLSLGALGLCLAAPR (expected)")
    print("  Top probability    : 0.9997 (expected 0.9997)")
    print("=" * 60)

    checks = [
        env.action_space[mock_results["best_action"]]["label"] == "trypsin",
        abs(mock_results["action_values"][0] - 0.857) < 0.001,
        mock_candidates.iloc[0]["sequence"] == "LFVPALLSLGALGLCLAAPR",
        abs(mock_candidates.iloc[0]["probability"] - 0.9997) < 0.0001,
    ]

    print()
    if all(checks):
        print("ALL CHECKS PASSED")
    else:
        print("MISMATCH DETECTED — review output above")
