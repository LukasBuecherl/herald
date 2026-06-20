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
