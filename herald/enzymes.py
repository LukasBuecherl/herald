"""
enzymes.py — Enzyme cleavage rules and properties for HERALD.

Defines ENZYME_RULES, a lookup table of cleavage specificity,
optimal conditions, and metadata for each protease used in the
in silico digestion pipeline.

Each entry contains:
    cleaves_after (list): Amino acids after which the enzyme cuts
    not_before (list): Amino acids that block cleavage at the next position
    ph_optimum (float): Optimal pH for enzymatic activity
    temp_optimum_celsius (float): Optimal temperature in Celsius
    source (str): Biological source of the commercial enzyme
    food_grade (bool): Whether the enzyme is approved for food use
"""

# Cleavage rules follow the P1/P1' convention:
# P1 = residue before the cut site (cleaves_after)
# P1' = residue after the cut site (not_before blocks cleavage)
ENZYME_RULES = {
    "trypsin": {
        "cleaves_after": ["K", "R"],
        "not_before": ["P"],
        "ph_optimum": 7.8,
        "temp_optimum_celsius": 37,
        "source": "porcine/bovine pancreas",
        "food_grade": True,
    },
    "chymotrypsin": {
        "cleaves_after": ["F", "Y", "W", "L"],
        "not_before": ["P"],
        "ph_optimum": 8.0,
        "temp_optimum_celsius": 37,
        "source": "bovine pancreas",
        "food_grade": True,
    },
}
