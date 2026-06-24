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
    "alcalase": {
        "cleaves_after": ["F", "L", "Y", "A", "V", "I", "M"],
        "not_before": [],
        "ph_optimum": 8.0,
        "temp_optimum_celsius": 55,
        "source": "bacillus licheniformis",
        "food_grade": True,
    },
    "papain": {
        "cleaves_after": ["R", "K", "F", "Y", "L", "V", "A"],
        "not_before": [],
        "ph_optimum": 6.5,
        "temp_optimum_celsius": 60,
        "source": "carica papaya",
        "food_grade": True,
    },
    "bromelain": {
        "cleaves_after": ["R", "K", "A", "Y", "G", "V"],
        "not_before": [],
        "ph_optimum": 7.0,
        "temp_optimum_celsius": 50,
        "source": "ananas comosus",
        "food_grade": True,
    },
    "pepsin": {
        "cleaves_before": ["F", "L", "W", "Y"],
        "ph_optimum": 2.0,
        "temp_optimum_celsius": 37,
        "source": "porcine stomach",
        "food_grade": True,
    },
}


def get_conditions(enzyme_names):
    """
    Extract optimal pH and temperature for an enzyme combination
    from ENZYME_RULES. For sequential combinations, returns lists.
    """
    if len(enzyme_names) == 1:
        enzyme = ENZYME_RULES[enzyme_names[0]]
        return enzyme["ph_optimum"], enzyme["temp_optimum_celsius"]
    else:
        ph = [ENZYME_RULES[e]["ph_optimum"] for e in enzyme_names]
        temp = [ENZYME_RULES[e]["temp_optimum_celsius"] for e in enzyme_names]
        return ph, temp


if __name__ == "__main__":
    print("=" * 60)
    print("HERALD — enzymes.py self-test")
    print("=" * 60)

    print(f"\nTotal enzymes defined: {len(ENZYME_RULES)}")
    print(f"Expected: 6\n")

    for name, rules in ENZYME_RULES.items():
        print(f"{name}")
        print(f"  Cleaves after  : {rules.get('cleaves_after', 'N/A')}")
        print(f"  Cleaves before : {rules.get('cleaves_before', 'N/A')}")
        print(f"  Not before     : {rules.get('not_before', 'N/A')}")
        print(f"  pH optimum     : {rules['ph_optimum']}")
        print(f"  Temp optimum   : {rules['temp_optimum_celsius']} °C")
        print(f"  Food grade     : {rules['food_grade']}")

    print("\n" + "=" * 60)
    print("Cross-check against manuscript (Section 2.2):")
    print("  Expected enzymes: trypsin, chymotrypsin, alcalase,")
    print("                    papain, bromelain, pepsin")
    print("  Trypsin cleaves after  : K, R (not before P)")
    print("  Chymotrypsin cleaves after: F, Y, W, L (not before P)")
    print("  Pepsin cleaves before  : F, L, W, Y")
    print("=" * 60)

    expected = ["trypsin", "chymotrypsin", "alcalase", "papain", "bromelain", "pepsin"]
    all_present = all(e in ENZYME_RULES for e in expected)
    all_food_grade = all(ENZYME_RULES[e]["food_grade"] for e in expected)

    if all_present and all_food_grade:
        print("ALL CHECKS PASSED")
    else:
        print("MISMATCH DETECTED — review output above")
