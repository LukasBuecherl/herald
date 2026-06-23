"""
proteins.py — UniProt sequence fetching and local caching for HERALD.

Provides protein_sequence() for fetching by accession ID and
WHEY_PROTEINS as a lookup table of target whey protein accession IDs.
"""

import os

import requests

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
os.makedirs(DATA_DIR, exist_ok=True)


def protein_sequence(acc_id):
    """
    Fetch a protein sequence from UniProt by accession ID.

    Checks local cache first. If not cached, fetches from the
    UniProt REST API and saves locally for future use.

    Args:
        acc_id (str): UniProt accession ID (e.g. 'P02754')

    Returns:
        str: Protein sequence as a single string, or None if fetch fails
    """
    cache_path = os.path.join(DATA_DIR, f"{acc_id}.txt")

    if not os.path.exists(cache_path):
        try:
            r = requests.get(
                f"https://rest.uniprot.org/uniprotkb/{acc_id}.fasta",
                timeout=15,
                verify=True,
            )
            r.raise_for_status()
        except requests.exceptions.HTTPError as errh:
            print("HTTP Error")
            print(errh.args[0])
            return None
        except requests.exceptions.ReadTimeout:
            print("Time out")
            return None
        except requests.exceptions.ConnectionError:
            print("Connection error")
            return None
        except requests.exceptions.RequestException:
            print("Exception request")
            return None

        # Strip FASTA header line (starts with '>') and join sequence lines
        clean_ID = "".join(r.text.splitlines()[1:])

        with open(cache_path, "w") as file:
            file.write(clean_ID)
        return clean_ID

    else:
        with open(cache_path, "r") as file:
            return file.read()


WHEY_PROTEINS = {
    "beta-lactoglobulin": "P02754",
    "alpha-lactalbumin": "P00711",
    "lactoferrin": "P24627",
    "BSA": "P02769",
}
