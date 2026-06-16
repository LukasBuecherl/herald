"""
database.py — AMP database retrieval and preprocessing for HERALD.

Provides utilities for downloading, caching, parsing, and combining curated
antimicrobial peptide sequence resources into a local processed AMP database.

This module currently supports APD6 FASTA downloads and locally downloaded
DBAASP peptide CSV files. The combined output is saved to data/processed as
amp_database.csv and can be used later for AMP-like peptide comparison,
baseline evaluation, and machine-learning model development.
"""

import glob
import os

import pandas as pd
import requests

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR_RAW = os.path.join(BASE_DIR, "data", "raw")
DATA_DIR_PROCESSED = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(DATA_DIR_RAW, exist_ok=True)
os.makedirs(DATA_DIR_PROCESSED, exist_ok=True)

cache_path_natural = os.path.join(DATA_DIR_RAW, "apd6_natural.fasta")
cache_path_animal = os.path.join(DATA_DIR_RAW, "apd6_animal.fasta")


def fetch_fasta(url, cache_path):
    """
    Fetch a FASTA file from a URL and cache it locally.
    Args:
        url (str): URL pointing to the FASTA file to download.
        cache_path (str): Local file path where the FASTA file should be saved.
    Returns:
        bool: True if the file exists locally or was downloaded successfully,
        False if the download failed.
    """
    if not os.path.exists(cache_path):
        try:
            r = requests.get(url, timeout=15, verify=True)
            r.raise_for_status()
            with open(cache_path, "wb") as f:
                f.write(r.content)
        except requests.exceptions.HTTPError as errh:
            print("HTTP Error")
            print(errh.args[0])
            return False
        except requests.exceptions.ReadTimeout:
            print("Time out")
            return False
        except requests.exceptions.ConnectionError:
            print("Connection error")
            return False
        except requests.exceptions.RequestException:
            print("Exception request")
            return False
    return True


def parse_fasta(cache_path):
    """
    Parse peptide sequences from a cached FASTA file.
    Args:
        cache_path (str): Local path to the FASTA file to parse.
    Returns:
        list[str]: List of peptide or protein sequences extracted from the FASTA file.
    """
    sequences = []
    current_sequence = ""
    with open(cache_path, "r") as f:
        for line in f:
            if line.startswith(">"):
                if current_sequence:
                    sequences.append(current_sequence)
                current_sequence = ""
            else:
                current_sequence += line.strip()
        if current_sequence:
            sequences.append(current_sequence)
    return sequences


def fetch_apd6():
    """
    Download, cache, and parse APD6 antimicrobial peptide FASTA files.
    Returns:
        pandas.DataFrame or None: DataFrame with columns "sequence" and "source"
        if APD6 files are available, or None if one of the downloads fails.
    """
    if not fetch_fasta(
        "https://aps.unmc.edu/assets/sequences/naturalAMPs_APD2024a.fasta",
        cache_path_natural,
    ):
        return None
    if not fetch_fasta(
        "https://aps.unmc.edu/assets/sequences/animalAMPs_APD2024a.fasta",
        cache_path_animal,
    ):
        return None
    sequences = parse_fasta(cache_path_natural) + parse_fasta(cache_path_animal)
    return pd.DataFrame({"sequence": sequences, "source": "APD6"})


def load_dbaasp():
    """
    Load locally downloaded DBAASP peptide CSV files.
    Returns:
        pandas.DataFrame: DataFrame with columns "sequence" and "source"
        containing peptide sequences from DBAASP.
    """
    csv_files = glob.glob(os.path.join(DATA_DIR_RAW, "peptides*.csv"))
    dataframes = [pd.read_csv(file) for file in csv_files]
    combined_df = pd.concat(dataframes, ignore_index=True)
    combined_df = combined_df.rename(columns={"SEQUENCE": "sequence"})
    combined_df["source"] = "DBAASP"
    df = combined_df[["sequence", "source"]].copy()
    df = df.dropna(subset=["sequence"])
    df = df[df["sequence"] != ""]
    return df


def build_amp_database():
    """
    Build a combined AMP sequence database from APD6 and DBAASP.
    Returns:
        pandas.DataFrame: Deduplicated AMP database with columns "sequence"
        and "source", also saved to data/processed/amp_database.csv.
    """
    dbaasp_df = load_dbaasp()
    apd6_df = fetch_apd6()

    if apd6_df is None:
        print("Warning: APD6 download failed, building database from DBAASP only.")
        amp_database = dbaasp_df
    else:
        amp_database = pd.concat([dbaasp_df, apd6_df], ignore_index=True)

    amp_database["sequence"] = amp_database["sequence"].str.upper()
    amp_database = amp_database.drop_duplicates(subset=["sequence"])
    amp_database = amp_database[
        amp_database["sequence"].str.match("^[ACDEFGHIKLMNPQRSTVWY]+$")
    ]
    amp_database = amp_database[amp_database["sequence"].str.len() >= 4]

    output_path = os.path.join(DATA_DIR_PROCESSED, "amp_database.csv")
    amp_database.to_csv(output_path, index=False)
    return amp_database
