"""Download NSL-KDD and UNSW-NB15 dataset files."""

import argparse
import os
import urllib.request
import sys

NSL_KDD_URLS = {
    "KDDTrain+.txt": [
        "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt",
        "https://raw.githubusercontent.com/jmnwong/NSL-KDD-Dataset/master/KDDTrain%2B.txt",
    ],
    "KDDTest+.txt": [
        "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.txt",
        "https://raw.githubusercontent.com/jmnwong/NSL-KDD-Dataset/master/KDDTest%2B.txt",
    ],
}

UNSW_NB15_URLS = {
    "UNSW_NB15_training-set.csv": [
        "https://raw.githubusercontent.com/Nir-J/ML-Projects/master/UNSW-Network_Packet_Classification/UNSW_NB15_training-set.csv",
    ],
    "UNSW_NB15_testing-set.csv": [
        "https://raw.githubusercontent.com/Nir-J/ML-Projects/master/UNSW-Network_Packet_Classification/UNSW_NB15_testing-set.csv",
        "https://raw.githubusercontent.com/ishaak15/UNSW-IDS-Feature-Selection/main/UNSW_NB15_testing-set.csv",
    ],
}


def download_file(filename, urls, dest_dir):
    dest = os.path.join(dest_dir, filename)
    if os.path.exists(dest):
        print(f"  {filename} already exists, skipping.")
        return True

    for url in urls:
        try:
            print(f"  Downloading {filename} from {url}...")
            urllib.request.urlretrieve(url, dest)
            size_kb = os.path.getsize(dest) / 1024
            print(f"  Saved {filename} ({size_kb:.0f} KB)")
            return True
        except Exception as e:
            print(f"  Failed: {e}")

    print(f"  ERROR: Could not download {filename} from any source.")
    return False


def download_dataset(name, urls, dest_dir):
    print(f"Downloading {name} dataset...")
    success = True
    for filename, file_urls in urls.items():
        if not download_file(filename, file_urls, dest_dir):
            success = False
    if success:
        print(f"{name} — all files downloaded successfully.")
    else:
        print(f"{name} — some downloads failed. Please download manually.")
    return success


def main():
    parser = argparse.ArgumentParser(description='Download IDS datasets')
    parser.add_argument('--dataset', type=str, default='all',
                        choices=['nsl-kdd', 'unsw-nb15', 'all'])
    args = parser.parse_args()

    dest_dir = os.path.dirname(os.path.abspath(__file__))
    success = True

    if args.dataset in ('nsl-kdd', 'all'):
        success &= download_dataset('NSL-KDD', NSL_KDD_URLS, dest_dir)

    if args.dataset in ('unsw-nb15', 'all'):
        success &= download_dataset('UNSW-NB15', UNSW_NB15_URLS, dest_dir)

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
