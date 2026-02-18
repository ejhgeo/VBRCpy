"""
Utility functions for data fetching and project setup.

Translated from MATLAB Projects/bayesian_fitting/fetch_data.m
"""

import os
import urllib.request
import zipfile
from pathlib import Path
from typing import List, Optional


BASE_URL = 'https://github.com/vbr-calc/vbrPublicData/raw/master/LAB_fitting_bayesian/data'

# Package-relative default: vbrc_V2Tpy/ (parent of bayesian_fitting_py/)
_PACKAGE_DIR = Path(__file__).resolve().parent          # bayesian_fitting_py/
_DEFAULT_DATA_PARENT = str(_PACKAGE_DIR.parent)         # vbrc_V2Tpy/


def build_project_directories(data_dir: str = './data') -> None:
    """
    Create necessary directory structure for the project.

    Parameters
    ----------
    data_dir : str
        Base data directory
    """
    subdirs = [
        'LAB_models',
        'plate_VBR',
        'Q_models',
        'vel_models',
        'tmp',
    ]
    
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    
    for subdir in subdirs:
        Path(data_dir, subdir).mkdir(parents=True, exist_ok=True)


def fetch_one_file(url: str, local_file: str) -> bool:
    """
    Download a file from a URL.

    Parameters
    ----------
    url : str
        URL to download from
    local_file : str
        Local path to save the file

    Returns
    -------
    bool
        True if successful, False otherwise
    """
    try:
        print(f"    Downloading {url}...")
        urllib.request.urlretrieve(url, local_file)
        return True
    except Exception as e:
        print(f"    Failed to download: {e}")
        return False


def fetch_data(data_dir_parent: Optional[str] = None) -> None:
    """
    Set up data directory and fetch all required data files.

    Parameters
    ----------
    data_dir_parent : str, optional
        Parent directory for the data folder.  When *None* (the default),
        data is placed under the package install directory
        (``vbrc_V2Tpy/data/``).
    """
    if data_dir_parent is None:
        data_dir_parent = _DEFAULT_DATA_PARENT
    data_dir = os.path.join(data_dir_parent, 'data')
    build_project_directories(data_dir)
    
    # Files to check/fetch
    files = [
        {
            'dir': 'vel_models',
            'fname': 'Shen_Ritzwoller_2016.mat',
            'zipped': True,
        },
        {
            'dir': 'vel_models',
            'fname': 'Porter_Liu_Holt_2015.mat',
            'zipped': True,
        },
        {
            'dir': 'Q_models',
            'fname': 'Dalton_Ekstrom_2008.mat',
            'zipped': False,
        },
        {
            'dir': 'LAB_models',
            'fname': 'HopperFischer2018.mat',
            'zipped': False,
        },
        {
            'dir': 'plate_VBR',
            'fname': 'sweep_log_gs.mat',
            'zipped': False,
        },
    ]
    
    # Check which files need to be fetched
    files_to_fetch = []
    for f in files:
        filepath = os.path.join(data_dir, f['dir'], f['fname'])
        if not os.path.exists(filepath):
            files_to_fetch.append(f)
    
    if not files_to_fetch:
        print("All data files present.")
        return
    
    print(f"Attempting to fetch {len(files_to_fetch)} missing files...")
    
    for f in files_to_fetch:
        dest_file = os.path.join(data_dir, f['dir'], f['fname'])
        
        print(f"Fetching {f['fname']}...")
        
        if f['zipped']:
            # Download zip file
            name_no_ext = os.path.splitext(f['fname'])[0]
            url = f"{BASE_URL}/{f['dir']}/{name_no_ext}.zip"
            tmp_file = os.path.join(data_dir, f['dir'], f'{name_no_ext}.zip')
            
            success = fetch_one_file(url, tmp_file)
            
            if success:
                try:
                    with zipfile.ZipFile(tmp_file, 'r') as zip_ref:
                        zip_ref.extractall(os.path.join(data_dir, f['dir']))
                    os.remove(tmp_file)
                    print(f"    Extracted {f['fname']}")
                except Exception as e:
                    print(f"    Failed to extract: {e}")
        else:
            url = f"{BASE_URL}/{f['dir']}/{f['fname']}"
            success = fetch_one_file(url, dest_file)
            
            if success:
                print(f"    Saved {f['fname']}")
    
    # Verify
    missing = []
    for f in files:
        filepath = os.path.join(data_dir, f['dir'], f['fname'])
        if not os.path.exists(filepath):
            missing.append(f['fname'])
    
    if missing:
        print("\n⚠️  Some files could not be fetched:")
        for fname in missing:
            print(f"    - {fname}")
        print("\nTo get the data manually:")
        print("1. Visit https://github.com/vbr-calc/vbrPublicData")
        print("2. Download the repository")
        print("3. Copy contents of LAB_fitting_bayesian/data to ./data/")
    else:
        print("\n✓ All data files successfully fetched!")


def _cli_main() -> None:
    """Entry point for ``fetch-vbr-data`` console script and ``python -m``."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Fetch required data files')
    parser.add_argument(
        '--data-dir', type=str, default=None,
        help=(
            'Parent directory for data folder '
            f'(default: package install dir, currently {_DEFAULT_DATA_PARENT})'
        ),
    )
    parser.add_argument(
        '-y', '--yes', action='store_true',
        help='Skip confirmation prompt and download immediately',
    )
    
    args = parser.parse_args()
    
    target = args.data_dir if args.data_dir is not None else _DEFAULT_DATA_PARENT
    data_path = os.path.join(target, 'data')

    if not args.yes:
        print(f"This will download ~180 MB of data files into:\n  {data_path}\n")
        answer = input("Proceed? [y/N] ").strip().lower()
        if answer not in ('y', 'yes'):
            print("Aborted.")
            raise SystemExit(0)

    fetch_data(args.data_dir)


if __name__ == '__main__':
    _cli_main()
