"""
Preprocessing runner - execute from the project root:

    python preprocess.py [bcss] [celeba] [isic]

If no arguments are given, all three datasets are processed.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.preprocessing import bcss, celeba, isic

_PROCESSORS = {
    "bcss":   bcss.preprocess,
    "celeba": celeba.preprocess,
    "isic":   isic.preprocess,
}

if __name__ == "__main__":
    targets = sys.argv[1:] if len(sys.argv) > 1 else list(_PROCESSORS.keys())
    unknown = [t for t in targets if t not in _PROCESSORS]
    if unknown:
        print(f"Unknown dataset(s): {unknown}. Choose from: {list(_PROCESSORS)}")
        sys.exit(1)

    for name in targets:
        print(f"Preprocessing: {name.upper()}")
        _PROCESSORS[name]()

    print("\nAll done.")
