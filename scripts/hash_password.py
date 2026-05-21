"""Utilitaire CLI pour générer un hash bcrypt.

Usage:
    python scripts/hash_password.py <mot_de_passe>
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.auth.password import hash_password


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python scripts/hash_password.py <mot_de_passe>", file=sys.stderr)
        sys.exit(1)
    plain = sys.argv[1]
    print(hash_password(plain))


if __name__ == "__main__":
    main()
