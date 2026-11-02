"""CLI: python -m engine.ml.fetch_weights [--all | name ...]"""
from __future__ import annotations

import argparse
import sys

from engine.ml.weights import MANIFEST, fetch, verify, weight_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m engine.ml.fetch_weights",
        description="Download and sha256-verify model weights.",
    )
    parser.add_argument("names", nargs="*", help=f"weights to fetch: {', '.join(MANIFEST)}")
    parser.add_argument("--all", action="store_true", help="fetch every weight in the manifest")
    args = parser.parse_args(argv)
    if args.all == bool(args.names):
        parser.error("give --all or one or more weight names, not both")
    names = list(MANIFEST) if args.all else args.names

    failed = 0
    for name in names:
        spec = MANIFEST.get(name)
        license_ = f" [{spec.license}]" if spec else ""
        try:
            if verify(name):
                status, path = "verified", weight_path(name)
            else:
                status, path = "downloaded", fetch(name)
        except Exception as exc:  # report every weight, then exit non-zero
            failed += 1
            reason = exc.args[0] if isinstance(exc, KeyError) and exc.args else exc
            print(f"{name}: failed{license_} {reason}")
        else:
            print(f"{name}: {status}{license_} {path}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
