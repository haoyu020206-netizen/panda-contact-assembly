from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fail when the contact-recovery benchmark misses its acceptance criteria."
    )
    parser.add_argument("summary", type=Path)
    args = parser.parse_args()

    with args.summary.open(encoding="utf-8") as handle:
        result = json.load(handle)
    acceptance = result.get("acceptance", {})
    checks = acceptance.get("checks", {})
    failed = [name for name, passed in checks.items() if not passed]
    if not acceptance.get("passed") or failed:
        details = ", ".join(failed) if failed else "acceptance block missing or invalid"
        raise SystemExit(f"contact-recovery acceptance failed: {details}")
    print(f"contact-recovery acceptance passed ({len(checks)} checks)")


if __name__ == "__main__":
    main()
