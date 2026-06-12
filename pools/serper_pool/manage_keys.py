from __future__ import annotations

import argparse
from pathlib import Path
import sys

from serper_pool.config import PoolSettings
from serper_pool.store import KeyStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage Serper API keys in the local key pool.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    add_parser = subparsers.add_parser("add", help="Add one or more keys.")
    add_parser.add_argument("keys", nargs="+", help="Serper API keys.")
    add_parser.add_argument("--label", help="Optional label for a single key.")
    add_parser.add_argument(
        "--label-prefix",
        default="serper",
        help="Label prefix when adding multiple keys.",
    )

    import_parser = subparsers.add_parser("import", help="Import keys from a text file.")
    import_parser.add_argument("file", help="Each line: key or label,key or label<TAB>key.")

    subparsers.add_parser("list", help="List keys and status.")
    subparsers.add_parser("stats", help="Show summary counts.")

    enable_parser = subparsers.add_parser("enable", help="Enable a key by id.")
    enable_parser.add_argument("id", type=int)

    disable_parser = subparsers.add_parser("disable", help="Disable a key by id.")
    disable_parser.add_argument("id", type=int)

    delete_parser = subparsers.add_parser("delete", help="Delete a key by id.")
    delete_parser.add_argument("id", type=int)
    return parser


def load_entries(path: Path) -> list[tuple[str | None, str]]:
    entries: list[tuple[str | None, str]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "\t" in line:
            label, key = (part.strip() for part in line.split("\t", 1))
        elif "," in line:
            label, key = (part.strip() for part in line.split(",", 1))
        else:
            label, key = None, line
        if key:
            entries.append((label or None, key))
    return entries


def print_rows(rows: list[dict]) -> None:
    if not rows:
        print("No keys in pool.")
        return

    header = (
        f"{'ID':<4} {'STATUS':<10} {'LABEL':<18} {'KEY':<16} "
        f"{'SUCCESS':<8} {'TOTAL':<8} {'FAIL':<6} {'COOLDOWN':<22} LAST_ERROR"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['id']:<4} "
            f"{row['status']:<10} "
            f"{(row.get('label') or '-'): <18} "
            f"{row['masked_key']:<16} "
            f"{row.get('success_count', 0):<8} "
            f"{row.get('total_requests', 0):<8} "
            f"{row.get('consecutive_failures', 0):<6} "
            f"{str(row.get('cooldown_until') or '-'): <22} "
            f"{row.get('last_error') or '-'}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    settings = PoolSettings.from_env()
    store = KeyStore(
        settings.db_path,
        max_success_per_key=settings.max_success_per_key,
    )
    store.initialize()

    if args.command == "add":
        if len(args.keys) > 1 and args.label:
            parser.error("--label can only be used when adding a single key.")
        for index, api_key in enumerate(args.keys, start=1):
            label = args.label
            if len(args.keys) > 1:
                label = f"{args.label_prefix}-{index}"
            row = store.add_key(api_key, label)
            print(
                f"Added id={row['id']} label={row.get('label') or '-'} key={row['masked_key']}"
            )
        return 0

    if args.command == "import":
        file_path = Path(args.file)
        entries = load_entries(file_path)
        rows = store.import_keys(entries)
        print(f"Imported {len(rows)} keys from {file_path}")
        return 0

    if args.command == "list":
        print_rows(store.list_keys())
        return 0

    if args.command == "stats":
        summary = store.summary()
        print(summary)
        return 0

    try:
        if args.command == "enable":
            store.enable_key(args.id)
            print(f"Enabled key id={args.id}")
            return 0
        if args.command == "disable":
            store.disable_key(args.id)
            print(f"Disabled key id={args.id}")
            return 0
        if args.command == "delete":
            store.delete_key(args.id)
            print(f"Deleted key id={args.id}")
            return 0
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    parser.error("Unknown command.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
