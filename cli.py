"""Command-line interface for ModelIO Lab.

Examples:
  python cli.py analyze "Great sound, weak battery" --provider fake --mode strict
  python cli.py analyze --native --provider openai
  python cli.py stress
  python cli.py swap --providers fake openai anthropic
  python cli.py footprint
"""

from __future__ import annotations

import argparse
import json
import sys

from modelio_lab.benchmark import dependency_footprint, load_sample_reviews, loc_comparison, model_swap, parser_stress_test
from modelio_lab.chains import analyze_review
from modelio_lab.config import PARSER_MODES
from modelio_lab.native_baseline import analyze_native


def print_table(rows: list[dict]) -> None:
    if not rows:
        print("(no rows)")
        return
    headers = list(rows[0])
    widths = {h: max(len(h), *(len(str(r.get(h))) for r in rows)) for h in headers}
    print("  ".join(h.ljust(widths[h]) for h in headers))
    print("  ".join("-" * widths[h] for h in headers))
    for row in rows:
        print("  ".join(str(row.get(h)).ljust(widths[h]) for h in headers))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="modelio-lab", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    analyze = sub.add_parser("analyze", help="Analyze one review")
    analyze.add_argument("review", nargs="?", help="Review text (defaults to the first sample)")
    analyze.add_argument("--provider", default="fake")
    analyze.add_argument("--mode", choices=PARSER_MODES, default="robust")
    analyze.add_argument("--retries", type=int, default=1)
    analyze.add_argument("--native", action="store_true", help="Use the native SDK baseline instead")
    analyze.add_argument("--trace", action="store_true", help="Print the request/response trace")

    sub.add_parser("stress", help="Parser stress test")

    swap = sub.add_parser("swap", help="Run one review across providers")
    swap.add_argument("--providers", nargs="+", default=["fake"])
    swap.add_argument("--mode", choices=PARSER_MODES, default="robust")
    swap.add_argument("--review")

    sub.add_parser("footprint", help="Dependency and LOC comparison")

    args = parser.parse_args(argv)
    default_review = load_sample_reviews()[0]["text"]

    if args.command == "analyze":
        review = args.review or default_review
        res = analyze_native(review, args.provider) if args.native else analyze_review(review, args.provider, args.mode, args.retries)
        print(f"pipeline={res.pipeline} provider={res.provider} mode={res.parser_mode} ok={res.ok} "
              f"attempts={res.attempts} latency={res.latency_ms:.0f}ms")
        print(json.dumps(res.result, indent=2) if res.ok else f"error: {res.error}")
        if args.trace and res.trace:
            print(json.dumps(res.trace, indent=2))
        return 0 if res.ok else 1

    if args.command == "stress":
        rows = parser_stress_test()
        print_table(rows)
        print(f"\nstrict: {sum(r['strict'] for r in rows)}/{len(rows)}  robust: {sum(r['robust'] for r in rows)}/{len(rows)}")
        return 0

    if args.command == "swap":
        print_table(model_swap(args.review or default_review, args.providers, args.mode))
        return 0

    if args.command == "footprint":
        print_table(dependency_footprint())
        print()
        print_table(loc_comparison())
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
