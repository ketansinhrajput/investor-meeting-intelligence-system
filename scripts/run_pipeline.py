#!/usr/bin/env python
"""Standalone script to run the pipeline on a transcript PDF."""

import argparse
import json
import sys
from pathlib import Path

# Add src to path for direct execution
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.graph import run_pipeline


def main():
    parser = argparse.ArgumentParser(
        description="Run the Call Transcript Intelligence pipeline"
    )
    parser.add_argument(
        "pdf_path",
        type=Path,
        help="Path to the earnings call transcript PDF",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output path for JSON report (default: <pdf_name>_report.json)",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        default=True,
        help="Pretty-print JSON output",
    )

    args = parser.parse_args()

    if not args.pdf_path.exists():
        print(f"Error: File not found: {args.pdf_path}")
        sys.exit(1)

    output_path = args.output
    if output_path is None:
        output_path = args.pdf_path.with_suffix("").with_suffix("_report.json")

    print(f"Processing: {args.pdf_path}")
    print(f"Output: {output_path}")

    try:
        report = run_pipeline(str(args.pdf_path))

        with open(output_path, "w", encoding="utf-8") as f:
            if args.pretty:
                json.dump(report, f, indent=2, ensure_ascii=False, default=str)
            else:
                json.dump(report, f, ensure_ascii=False, default=str)

        print(f"\nReport saved to: {output_path}")

        # Print summary
        qa_count = len(report.get("qa_units", []))
        stmt_count = len(report.get("strategic_statements", []))
        topic_count = len(report.get("topic_summaries", []))

        print(f"\nSummary:")
        print(f"  Q&A Units: {qa_count}")
        print(f"  Strategic Statements: {stmt_count}")
        print(f"  Topics: {topic_count}")

    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
