#!/usr/bin/env python
"""Validate a generated report against the JSON schema."""

import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Validate a report against the JSON schema"
    )
    parser.add_argument(
        "report_path",
        type=Path,
        help="Path to the JSON report to validate",
    )

    args = parser.parse_args()

    if not args.report_path.exists():
        print(f"Error: File not found: {args.report_path}")
        sys.exit(1)

    # Load report
    try:
        with open(args.report_path, "r", encoding="utf-8") as f:
            report = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON: {e}")
        sys.exit(1)

    # Load schema
    schema_path = Path(__file__).parent.parent / "schemas" / "report_schema.json"
    if not schema_path.exists():
        print(f"Error: Schema not found: {schema_path}")
        sys.exit(1)

    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)

    # Validate
    try:
        import jsonschema

        jsonschema.validate(report, schema)
        print("Validation successful! Report conforms to schema.")

    except ImportError:
        print("Warning: jsonschema not installed. Skipping validation.")
        print("Install with: pip install jsonschema")

    except jsonschema.ValidationError as e:
        print(f"Validation failed: {e.message}")
        print(f"Path: {' -> '.join(str(p) for p in e.absolute_path)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
