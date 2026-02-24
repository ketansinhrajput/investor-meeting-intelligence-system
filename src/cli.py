"""Command-line interface for the Call Transcript Intelligence System."""

import json
import sys
from pathlib import Path

import structlog
import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, TextColumn
from rich.table import Table

from src.pipeline.graph import run_pipeline

# Configure structlog for CLI
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

app = typer.Typer(
    name="cti",
    help="Call Transcript Intelligence System - Analyze earnings call transcripts",
    add_completion=False,
)
console = Console()


@app.command()
def analyze(
    pdf_path: Path = typer.Argument(
        ...,
        help="Path to the earnings call transcript PDF",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    output: Path = typer.Option(
        None,
        "--output",
        "-o",
        help="Output file path for the JSON report (default: <pdf_name>_report.json)",
    ),
    pretty: bool = typer.Option(
        True,
        "--pretty/--compact",
        help="Pretty-print JSON output",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose logging",
    ),
) -> None:
    """Analyze an earnings call transcript and generate a structured report."""
    # Set log level
    if verbose:
        import logging

        logging.basicConfig(level=logging.DEBUG)
    else:
        import logging

        logging.basicConfig(level=logging.WARNING)

    console.print(
        Panel.fit(
            "[bold blue]Call Transcript Intelligence System[/bold blue]\n"
            "Analyzing earnings call transcript...",
            border_style="blue",
        )
    )

    console.print(f"\n[dim]Input:[/dim] {pdf_path}")

    # Determine output path
    if output is None:
        output = pdf_path.with_suffix("").with_suffix("_report.json")

    console.print(f"[dim]Output:[/dim] {output}\n")

    try:
        console.print("[yellow]Processing transcript... (this may take a few minutes)[/yellow]\n")

        # Run pipeline
        report = run_pipeline(str(pdf_path))

        console.print("[green]Processing complete![/green]")

        # Write output
        with open(output, "w", encoding="utf-8") as f:
            if pretty:
                json.dump(report, f, indent=2, ensure_ascii=False, default=str)
            else:
                json.dump(report, f, ensure_ascii=False, default=str)

        # Display summary
        _display_summary(report)

        console.print(f"\n[green]Report saved to:[/green] {output}")

    except Exception as e:
        console.print(f"\n[red]Error:[/red] {e}")
        if verbose:
            console.print_exception()
        sys.exit(1)


@app.command()
def validate(
    report_path: Path = typer.Argument(
        ...,
        help="Path to the JSON report to validate",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
) -> None:
    """Validate a generated report against the JSON schema."""
    try:
        import jsonschema

        # Load report
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)

        # Load schema
        schema_path = Path(__file__).parent.parent / "schemas" / "report_schema.json"
        with open(schema_path, "r", encoding="utf-8") as f:
            schema = json.load(f)

        # Validate
        jsonschema.validate(report, schema)

        console.print("[green]Validation successful![/green] Report conforms to schema.")

    except jsonschema.ValidationError as e:
        console.print(f"[red]Validation failed:[/red] {e.message}")
        console.print(f"[dim]Path:[/dim] {' -> '.join(str(p) for p in e.absolute_path)}")
        sys.exit(1)

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)


@app.command()
def info() -> None:
    """Display system information and configuration."""
    from src import __version__
    from src.config.settings import get_settings

    settings = get_settings()

    console.print(
        Panel.fit(
            "[bold blue]Call Transcript Intelligence System[/bold blue]",
            border_style="blue",
        )
    )

    table = Table(show_header=False, box=None)
    table.add_column("Setting", style="dim")
    table.add_column("Value")

    table.add_row("Version", __version__)
    table.add_row("LLM Model", settings.llm_model_name)
    table.add_row("Ollama URL", settings.llm_ollama_base_url)
    table.add_row("Temperature", str(settings.llm_temperature))
    table.add_row("Context Window", str(settings.llm_num_ctx))
    table.add_row("Chunk Target", f"{settings.chunk_target_tokens} tokens")
    table.add_row("Chunk Overlap", f"{settings.chunk_overlap_tokens} tokens")

    console.print(table)


def _display_summary(report: dict) -> None:
    """Display a summary of the analysis results.

    Args:
        report: The generated report dict.
    """
    console.print("\n[bold]Analysis Summary[/bold]")
    console.print("-" * 40)

    # Call metadata
    metadata = report.get("call_metadata", {})
    if metadata.get("company_name"):
        console.print(f"[dim]Company:[/dim] {metadata['company_name']}")
    if metadata.get("fiscal_quarter") and metadata.get("fiscal_year"):
        console.print(
            f"[dim]Period:[/dim] {metadata['fiscal_quarter']} {metadata['fiscal_year']}"
        )

    console.print(f"[dim]Pages:[/dim] {metadata.get('total_pages', 'N/A')}")

    # Counts
    table = Table(show_header=False, box=None)
    table.add_column("Metric", style="dim")
    table.add_column("Count", justify="right")

    table.add_row("Participants", str(metadata.get("participant_count", 0)))
    table.add_row("Q&A Units", str(len(report.get("qa_units", []))))
    table.add_row("Strategic Statements", str(len(report.get("strategic_statements", []))))
    table.add_row("Topics Identified", str(len(report.get("topic_summaries", []))))

    console.print(table)

    # Top topics
    topics = report.get("topic_summaries", [])[:5]
    if topics:
        console.print("\n[bold]Top Topics[/bold]")
        for i, topic in enumerate(topics, 1):
            console.print(f"  {i}. {topic.get('topic_name')} ({topic.get('mention_count')} mentions)")

    # Processing info
    proc_meta = report.get("processing_metadata", {})
    errors = proc_meta.get("errors", [])
    warnings = proc_meta.get("warnings", [])

    if errors:
        console.print(f"\n[yellow]Warnings/Errors:[/yellow] {len(errors) + len(warnings)}")

    console.print(
        f"\n[dim]Processed in {proc_meta.get('total_duration_seconds', 0):.1f}s "
        f"with {proc_meta.get('llm_calls_made', 0)} LLM calls[/dim]"
    )


if __name__ == "__main__":
    app()
