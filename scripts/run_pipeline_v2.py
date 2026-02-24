#!/usr/bin/env python
"""Standalone script to run the v2 pipeline on a transcript PDF.

Pipeline v2 uses multi-stage architecture with:
- Boundary detection for high recall
- Speaker registry for canonical names
- Separate extraction and enrichment stages
"""

import argparse
import json
import sys
from pathlib import Path

# Add src to path for direct execution
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline_v2 import run_pipeline_v2, PipelineV2Error


def state_to_report(state) -> dict:
    """Convert PipelineV2State to output report dict."""
    report = {
        "report_version": "2.0",
        "source_file": state.source_file,
        "total_pages": state.total_pages,
    }

    # Metadata
    if state.metadata:
        report["call_metadata"] = {
            "company_name": state.metadata.company_name,
            "ticker_symbol": state.metadata.ticker_symbol,
            "fiscal_quarter": state.metadata.fiscal_quarter,
            "fiscal_year": state.metadata.fiscal_year,
            "call_date": str(state.metadata.call_date) if state.metadata.call_date else None,
            "call_title": state.metadata.call_title,
        }

    # Speaker registry
    if state.speaker_registry:
        report["speaker_registry"] = {
            speaker_id: {
                "canonical_name": info.canonical_name,
                "aliases": info.aliases,
                "role": info.role.value,
                "title": info.title,
                "company": info.company,
                "turn_count": info.turn_count,
            }
            for speaker_id, info in state.speaker_registry.speakers.items()
        }

    # Sections summary
    if state.boundary_result:
        report["sections"] = [
            {
                "section_id": s.section_id,
                "section_type": s.section_type.value,
                "start_page": s.start_page,
                "end_page": s.end_page,
                "detected_speakers": s.detected_speakers,
                "confidence": s.detection_confidence,
            }
            for s in state.boundary_result.sections
        ]
        report["section_stats"] = {
            "total_sections": state.boundary_result.total_sections,
            "qa_sections": state.boundary_result.qa_section_count,
            "coverage_percent": state.boundary_result.coverage_percent,
        }

    # Q&A units (enriched if available, otherwise raw)
    if state.enriched_qa_units:
        report["qa_units"] = [
            {
                "qa_id": qa.qa_id,
                "questioner": qa.questioner_name,
                "responders": qa.responder_names,
                "question_text": qa.question_text,
                "response_text": qa.response_text,
                "is_follow_up": qa.is_follow_up,
                "follow_up_of": qa.follow_up_of,
                "start_page": qa.start_page,
                "end_page": qa.end_page,
                "topics": [
                    {
                        "name": t.topic_name,
                        "category": t.topic_category,
                        "relevance": t.relevance_score,
                    }
                    for t in qa.topics
                ],
                "investor_intent": {
                    "intent": qa.investor_intent.primary_intent,
                    "confidence": qa.investor_intent.confidence,
                    "reasoning": qa.investor_intent.reasoning,
                } if qa.investor_intent else None,
                "response_posture": {
                    "posture": qa.response_posture.primary_posture,
                    "confidence": qa.response_posture.confidence,
                    "reasoning": qa.response_posture.reasoning,
                } if qa.response_posture else None,
                "key_evidence": [
                    {
                        "quote": e.quote,
                        "page": e.page_number,
                        "speaker": e.speaker_name,
                    }
                    for e in qa.key_evidence
                ],
                "summary": qa.summary,
            }
            for qa in state.enriched_qa_units
        ]
    elif state.qa_extraction_result:
        report["qa_units"] = [
            {
                "qa_id": qa.qa_id,
                "questioner": qa.questioner_name,
                "responders": qa.responder_names,
                "question_text": qa.question_text,
                "response_text": qa.response_text,
                "is_follow_up": qa.is_follow_up,
                "follow_up_of": qa.follow_up_of,
                "start_page": qa.start_page,
                "end_page": qa.end_page,
            }
            for qa in state.qa_extraction_result.qa_units
        ]

    # Strategic statements (enriched if available)
    if state.enriched_strategic_statements:
        report["strategic_statements"] = [
            {
                "statement_id": stmt.statement_id,
                "speaker": stmt.speaker_name,
                "text": stmt.text,
                "type": stmt.statement_type,
                "forward_looking": stmt.is_forward_looking,
                "page": stmt.page_number,
                "topics": [
                    {
                        "name": t.topic_name,
                        "category": t.topic_category,
                        "relevance": t.relevance_score,
                    }
                    for t in stmt.topics
                ],
                "summary": stmt.summary,
            }
            for stmt in state.enriched_strategic_statements
        ]
    elif state.strategic_extraction_result:
        report["strategic_statements"] = [
            {
                "statement_id": stmt.statement_id,
                "speaker": stmt.speaker_name,
                "text": stmt.text,
                "type": stmt.statement_type,
                "forward_looking": stmt.is_forward_looking,
                "page": stmt.page_number,
            }
            for stmt in state.strategic_extraction_result.statements
        ]

    # Processing metadata
    report["processing_metadata"] = {
        "processing_start": str(state.processing_start) if state.processing_start else None,
        "processing_end": str(state.processing_end) if state.processing_end else None,
        "stage_durations": state.stage_durations,
        "llm_calls_made": state.llm_calls_made,
        "validation_passed": state.validation_passed,
        "validation_issues": state.validation_issues,
        "errors": state.errors,
        "warnings": state.warnings,
    }

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Run the Call Transcript Intelligence Pipeline v2"
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
        help="Output path for JSON report (default: <pdf_name>_v2_report.json)",
    )
    parser.add_argument(
        "--skip-enrichment",
        action="store_true",
        help="Skip enrichment stage (faster but less detail)",
    )
    parser.add_argument(
        "--max-qa-enrichment",
        type=int,
        default=None,
        help="Limit number of Q&A units to enrich (for testing)",
    )
    parser.add_argument(
        "--max-strategic-enrichment",
        type=int,
        default=None,
        help="Limit number of strategic statements to enrich",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Output compact JSON (no indentation)",
    )

    args = parser.parse_args()

    if not args.pdf_path.exists():
        print(f"Error: File not found: {args.pdf_path}")
        sys.exit(1)

    output_path = args.output
    if output_path is None:
        output_path = args.pdf_path.parent / (
            args.pdf_path.stem + "_v2_report.json"
        )


    print(f"Pipeline v2 - Processing: {args.pdf_path}")
    print(f"Output: {output_path}")
    if args.skip_enrichment:
        print("Enrichment: SKIPPED")
    print()

    try:
        state = run_pipeline_v2(
            pdf_path=args.pdf_path,
            skip_enrichment=args.skip_enrichment,
            max_qa_enrichment=args.max_qa_enrichment,
            max_strategic_enrichment=args.max_strategic_enrichment,
        )

        # Convert to report format
        report = state_to_report(state)

        with open(output_path, "w", encoding="utf-8") as f:
            if args.compact:
                json.dump(report, f, ensure_ascii=False, default=str)
            else:
                json.dump(report, f, indent=2, ensure_ascii=False, default=str)

        print(f"\nReport saved to: {output_path}")

        # Print summary
        print(f"\n{'='*50}")
        print("SUMMARY")
        print(f"{'='*50}")

        if state.metadata:
            print(f"Company: {state.metadata.company_name or 'Unknown'}")
            print(f"Quarter: {state.metadata.fiscal_quarter or '?'} {state.metadata.fiscal_year or ''}")

        if state.boundary_result:
            print(f"\nSections Detected: {state.boundary_result.total_sections}")
            print(f"  - Q&A Sections: {state.boundary_result.qa_section_count}")
            print(f"  - Coverage: {state.boundary_result.coverage_percent:.1f}%")

        if state.speaker_registry:
            print(f"\nSpeakers: {state.speaker_registry.total_speakers}")
            print(f"  - Management: {state.speaker_registry.management_count}")
            print(f"  - Analysts: {state.speaker_registry.analyst_count}")

        qa_count = len(state.enriched_qa_units) if state.enriched_qa_units else (
            state.qa_extraction_result.total_qa_units if state.qa_extraction_result else 0
        )
        stmt_count = len(state.enriched_strategic_statements) if state.enriched_strategic_statements else (
            state.strategic_extraction_result.total_statements if state.strategic_extraction_result else 0
        )

        print(f"\nQ&A Units: {qa_count}")
        print(f"Strategic Statements: {stmt_count}")
        print(f"LLM Calls: {state.llm_calls_made}")

        if state.processing_start and state.processing_end:
            duration = (state.processing_end - state.processing_start).total_seconds()
            print(f"Duration: {duration:.1f}s")

        if not state.validation_passed:
            print(f"\nValidation Issues:")
            for issue in state.validation_issues:
                print(f"  - {issue}")

        if state.errors:
            print(f"\nErrors: {len(state.errors)}")
            for err in state.errors:
                print(f"  - [{err.get('stage', '?')}] {err.get('error', '?')}")

    except PipelineV2Error as e:
        print(f"\nPipeline Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
