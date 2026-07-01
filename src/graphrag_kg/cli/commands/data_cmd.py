"""CLI commands for test data generation.

Usage:
    graphrag-kg data generate [--scenario SCENARIO] [--seed SEED]
                              [--entity-count N] [--doc-count N]
                              [--output DIR] [--formats FORMATS]
    graphrag-kg data ground-truth [--scenario SCENARIO] [--output DIR]
    graphrag-kg data list-scenarios
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from graphrag_kg.cli.utils import (
    console,
    print_error,
    print_header,
    print_info,
    print_json_syntax,
    print_success,
    print_table,
)
from graphrag_kg.data.generator import TestDataGenerator

data_app = typer.Typer(help="Test data generation and ground truth management")


@data_app.command("generate")
def generate(
    scenario: str = typer.Option(
        "pharma_supply_chain",
        "--scenario", "-s",
        help="Scenario to generate",
    ),
    seed: int = typer.Option(42, "--seed", help="Random seed for reproducibility"),
    entity_count: int = typer.Option(
        0, "--entity-count", "-e",
        help="Override entity count (0 = use scenario defaults)",
    ),
    doc_count: int = typer.Option(
        0, "--doc-count", "-d",
        help="Override document count (0 = use scenario defaults)",
    ),
    output: Path = typer.Option(
        Path("tests/fixtures/generated"),
        "--output", "-o",
        help="Output directory for generated data",
    ),
    formats: str = typer.Option(
        "md,txt,html",
        "--formats", "-f",
        help="Output formats: md,txt,html,pdf (comma-separated)",
    ),
) -> None:
    """Generate a test data scenario with ground truth."""
    print_header(f"Generating Test Data: {scenario}")

    output_dir = output / scenario
    format_list = [f.strip() for f in formats.split(",")]

    try:
        generator = TestDataGenerator(
            scenario=scenario,
            seed=seed,
            entity_count=entity_count,
            doc_count=doc_count,
            output_formats=format_list,
        )

        print_info(f"Scenario: {scenario}")
        print_info(f"Seed: {seed}")
        print_info(f"Output: {output_dir}")
        print_info(f"Formats: {', '.join(format_list)}")

        ground_truth = generator.generate(output_dir)

        # Print summary
        print_success(f"Generated {len(ground_truth.document_files)} document files")
        print_success(f"Entities: {len(ground_truth.entities)}")
        print_success(f"Relationships: {len(ground_truth.relationships)}")
        print_success(f"Communities: {len(ground_truth.communities)}")
        print_success(f"Test Queries: {len(ground_truth.test_queries)}")

        print_header("Entity Types")
        rows = [[etype, str(count)] for etype, count in
                sorted(ground_truth.entity_type_counts.items())]
        print_table("Entity Type Distribution", ["Type", "Count"], rows)

        print_header("Generated Files")
        for f in sorted(ground_truth.document_files):
            console.print(f"  - {f}")
        console.print(f"  - ground_truth.json")
        console.print(f"  - queries.json")
        console.print(f"  - README.md")

        print_header("Test Queries Preview")
        for i, q in enumerate(ground_truth.test_queries[:3], 1):
            console.print(f"  [bold]{i}.[/bold] {q.question}")
            if q.hops_description:
                console.print(f"     [dim]Hops: {q.hops_description}[/dim]")

        if len(ground_truth.test_queries) > 3:
            console.print(f"  [dim]... and {len(ground_truth.test_queries) - 3} more[/dim]")

    except Exception as e:
        print_error(f"Failed to generate test data: {e}")
        raise typer.Exit(code=1)


@data_app.command("ground-truth")
def ground_truth(
    scenario: str = typer.Option(
        "pharma_supply_chain",
        "--scenario", "-s",
        help="Scenario to view",
    ),
    output: Path = typer.Option(
        Path("tests/fixtures/generated"),
        "--output", "-o",
        help="Directory containing generated data",
    ),
) -> None:
    """View the ground truth for a generated scenario."""
    gt_path = output / scenario / "ground_truth.json"

    if not gt_path.exists():
        print_error(
            f"Ground truth not found at {gt_path}. "
            f"Run 'graphrag-kg data generate --scenario {scenario}' first."
        )
        raise typer.Exit(code=1)

    with open(gt_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print_header(f"Ground Truth: {scenario}")
    print_info(f"Seed: {data['seed']}")
    print_info(f"Documents: {data['document_count']}")
    print_info(f"Entities: {len(data['expected_entities'])}")
    print_info(f"Relationships: {len(data['expected_relationships'])}")
    print_info(f"Communities: {len(data['expected_communities'])}")
    print_info(f"Test Queries: {len(data['test_queries'])}")

    # Show entities
    print_header("Expected Entities (sample)")
    for e in data["expected_entities"][:5]:
        console.print(f"  [bold]{e['name']}[/bold] ({e['type']})")
        if e.get("description_contains"):
            console.print(f"    [dim]Description: {', '.join(e['description_contains'])}[/dim]")
        if e.get("mentioned_in_docs"):
            console.print(f"    [dim]In docs: {', '.join(e['mentioned_in_docs'])}[/dim]")

    if len(data["expected_entities"]) > 5:
        console.print(f"  [dim]... and {len(data['expected_entities']) - 5} more entities[/dim]")

    # Show test queries
    print_header("Test Queries")
    for i, q in enumerate(data["test_queries"], 1):
        console.print(f"  [bold]{i}.[/bold] {q['question']}")
        if q.get("hops_description"):
            console.print(f"     [dim]→ {q['hops_description']}[/dim]")
        if q.get("expected_relationship_path"):
            console.print(f"     [dim]→ Path: {q['expected_relationship_path']}[/dim]")
        if q.get("expected_entities_in_response"):
            console.print(
                f"     [dim]→ Expected entities: "
                f"{', '.join(q['expected_entities_in_response'])}[/dim]"
            )
        console.print()

    # Show full JSON option
    print_info(f"Full JSON at: {gt_path}")


@data_app.command("list-scenarios")
def list_scenarios() -> None:
    """List all available test data scenarios."""
    scenarios = TestDataGenerator.list_scenarios()

    print_header("Available Test Data Scenarios")

    descriptions = {
        "pharma_supply_chain": "Pharmaceutical supply chain with strong multi-hop relationships",
        "tech_company": "Technology company ecosystem with org/person/tech relationships",
    }

    rows = []
    for s in scenarios:
        desc = descriptions.get(s, "No description available")
        rows.append([s, desc])

    print_table("Scenarios", ["Name", "Description"], rows)

    console.print()
    console.print("Use [bold]graphrag-kg data generate --scenario <name>[/bold] to generate.")
