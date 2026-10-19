"""Build an RDF ontology and evaluate ontology-truth-based link predictions."""

from __future__ import annotations

import argparse
import csv
import html
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import quote

import networkx as nx
from rdflib import Graph, Literal, Namespace, RDF, RDFS, OWL, URIRef
if __package__:
    from .kg_viewer import write_full_graph_viewer
    from .prediction_viewer import write_prediction_explorer
else:
    from kg_viewer import write_full_graph_viewer
    from prediction_viewer import write_prediction_explorer

Triple = tuple[str, str, str]
RuleBody = tuple[str, str]
BASE_URI = "http://example.org/fb15k237/"


@dataclass(frozen=True)
class PathRule:
    first_relation: str
    second_relation: str
    predicted_relation: str
    support: int
    confidence: float


def read_triples(path: Path) -> list[Triple]:
    """Read strict tab-separated head, relation, tail records."""
    triples: list[Triple] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            line = raw_line.rstrip("\r\n")
            if not line:
                continue
            fields = line.split("\t")
            if len(fields) != 3 or any(not field for field in fields):
                raise ValueError(
                    f"{path}:{line_number}: expected three non-empty "
                    "tab-separated fields"
                )
            triples.append((fields[0], fields[1], fields[2]))
    if not triples:
        raise ValueError(f"No triples found in {path}")
    return triples


def build_knowledge_graph(triples: Iterable[Triple]) -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph()
    for index, (head, relation, tail) in enumerate(triples):
        graph.add_edge(head, tail, key=str(index), relation=relation)
    return graph


def _entity_uri(entity: str) -> URIRef:
    return URIRef(f"{BASE_URI}entity/{quote(entity, safe='')}")


def _relation_uri(relation: str) -> URIRef:
    return URIRef(f"{BASE_URI}relation/{quote(relation, safe='')}")


def _role_class_uri(role: str, relation: str) -> URIRef:
    return URIRef(
        f"{BASE_URI}class/{role}/{quote(relation, safe='')}"
    )


def build_ontology(triples: Iterable[Triple]) -> Graph:
    """Create relation-specific head/tail classes and assert all KG truths."""
    ontology = Graph()
    ontology.bind("kg", Namespace(BASE_URI))
    relation_roles: dict[str, tuple[URIRef, URIRef]] = {}

    for head, relation, tail in triples:
        if relation not in relation_roles:
            head_class = _role_class_uri("head", relation)
            tail_class = _role_class_uri("tail", relation)
            property_uri = _relation_uri(relation)
            relation_roles[relation] = (head_class, tail_class)
            ontology.add((head_class, RDF.type, OWL.Class))
            ontology.add((tail_class, RDF.type, OWL.Class))
            ontology.add((property_uri, RDF.type, OWL.ObjectProperty))
            ontology.add((property_uri, RDFS.domain, head_class))
            ontology.add((property_uri, RDFS.range, tail_class))
            ontology.add((property_uri, RDFS.label, Literal(relation)))

        head_class, tail_class = relation_roles[relation]
        head_uri = _entity_uri(head)
        tail_uri = _entity_uri(tail)
        ontology.add((head_uri, RDF.type, head_class))
        ontology.add((tail_uri, RDF.type, tail_class))
        ontology.add((head_uri, _relation_uri(relation), tail_uri))

    return ontology


def _path_counts(
    triples: Iterable[Triple],
    max_paths_per_pivot: int,
) -> tuple[Counter[RuleBody], Counter[tuple[RuleBody, str]], int, int]:
    outgoing: dict[str, list[tuple[str, str]]] = defaultdict(list)
    incoming: dict[str, list[tuple[str, str]]] = defaultdict(list)
    relations_by_pair: dict[tuple[str, str], set[str]] = defaultdict(set)
    for head, relation, tail in triples:
        outgoing[head].append((relation, tail))
        incoming[tail].append((relation, head))
        relations_by_pair[(head, tail)].add(relation)

    body_support: Counter[RuleBody] = Counter()
    rule_support: Counter[tuple[RuleBody, str]] = Counter()
    skipped_pivots = 0
    skipped_paths = 0

    for pivot, incoming_edges in incoming.items():
        outgoing_edges = outgoing.get(pivot, ())
        possible_paths = len(incoming_edges) * len(outgoing_edges)
        if possible_paths > max_paths_per_pivot:
            skipped_pivots += 1
            skipped_paths += possible_paths
            continue

        for first_relation, head in incoming_edges:
            for second_relation, tail in outgoing_edges:
                body = (first_relation, second_relation)
                body_support[body] += 1
                for predicted_relation in relations_by_pair.get((head, tail), ()):
                    rule_support[(body, predicted_relation)] += 1

    return body_support, rule_support, skipped_pivots, skipped_paths


def learn_path_rules(
    triples: Iterable[Triple],
    *,
    min_support: int = 2,
    min_confidence: float = 0.1,
    max_rules: int = 3_000,
    max_paths_per_pivot: int = 10_000,
) -> tuple[list[PathRule], dict[str, int]]:
    """Mine two-hop Horn rules and bound work at high-degree pivot entities."""
    if min_support < 1 or max_rules < 1 or max_paths_per_pivot < 1:
        raise ValueError("min_support, max_rules, and max_paths_per_pivot must be positive")
    if not 0 <= min_confidence <= 1:
        raise ValueError("min_confidence must be between 0 and 1")

    body_support, rule_support, skipped_pivots, skipped_paths = _path_counts(
        triples, max_paths_per_pivot
    )
    rules = [
        PathRule(
            first_relation=body[0],
            second_relation=body[1],
            predicted_relation=target,
            support=support,
            confidence=support / body_support[body],
        )
        for (body, target), support in rule_support.items()
        if support >= min_support
        and support / body_support[body] >= min_confidence
    ]
    rules.sort(
        key=lambda rule: (
            -rule.confidence,
            -rule.support,
            rule.first_relation,
            rule.second_relation,
            rule.predicted_relation,
        )
    )
    stats = {
        "two_hop_body_patterns": len(body_support),
        "rules_before_limit": len(rules),
        "skipped_high_degree_pivots": skipped_pivots,
        "skipped_two_hop_paths": skipped_paths,
    }
    return rules[:max_rules], stats


def predict_missing_edges(
    triples: Iterable[Triple],
    rules: Iterable[PathRule],
    *,
    max_paths_per_pivot: int = 10_000,
) -> tuple[dict[Triple, tuple[float, int]], dict[str, int]]:
    """Apply learned rules and return unique triples absent from the KG."""
    triples = list(triples)
    known = set(triples)
    outgoing: dict[str, list[tuple[str, str]]] = defaultdict(list)
    incoming: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for head, relation, tail in triples:
        outgoing[head].append((relation, tail))
        incoming[tail].append((relation, head))

    rules_by_body: dict[RuleBody, list[PathRule]] = defaultdict(list)
    for rule in rules:
        rules_by_body[(rule.first_relation, rule.second_relation)].append(rule)

    predictions: dict[Triple, tuple[float, int]] = {}
    skipped_pivots = 0
    skipped_paths = 0
    for pivot, incoming_edges in incoming.items():
        outgoing_edges = outgoing.get(pivot, ())
        possible_paths = len(incoming_edges) * len(outgoing_edges)
        if possible_paths > max_paths_per_pivot:
            skipped_pivots += 1
            skipped_paths += possible_paths
            continue

        for first_relation, head in incoming_edges:
            for second_relation, tail in outgoing_edges:
                for rule in rules_by_body.get((first_relation, second_relation), ()):
                    candidate = (head, rule.predicted_relation, tail)
                    if candidate in known:
                        continue
                    current = predictions.get(candidate, (0.0, 0))
                    predictions[candidate] = (
                        max(current[0], rule.confidence),
                        max(current[1], rule.support),
                    )

    return predictions, {
        "skipped_high_degree_pivots": skipped_pivots,
        "skipped_two_hop_paths": skipped_paths,
    }


def _write_triples(path: Path, triples: Iterable[Triple]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerows(triples)


def _write_rules(path: Path, rules: Iterable[PathRule]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(
            ["body_relation_1", "body_relation_2", "predicted_relation",
             "support", "confidence"]
        )
        for rule in rules:
            writer.writerow(
                [
                    rule.first_relation,
                    rule.second_relation,
                    rule.predicted_relation,
                    rule.support,
                    f"{rule.confidence:.8f}",
                ]
            )


def _evaluate_predictions(
    predictions: dict[Triple, tuple[float, int]],
    known: set[Triple],
    validation: set[Triple],
    test: set[Triple],
) -> dict[str, object]:
    predicted = set(predictions)

    def split_metrics(expected: set[Triple]) -> dict[str, float | int]:
        hits = len(predicted & expected)
        return {
            "known_edges": len(expected),
            "predicted_edges_found": hits,
            "precision": hits / len(predicted) if predicted else 0.0,
            "recall": hits / len(expected) if expected else 0.0,
        }

    return {
        "training_edges": len(known),
        "validation_edges": len(validation),
        "test_edges": len(test),
        "predicted_unique_edges": len(predicted),
        "predicted_edges_not_in_training": len(predicted - known),
        "validation": split_metrics(validation),
        "test": split_metrics(test),
        "note": (
            "Validation/test matches are observed held-out positives. "
            "Non-matches are unknown, not confirmed false facts."
        ),
    }


def write_prediction_graph(
    path: Path,
    triples: Iterable[Triple],
    predictions: dict[Triple, tuple[float, int]],
    validation: Iterable[Triple],
    test: Iterable[Triple],
    rules: Iterable[PathRule],
    preferred_prediction: Triple | None = None,
) -> None:
    """Render a verified predicted edge and its two training-edge supports."""
    train = set(triples)
    validation_set = set(validation)
    test_set = set(test)
    outgoing: dict[tuple[str, str], list[str]] = defaultdict(list)
    for head, relation, tail in train:
        outgoing[(head, relation)].append(tail)

    rules_by_target: dict[str, list[PathRule]] = defaultdict(list)
    for rule in rules:
        rules_by_target[rule.predicted_relation].append(rule)

    examples = [
        (triple, status, predictions[triple])
        for status, known in (("TEST", test_set), ("VALIDATION", validation_set))
        for triple in predictions.keys() & known
    ]
    examples.sort(
        key=lambda item: (
            sum(map(len, item[0])),
            -item[2][0],
            item[0],
        )
    )
    if preferred_prediction is not None:
        examples.sort(key=lambda item: item[0] != preferred_prediction)

    selected: tuple[Triple, str, tuple[float, int], str, str, str] | None = None
    for (head, relation, tail), status, score in examples:
        for rule in rules_by_target.get(relation, ()):
            for middle in outgoing.get((head, rule.first_relation), ()):
                if tail in outgoing.get((middle, rule.second_relation), ()):
                    selected = (
                        (head, relation, tail),
                        status,
                        score,
                        rule.first_relation,
                        middle,
                        rule.second_relation,
                    )
                    break
            if selected is not None:
                break
        if selected is not None:
            break

    if selected is None:
        raise ValueError(
            "No predicted edge matched validation/test with a training "
            "two-hop explanation; cannot create a verified example graph."
        )

    (head, relation, tail), status, (confidence, support), first, middle, second = selected

    def text(value: str) -> str:
        return html.escape(value, quote=True)

    def wrapped_lines(value: str, width: int = 58) -> list[str]:
        words = value.split("/")
        lines: list[str] = []
        current = ""
        for word in words:
            part = f"/{word}" if current or lines else word
            if current and len(current) + len(part) > width:
                lines.append(current)
                current = part
            else:
                current += part
        if current:
            lines.append(current)
        return lines or [""]

    def relation_label(identifier: str, value: str, x: int, y: int) -> str:
        lines = wrapped_lines(value, width=50)
        start_y = y - (len(lines) - 1) * 7
        relation_text = "".join(
            f'<tspan x="{x}" dy="{0 if index == 0 else 15}">'
            f'{text(line)}</tspan>'
            for index, line in enumerate(lines)
        )
        box_height = max(38, len(lines) * 15 + 12)
        return (
            f'<rect x="{x - 225}" y="{y - 21}" width="450" '
            f'height="{box_height}" rx="8" fill="#ffffff" stroke="#cbd5e1"/>'
            f'<text x="{x}" y="{start_y}" text-anchor="middle" '
            'font-family="Arial,sans-serif" font-size="11" fill="#334155">'
            f'<tspan font-weight="700">{identifier}: </tspan>{relation_text}</text>'
        )

    first_fact = f"{head} --{first}--> {middle}"
    second_fact = f"{middle} --{second}--> {tail}"
    inferred_fact = f"{head} --{relation}--> {tail}"

    def fact_text(value: str, y: int, color: str) -> str:
        lines = wrapped_lines(value, width=110)
        spans = "".join(
            f'<tspan x="700" dy="{0 if index == 0 else 19}">'
            f'{text(line)}</tspan>'
            for index, line in enumerate(lines)
        )
        return (
            f'<text x="700" y="{y}" text-anchor="middle" '
            f'font-family="Arial,sans-serif" font-size="13" fill="{color}">'
            f'{spans}</text>'
        )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="100%" height="auto" viewBox="0 0 1400 980" preserveAspectRatio="xMidYMid meet">
<defs>
  <marker id="arrow-observed" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto">
    <path d="M0,0 L10,5 L0,10 z" fill="#16803c"/>
  </marker>
  <marker id="arrow-predicted" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto">
    <path d="M0,0 L10,5 L0,10 z" fill="#d97706"/>
  </marker>
</defs>
<rect width="1400" height="980" fill="#f8fafc"/>
<text x="700" y="48" text-anchor="middle" font-family="Arial,sans-serif" font-size="27" font-weight="700" fill="#0f172a">Predicted edge and its supporting facts</text>
<text x="700" y="80" text-anchor="middle" font-family="Arial,sans-serif" font-size="16" fill="#475569">Apply the two-hop rule to the two observed green edges.</text>
<path d="M245,400 Q700,130 1155,400" fill="none" stroke="#d97706" stroke-width="5" stroke-dasharray="13 9" marker-end="url(#arrow-predicted)"/>
<rect x="490" y="126" width="420" height="68" rx="12" fill="#fff7ed" stroke="#d97706" stroke-width="2"/>
<text x="700" y="153" text-anchor="middle" font-family="Arial,sans-serif" font-size="15" font-weight="700" fill="#9a3412">PREDICTED EDGE — MATCHES {status} DATA</text>
<text x="700" y="178" text-anchor="middle" font-family="Arial,sans-serif" font-size="14" fill="#7c2d12">confidence {confidence:.3f} · rule support {support}</text>
<line x1="270" y1="460" x2="620" y2="460" stroke="#16803c" stroke-width="5" marker-end="url(#arrow-observed)"/>
<line x1="780" y1="460" x2="1130" y2="460" stroke="#16803c" stroke-width="5" marker-end="url(#arrow-observed)"/>
{relation_label("r1", first, 445, 425)}
{relation_label("r2", second, 955, 425)}
<circle cx="190" cy="460" r="78" fill="#dbeafe" stroke="#2563eb" stroke-width="3"/>
<circle cx="700" cy="460" r="78" fill="#e2e8f0" stroke="#64748b" stroke-width="3"/>
<circle cx="1210" cy="460" r="78" fill="#dcfce7" stroke="#16a34a" stroke-width="3"/>
<text x="190" y="451" text-anchor="middle" font-family="Arial,sans-serif" font-size="16" font-weight="700" fill="#0f172a">HEAD</text>
<text x="190" y="477" text-anchor="middle" font-family="Arial,sans-serif" font-size="14" fill="#334155">{text(head)}</text>
<text x="700" y="451" text-anchor="middle" font-family="Arial,sans-serif" font-size="16" font-weight="700" fill="#0f172a">MIDDLE</text>
<text x="700" y="477" text-anchor="middle" font-family="Arial,sans-serif" font-size="14" fill="#334155">{text(middle)}</text>
<text x="1210" y="451" text-anchor="middle" font-family="Arial,sans-serif" font-size="16" font-weight="700" fill="#0f172a">TAIL</text>
<text x="1210" y="477" text-anchor="middle" font-family="Arial,sans-serif" font-size="14" fill="#334155">{text(tail)}</text>
<text x="700" y="585" text-anchor="middle" font-family="Arial,sans-serif" font-size="15" font-weight="700" fill="#166534">Observed supporting training facts</text>
{fact_text("1. " + first_fact, 617, "#166534")}
{fact_text("2. " + second_fact, 660, "#166534")}
<text x="700" y="724" text-anchor="middle" font-family="Arial,sans-serif" font-size="15" font-weight="700" fill="#9a3412">Rule applied</text>
{fact_text("r1(X,Y) AND r2(Y,Z) => r3(X,Z)", 754, "#9a3412")}
{fact_text("r1 = " + first, 786, "#475569")}
{fact_text("r2 = " + second, 818, "#475569")}
{fact_text("r3 = " + relation, 850, "#475569")}
<text x="700" y="906" text-anchor="middle" font-family="Arial,sans-serif" font-size="14" fill="#9a3412">Dashed orange edge matches a held-out {status} triple; solid green edges are training facts.</text>
</svg>
"""
    path.write_text(svg, encoding="utf-8")


def run_pipeline(
    root: Path,
    *,
    min_support: int = 2,
    min_confidence: float = 0.1,
    max_rules: int = 3_000,
    max_paths_per_pivot: int = 10_000,
) -> dict[str, object]:
    root = root.resolve()
    dataset_dir = root / "dataset"
    results_dir = root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    train = read_triples(dataset_dir / "train.txt")
    validation = read_triples(dataset_dir / "valid.txt")
    test = read_triples(dataset_dir / "test.txt")
    train_set = set(train)

    graph = build_knowledge_graph(train)
    graph_path = results_dir / "knowledge_graph.graphml"
    nx.write_graphml(graph, graph_path)
    viewer_path = results_dir / "full_kg_viewer.html"
    write_full_graph_viewer(train, viewer_path)

    ontology = build_ontology(train)
    ontology_path = results_dir / "ontology.ttl"
    ontology.serialize(destination=str(ontology_path), format="turtle")
    truths_path = results_dir / "truths.tsv"
    _write_triples(truths_path, train)

    rules, mining_stats = learn_path_rules(
        train,
        min_support=min_support,
        min_confidence=min_confidence,
        max_rules=max_rules,
        max_paths_per_pivot=max_paths_per_pivot,
    )
    _write_rules(results_dir / "learned_rules.tsv", rules)
    predictions, prediction_stats = predict_missing_edges(
        train, rules, max_paths_per_pivot=max_paths_per_pivot
    )
    prediction_viewer_path = results_dir / "prediction_explorer.html"
    write_prediction_explorer(
        train, predictions, rules, validation, test, prediction_viewer_path
    )
    predicted_path = results_dir / "predicted_edges.tsv"
    with predicted_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(["head", "relation", "tail", "confidence", "rule_support"])
        for (head, relation, tail), (confidence, support) in sorted(
            predictions.items(), key=lambda item: (-item[1][0], item[0])
        ):
            writer.writerow(
                [head, relation, tail, f"{confidence:.8f}", support]
            )

    metrics = _evaluate_predictions(
        predictions, train_set, set(validation), set(test)
    )
    example_graph_path = results_dir / "prediction_example.svg"
    write_prediction_graph(
        example_graph_path, train, predictions, validation, test, rules
    )
    metrics["graph_nodes"] = graph.number_of_nodes()
    metrics["graph_edges"] = graph.number_of_edges()
    metrics["ontology_triples"] = len(ontology)
    metrics["learned_rules"] = len(rules)
    metrics["prediction_example_graph"] = str(example_graph_path)
    metrics["full_graph_viewer"] = str(viewer_path)
    metrics["prediction_explorer"] = str(prediction_viewer_path)
    metrics["mining"] = mining_stats
    metrics["prediction"] = prediction_stats
    metrics["parameters"] = {
        "min_support": min_support,
        "min_confidence": min_confidence,
        "max_rules": max_rules,
        "max_paths_per_pivot": max_paths_per_pivot,
    }
    metrics_path = results_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a knowledge graph/ontology and predict missing edges."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Project root containing dataset/train.txt, valid.txt, and test.txt.",
    )
    parser.add_argument("--min-support", type=int, default=2)
    parser.add_argument("--min-confidence", type=float, default=0.1)
    parser.add_argument("--max-rules", type=int, default=3_000)
    parser.add_argument("--max-paths-per-pivot", type=int, default=10_000)
    args = parser.parse_args()
    metrics = run_pipeline(
        args.root,
        min_support=args.min_support,
        min_confidence=args.min_confidence,
        max_rules=args.max_rules,
        max_paths_per_pivot=args.max_paths_per_pivot,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
