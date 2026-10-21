"""Create an HTML dashboard from existing KG prediction outputs."""

from __future__ import annotations

import argparse
import csv
import html
import json
from collections import Counter
from pathlib import Path

if __package__:
    from .kg_pipeline import _evaluate_predictions, read_triples
else:
    from kg_pipeline import _evaluate_predictions, read_triples

Triple = tuple[str, str, str]


def read_predictions(path: Path) -> dict[Triple, tuple[float, int]]:
    predictions: dict[Triple, tuple[float, int]] = {}
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        expected = {"head", "relation", "tail", "confidence", "rule_support"}
        if reader.fieldnames is None or not expected.issubset(reader.fieldnames):
            raise ValueError(f"{path}: missing required predicted-edge columns")
        for line_number, row in enumerate(reader, start=2):
            if any(row.get(column) is None for column in expected):
                raise ValueError(f"{path}:{line_number}: incomplete prediction row")
            triple = (row["head"], row["relation"], row["tail"])
            if not all(triple):
                raise ValueError(f"{path}:{line_number}: empty triple field")
            if triple in predictions:
                raise ValueError(f"{path}:{line_number}: duplicate predicted edge")
            predictions[triple] = (
                float(row["confidence"]),
                int(row["rule_support"]),
            )
    return predictions


def _check_metrics(actual: dict[str, object], saved: dict[str, object]) -> None:
    for key in (
        "training_edges",
        "validation_edges",
        "test_edges",
        "predicted_unique_edges",
        "predicted_edges_not_in_training",
    ):
        if actual[key] != saved.get(key):
            raise ValueError(
                f"Calculated metric {key}={actual[key]} does not match "
                f"results/metrics.json value {saved.get(key)}"
            )

    for split in ("validation", "test"):
        calculated = actual[split]
        recorded = saved.get(split)
        if not isinstance(recorded, dict):
            raise ValueError(f"Missing {split} metrics in results/metrics.json")
        for key in ("known_edges", "predicted_edges_found"):
            if calculated[key] != recorded.get(key):
                raise ValueError(
                    f"Calculated {split}.{key}={calculated[key]} does not "
                    f"match results/metrics.json value {recorded.get(key)}"
                )
        for key in ("precision", "recall"):
            if abs(float(calculated[key]) - float(recorded.get(key, -1))) > 1e-12:
                raise ValueError(
                    f"Calculated {split}.{key} does not match "
                    "results/metrics.json"
                )


def build_dashboard(root: Path) -> dict[str, object]:
    root = root.resolve()
    dataset_dir = root / "dataset"
    results_dir = root / "results"
    train = read_triples(dataset_dir / "train.txt")
    validation = read_triples(dataset_dir / "valid.txt")
    test_rows = read_triples(dataset_dir / "test.txt")
    predictions = read_predictions(results_dir / "predicted_edges.tsv")

    train_set, validation_set, test_set = set(train), set(validation), set(test_rows)
    all_rows = [*train, *validation, *test_rows]
    if len(set(all_rows)) != len(all_rows):
        raise ValueError("Dataset splits contain duplicate triples")
    entity_count = len(
        {entity for head, _, tail in all_rows for entity in (head, tail)}
    )
    relation_count = len({relation for _, relation, _ in all_rows})
    split_overlaps = {
        "training_validation": len(train_set & validation_set),
        "training_test": len(train_set & test_set),
        "validation_test": len(validation_set & test_set),
    }
    if any(split_overlaps.values()):
        raise ValueError(
            "The dataset splits overlap; candidate categories would not be "
            "mutually exclusive. See split-overlap counts: "
            f"{split_overlaps}"
        )

    predicted_set = set(predictions)
    candidate_categories = {
        "training": predicted_set & train_set,
        "validation": predicted_set & validation_set,
        "test": predicted_set & test_set,
        "none": predicted_set - (train_set | validation_set | test_set),
    }
    category_overlap_counts = {
        "training_validation": len(
            candidate_categories["training"] & candidate_categories["validation"]
        ),
        "training_test": len(
            candidate_categories["training"] & candidate_categories["test"]
        ),
        "training_none": len(
            candidate_categories["training"] & candidate_categories["none"]
        ),
        "validation_test": len(
            candidate_categories["validation"] & candidate_categories["test"]
        ),
        "validation_none": len(
            candidate_categories["validation"] & candidate_categories["none"]
        ),
        "test_none": len(
            candidate_categories["test"] & candidate_categories["none"]
        ),
    }
    metrics = _evaluate_predictions(
        predictions, train_set, validation_set, test_set
    )
    saved_metrics_path = results_dir / "metrics.json"
    saved_metrics = json.loads(saved_metrics_path.read_text(encoding="utf-8"))
    _check_metrics(metrics, saved_metrics)

    recovered = [triple for triple in test_rows if triple in predicted_set]
    not_recovered = [triple for triple in test_rows if triple not in predicted_set]
    if len(recovered) != metrics["test"]["predicted_edges_found"]:
        raise ValueError("Recovered test triples do not match evaluation metrics")
    if len(recovered) + len(not_recovered) != len(test_rows):
        raise ValueError("Test recovered and missed rows do not cover the test set")

    test_by_relation: Counter[str] = Counter(
        relation for _, relation, _ in test_rows
    )
    recovered_by_relation: Counter[str] = Counter(
        relation for _, relation, _ in recovered
    )
    predicted_by_relation: Counter[str] = Counter(
        relation for _, relation, _ in predictions
    )
    relationship_metrics = []
    for relation, test_count in sorted(
        test_by_relation.items(), key=lambda item: (-item[1], item[0])
    ):
        hits = recovered_by_relation[relation]
        candidate_count = predicted_by_relation[relation]
        precision = hits / candidate_count if candidate_count else 0.0
        recall = hits / test_count if test_count else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        relationship_metrics.append(
            {
                "relation": relation,
                "test_triplets": test_count,
                "candidate_predictions": candidate_count,
                "recovered_test_triplets": hits,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

    precision = float(metrics["test"]["precision"])
    recall = float(metrics["test"]["recall"])
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    if len(candidate_categories["training"]) != 0:
        raise ValueError(
            "Prediction candidates overlap training data; metrics cannot be "
            "reported as missing-edge precision."
        )
    if sum(len(group) for group in candidate_categories.values()) != len(
        predictions
    ):
        raise ValueError("Candidate split categories do not partition predictions")

    dashboard_data = {
        "statistics": {
            "entities": entity_count,
            "distinct_relationship_types": relation_count,
            "training_triplets": len(train),
            "validation_triplets": len(validation),
            "test_triplets": len(test_rows),
            "total_original_triplets": len(all_rows),
            "unique_candidate_edges": len(predictions),
        },
        "candidate_categories": {
            name: len(edges) for name, edges in candidate_categories.items()
        },
        "candidate_category_overlaps": category_overlap_counts,
        "split_overlaps": split_overlaps,
        "evaluation": {
            "test_triplets_correctly_predicted": len(recovered),
            "test_triplets_not_predicted": len(not_recovered),
            "precision": metrics["test"]["precision"],
            "recall": metrics["test"]["recall"],
            "f1": f1,
        },
        "relationship_metrics": relationship_metrics,
        "recovered_test": recovered,
        "missed_test": not_recovered,
        "candidate_explorer": "prediction_explorer.html",
        "explanation": (
            "A candidate is an edge generated by the rule system. A test "
            "match is an exact head/relation/tail triple present in test.txt. "
            "A candidate matching none of the supplied splits has unknown "
            "truth status, not a confirmed false edge."
        ),
    }

    html_document = _dashboard_html(dashboard_data)
    output_path = results_dir / "evaluation_dashboard.html"
    output_path.write_text(html_document, encoding="utf-8")
    return dashboard_data


def _dashboard_html(data: dict[str, object]) -> str:
    payload = json.dumps(data, ensure_ascii=True, separators=(",", ":"))
    template = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>FB15K-237 Prediction Evaluation</title>
<style>
:root{color-scheme:dark;--bg:#0b1220;--panel:#111c2e;--border:#26354d;--text:#e5edf8;--muted:#9aabc2;--green:#31c978;--orange:#f59e0b;--blue:#59b7ff}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.5 system-ui,"Segoe UI",sans-serif}
header{padding:22px max(18px,calc((100vw - 1280px)/2));background:var(--panel);border-bottom:1px solid var(--border)}
h1{margin:0 0 6px;font-size:24px}.muted{color:var(--muted)}main{max-width:1280px;margin:auto;padding:18px;display:grid;gap:16px}
section,.card{background:var(--panel);border:1px solid var(--border);border-radius:10px;padding:16px}
h2{font-size:17px;margin:0 0 12px}h3{font-size:14px;margin:12px 0 8px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:10px}
.card .label{color:var(--muted)}.card .number{font-size:25px;font-weight:700;margin-top:5px;font-variant-numeric:tabular-nums}
.metrics{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:10px}
.metric{padding:12px;background:#0c1728;border-radius:8px}.metric b{font-size:22px;display:block}
.bar-row{display:grid;grid-template-columns:minmax(190px,340px) 1fr 110px;gap:10px;align-items:center;margin:9px 0}
.bar-bg{height:18px;background:#1d2b40;border-radius:9px;overflow:hidden}.bar-fill{height:100%;border-radius:9px}
.train{background:#9aabc2}.validation{background:#59b7ff}.test{background:#31c978}.none{background:#f59e0b}.missed{background:#e76f7a}
.controls{display:flex;gap:9px;flex-wrap:wrap;margin:8px 0}input,select,button{font:inherit;color:var(--text);background:#0b1423;border:1px solid var(--border);border-radius:7px;padding:9px 11px}input{min-width:230px;flex:1}button,a.button{cursor:pointer;text-decoration:none;display:inline-block}button:hover,a.button:hover{border-color:var(--blue)}
.table-wrap{max-height:58vh;overflow:auto}table{width:100%;border-collapse:collapse;font-size:12px}th,td{padding:8px;text-align:left;border-bottom:1px solid var(--border);overflow-wrap:anywhere}th{position:sticky;top:0;background:#17243a}td{font-family:ui-monospace,Consolas,monospace}
.pager{display:flex;gap:10px;align-items:center;margin-top:10px;color:var(--muted)}.warning{border-left:4px solid var(--orange);padding:10px 12px;background:#261d11;color:#f6dbab}
.chart-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:16px}.donut-wrap{display:flex;align-items:center;justify-content:center;gap:18px;flex-wrap:wrap;min-height:220px}
.donut{width:180px;height:180px;border-radius:50%;display:grid;place-items:center;position:relative}.donut:after{content:"";position:absolute;inset:27px;background:var(--panel);border-radius:50%}.donut-label{z-index:1;text-align:center;font-weight:700}.legend{display:grid;gap:9px;color:var(--muted)}
.swatch{display:inline-block;width:11px;height:11px;border-radius:2px;margin-right:7px}.formula{font-family:ui-monospace,Consolas,monospace;color:#d5e5f7}
.relationship-chart{display:grid;gap:7px;max-height:520px;overflow:auto}.rel-row{display:grid;grid-template-columns:minmax(150px,390px) 1fr 65px;gap:10px;align-items:center;font-size:12px}.rel-name{overflow-wrap:anywhere;color:#c7d5e7}.rel-bar-bg{height:15px;background:#1d2b40;border-radius:8px;overflow:hidden}.rel-bar-fill{height:100%;background:var(--blue);border-radius:8px}
.small{font-size:12px}@media(max-width:650px){.bar-row{grid-template-columns:1fr 90px}.bar-bg{grid-column:1/-1;grid-row:2}}
</style>
</head>
<body>
<header><h1>FB15K-237 missing-edge evaluation</h1><div class="muted">Calculated from the existing train, validation, test, prediction, and metrics files. No model retraining performed.</div></header>
<main>
<section><h2>Overall dataset statistics</h2><div id="stats" class="cards"></div></section>
<div class="chart-grid">
<section><h2>Original split triplet counts</h2><div id="datasetBars"></div><p class="muted small">Original total = training + validation + test; duplicate triples and split overlaps were checked.</p></section>
<section><h2>Test triplets recovered</h2><div class="donut-wrap"><div id="testDonut" class="donut"><div id="donutLabel" class="donut-label"></div></div><div class="legend"><div><span class="swatch" style="background:#31c978"></span>Recovered: <b id="hitCount"></b></div><div><span class="swatch" style="background:#e76f7a"></span>Not recovered: <b id="missCount"></b></div></div></div><div id="testMetrics" class="metrics"></div></section>
</div>
<div class="chart-grid">
<section><h2>Precision, recall and F1-score</h2><div id="metricBars"></div><p class="muted small">Metrics use a test-positive, candidate-ranking-as-a-set evaluation consistent with the existing evaluation function.</p></section>
<section><h2>Candidate predictions vs. test size</h2><div id="candidateVsTest"></div><p class="muted small">These counts have different roles: candidate predictions are the precision denominator; known test triples are the recall denominator.</p></section>
</div>
<section><h2>Candidate edges by dataset match</h2><div id="candidateBars"></div><p id="overlaps" class="muted small"></p><p class="muted small">“None” means no exact triple match in any supplied split; it does not mean false.</p></section>
<section><h2>Relationship-wise test performance</h2><p class="muted small">For each relation: precision = recovered candidates / candidates for that relation; recall = recovered test triples / test triples for that relation; F1 is their harmonic mean. This is per-relation ranking-free set evaluation.</p><div class="controls"><label for="relationMetric">Metric:</label><select id="relationMetric"><option value="recall">Recall</option><option value="precision">Precision</option><option value="f1">F1-score</option></select><label for="relationLimit">Show:</label><select id="relationLimit"><option value="10">Top 10 test relations</option><option value="20" selected>Top 20 test relations</option><option value="50">Top 50 test relations</option><option value="all">All test relations</option></select></div><div id="relationshipChart" class="relationship-chart"></div></section>
<section><h2>Metric definitions and availability</h2><div class="table-wrap"><table><thead><tr><th>Metric</th><th>Calculated value</th><th>Explanation / availability</th></tr></thead><tbody id="metricDetails"></tbody></table></div><p class="warning">Accuracy and true-negative-based specificity are unavailable: the dataset does not enumerate true negatives, and unobserved edges must not be assumed false. Hits@1/3/10 and MRR are not reported because no ranking evaluator or ranked test candidates are present in the saved evaluation outputs.</p></section>
<section><h2>Recovered and unrecovered test triplets</h2>
<p class="muted">The tables contain the full corresponding test triples, not only a sample. Search head, relation, or tail; use pagination to browse.</p>
<div class="controls">
<select id="which"><option value="missed">Test triplets not predicted</option><option value="recovered">Correctly recovered test triplets</option></select>
<input id="search" placeholder="Search head, relation, or tail">
<button id="download">Download current table as TSV</button>
</div>
<div id="tableCount" class="muted small"></div><div class="table-wrap"><table><thead><tr><th>Head</th><th>Relation</th><th>Tail</th></tr></thead><tbody id="triples"></tbody></table></div>
<div class="pager"><button id="previous">Previous</button><span id="page"></span><button id="next">Next</button></div>
</section>
<section><h2>Evaluation method, leakage checks and interpretation</h2><p id="explanation"></p><p id="leakage"></p><div class="warning">An edge absent from train, validation, and test is unobserved in this dataset. Do not label it false solely because it has no split match. The test precision is a benchmark match rate over all generated candidates, not a proof that all nonmatching candidates are false.</div><p><a class="button" href="__CANDIDATE_EXPLORER__">Open the candidate-edge explorer with supporting rule paths</a></p></section>
</main>
<script>
const DATA=__DASHBOARD_DATA__;
const fmt=n=>Number(n).toLocaleString();
const stats=DATA.statistics, cats=DATA.candidate_categories, evaluation=DATA.evaluation;
document.querySelector("#stats").innerHTML=[
 ["Entities",stats.entities],["Distinct relationship types",stats.distinct_relationship_types],
 ["Training triplets",stats.training_triplets],["Validation triplets",stats.validation_triplets],
 ["Test triplets",stats.test_triplets],["Total original triplets",stats.total_original_triplets],
 ["Unique candidate edges",stats.unique_candidate_edges]
].map(([label,value])=>`<div class="card"><div class="label">${label}</div><div class="number">${fmt(value)}</div></div>`).join("");
const chartBar=(label,value,max,color)=>`<div class="bar-row"><span>${label}</span><div class="bar-bg"><div class="bar-fill ${color}" style="width:${max?100*value/max:0}%"></div></div><b>${fmt(value)}</b></div>`;
const maxOriginal=Math.max(stats.training_triplets,stats.validation_triplets,stats.test_triplets);
document.querySelector("#datasetBars").innerHTML=[
 chartBar("Training",stats.training_triplets,maxOriginal,"train"),
 chartBar("Validation",stats.validation_triplets,maxOriginal,"validation"),
 chartBar("Test",stats.test_triplets,maxOriginal,"test")
].join("");
document.querySelector("#testMetrics").innerHTML=[
 ["Test triplets correctly predicted",evaluation.test_triplets_correctly_predicted],
 ["Test triplets not predicted",evaluation.test_triplets_not_predicted],
 ["Precision",`${(evaluation.precision*100).toFixed(3)}%`],
 ["Recall",`${(evaluation.recall*100).toFixed(3)}%`],
 ["F1-score",`${(evaluation.f1*100).toFixed(3)}%`]
].map(([label,value])=>`<div class="metric"><span class="muted">${label}</span><b>${typeof value==="number"?fmt(value):value}</b></div>`).join("");
const testTotal=stats.test_triplets;
const recoveredShare=100*evaluation.test_triplets_correctly_predicted/testTotal;
document.querySelector("#testDonut").style.background=`conic-gradient(#31c978 0 ${recoveredShare}%,#e76f7a ${recoveredShare}% 100%)`;
document.querySelector("#donutLabel").innerHTML=`${recoveredShare.toFixed(1)}%<br><span class="muted small">recovered</span>`;
document.querySelector("#hitCount").textContent=fmt(evaluation.test_triplets_correctly_predicted);
document.querySelector("#missCount").textContent=fmt(evaluation.test_triplets_not_predicted);
const metricBar=(label,value,color)=>`<div class="bar-row"><span>${label}</span><div class="bar-bg"><div class="bar-fill ${color}" style="width:${100*value}%"></div></div><b>${(100*value).toFixed(3)}%</b></div>`;
document.querySelector("#metricBars").innerHTML=[
 metricBar("Precision",evaluation.precision,"validation"),
 metricBar("Recall",evaluation.recall,"test"),
 metricBar("F1-score",evaluation.f1,"none")
].join("");
const countMaximum=Math.max(stats.unique_candidate_edges,stats.test_triplets);
document.querySelector("#candidateVsTest").innerHTML=[
 chartBar("Unique candidate predictions",stats.unique_candidate_edges,countMaximum,"validation"),
 chartBar("Known test triplets",stats.test_triplets,countMaximum,"test")
].join("");
const candidateRows=[["Training",cats.training,"train"],["Validation",cats.validation,"validation"],["Test",cats.test,"test"],["None of these splits",cats.none,"none"]];
document.querySelector("#candidateBars").innerHTML=candidateRows.map(([label,count,cls])=>`<div class="bar-row"><span>${label}</span><div class="bar-bg"><div class="bar-fill ${cls}" style="width:${100*count/stats.unique_candidate_edges}%"></div></div><b>${fmt(count)}</b></div>`).join("");
document.querySelector("#overlaps").textContent=`Candidate-category pairwise overlaps: ${Object.entries(DATA.candidate_category_overlaps).map(([pair,count])=>`${pair.replaceAll("_"," + ")}: ${fmt(count)}`).join(" · ")}. Dataset split overlaps: ${Object.entries(DATA.split_overlaps).map(([pair,count])=>`${pair.replaceAll("_"," + ")}: ${fmt(count)}`).join(" · ")}.`;
const percent=n=>`${(100*n).toFixed(3)}%`;
const metricRows=[
 ["Precision",percent(evaluation.precision),`Existing evaluation definition: ${fmt(evaluation.test_triplets_correctly_predicted)} predicted edges found in test / ${fmt(stats.unique_candidate_edges)} unique candidate edges.`],
 ["Recall",percent(evaluation.recall),`${fmt(evaluation.test_triplets_correctly_predicted)} recovered test triples / ${fmt(stats.test_triplets)} total test triples.`],
 ["F1-score",percent(evaluation.f1),"Harmonic mean of precision and recall: 2 × P × R / (P + R)."],
 ["Hits@1 / Hits@3 / Hits@10","Unavailable","No ranked candidate list or Hits@K evaluation code is present."],
 ["Mean Reciprocal Rank (MRR)","Unavailable","No test-candidate ranking or reciprocal-rank evaluator is present."],
 ["Accuracy","Unavailable","Reliable true-negative labels are unavailable; unobserved triples are not necessarily false."]
];
document.querySelector("#metricDetails").innerHTML=metricRows.map(r=>`<tr><td>${r[0]}</td><td>${r[1]}</td><td>${r[2]}</td></tr>`).join("");
document.querySelector("#explanation").textContent=DATA.explanation;
document.querySelector("#leakage").textContent=`Leakage check: train/validation/test triple overlaps are all zero. The saved predictions match zero training triples, and the existing pipeline builds rules and candidates from train.txt; validation.txt and test.txt are only compared after prediction generation.`;
const htmlEscape=s=>String(s).replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
function renderRelationshipChart(){
 const metric=document.querySelector("#relationMetric").value,limit=document.querySelector("#relationLimit").value;
 const relations=limit==="all"?DATA.relationship_metrics:DATA.relationship_metrics.slice(0,Number(limit));
 document.querySelector("#relationshipChart").innerHTML=relations.map(r=>`<div class="rel-row" title="Test ${fmt(r.test_triplets)} · candidates ${fmt(r.candidate_predictions)} · recovered ${fmt(r.recovered_test_triplets)}"><span class="rel-name">${htmlEscape(r.relation)}</span><div class="rel-bar-bg"><div class="rel-bar-fill" style="width:${100*r[metric]}%"></div></div><b>${(100*r[metric]).toFixed(2)}%</b></div>`).join("");
}
document.querySelector("#relationMetric").addEventListener("change",renderRelationshipChart);
document.querySelector("#relationLimit").addEventListener("change",renderRelationshipChart);
renderRelationshipChart();
let pageIndex=0;const pageSize=50;let filtered=[];
function renderTable(){const key=document.querySelector("#which").value,query=document.querySelector("#search").value.trim().toLowerCase(),rows=DATA[key+"_test"]||DATA[key];filtered=rows.filter(t=>!query||t.some(x=>x.toLowerCase().includes(query)));const pages=Math.ceil(filtered.length/pageSize);pageIndex=Math.max(0,Math.min(pageIndex,Math.max(0,pages-1)));const start=pageIndex*pageSize;document.querySelector("#triples").innerHTML=filtered.slice(start,start+pageSize).map(t=>`<tr><td>${htmlEscape(t[0])}</td><td>${htmlEscape(t[1])}</td><td>${htmlEscape(t[2])}</td></tr>`).join("");document.querySelector("#tableCount").textContent=`${fmt(filtered.length)} matching triples`;document.querySelector("#page").textContent=pages?`Page ${pageIndex+1} of ${pages}`:"No results";document.querySelector("#previous").disabled=pageIndex===0;document.querySelector("#next").disabled=pageIndex+1>=pages}
document.querySelector("#which").addEventListener("change",()=>{pageIndex=0;renderTable()});
document.querySelector("#search").addEventListener("input",()=>{pageIndex=0;renderTable()});
document.querySelector("#previous").addEventListener("click",()=>{if(pageIndex>0){pageIndex--;renderTable()}});
document.querySelector("#next").addEventListener("click",()=>{pageIndex++;renderTable()});
document.querySelector("#download").addEventListener("click",()=>{const csv=["head\trelation\ttail",...filtered.map(t=>t.map(x=>`"${x.replaceAll('"','""')}"`).join("\t"))].join("\n");const url=URL.createObjectURL(new Blob([csv],{type:"text/tab-separated-values"}));const a=document.createElement("a");a.href=url;a.download=`${document.querySelector("#which").value}_test_triplets.tsv`;a.click();URL.revokeObjectURL(url)});
renderTable();
</script>
</body>
</html>
"""
    return template.replace("__DASHBOARD_DATA__", payload).replace(
        "__CANDIDATE_EXPLORER__", html.escape(str(data["candidate_explorer"]), quote=True)
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build an evaluation dashboard from existing KG outputs."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Project root containing dataset and results directories.",
    )
    args = parser.parse_args()
    data = build_dashboard(args.root)
    evaluation = data["evaluation"]
    print(
        json.dumps(
            {
                "output": str(args.root.resolve() / "results" / "evaluation_dashboard.html"),
                "candidate_categories": data["candidate_categories"],
                "candidate_category_overlaps": data["candidate_category_overlaps"],
                "test": evaluation,
                "split_overlaps": data["split_overlaps"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
