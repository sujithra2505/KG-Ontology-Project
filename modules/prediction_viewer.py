"""Build a standalone viewer for predictions and their supporting KG paths."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .kg_pipeline import PathRule

Triple = tuple[str, str, str]


_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Predicted edges and supporting paths</title>
<style>
:root{color-scheme:dark;--bg:#0b1220;--panel:#111c2e;--border:#26354d;--text:#e5edf8;--muted:#9aabc2;--blue:#59b7ff;--green:#31c978;--orange:#f59e0b}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 system-ui,"Segoe UI",sans-serif}
header{padding:15px 20px;background:var(--panel);border-bottom:1px solid var(--border)}
h1{font-size:19px;margin:0 0 8px}.subtitle{color:var(--muted)}main{max-width:1500px;margin:auto;padding:16px;display:grid;gap:14px}
section{background:var(--panel);border:1px solid var(--border);border-radius:10px;padding:14px}
.controls{display:flex;gap:9px;flex-wrap:wrap;align-items:center}
input,select,button{font:inherit;color:var(--text);background:#0b1423;border:1px solid var(--border);border-radius:7px;padding:9px 11px}
input{min-width:220px;flex:1}select{max-width:300px}button{cursor:pointer}button:hover{border-color:var(--blue)}
#counts{color:var(--muted);margin:11px 0 0}.layout{display:grid;grid-template-columns:minmax(0,1.4fr) minmax(320px,1fr);gap:14px}
h2{font-size:15px;margin:0 0 10px}.table-wrap{max-height:62vh;overflow:auto}
table{border-collapse:collapse;width:100%;font-size:12px}th,td{text-align:left;padding:8px 7px;border-bottom:1px solid var(--border);overflow-wrap:anywhere}
th{position:sticky;top:0;background:#17243a;color:#c7d5e7}td.entity{font-family:ui-monospace,Consolas,monospace}
.status{display:inline-block;border-radius:12px;padding:2px 7px;background:#26354d;color:#c7d5e7;font-size:11px}
.status.valid,.status.test{background:#123d30;color:#9ef0c5}.status.unknown{background:#3a2a14;color:#ffd58c}
#graph{width:100%;min-height:250px;border:1px solid var(--border);border-radius:8px;background:#0b1423}
.edge-list{padding:10px 0;border-top:1px solid var(--border);overflow-wrap:anywhere;font-family:ui-monospace,Consolas,monospace;font-size:12px}
.edge-list b{font-family:system-ui,sans-serif}.green{color:var(--green)}.orange{color:var(--orange)}
.pager{display:flex;gap:8px;align-items:center;padding:9px 2px;color:var(--muted)}
#proofStatus{color:var(--muted);margin:6px 0 10px}#selectedEdge{overflow-wrap:anywhere;font-family:ui-monospace,Consolas,monospace}
@media(max-width:900px){
  .layout{grid-template-columns:1fr}.table-wrap{max-height:45vh}
  table,tbody{display:block;width:100%}thead{display:none}
  tbody tr{display:block;padding:7px 0;border-bottom:1px solid var(--border)}
  tbody td{display:grid;grid-template-columns:88px minmax(0,1fr);gap:8px;padding:5px 3px;border:0;overflow-wrap:anywhere;word-break:break-word}
  tbody td:before{content:attr(data-label);font-weight:700;color:var(--muted)}
}
</style>
</head>
<body>
<header>
  <h1>Predicted edges and their supporting paths</h1>
  <div class="subtitle">Choose any predicted edge to see the two observed training edges and rule that produced it.</div>
</header>
<main>
  <section>
    <div class="controls">
      <input id="query" placeholder="Filter by entity or relation; press Enter">
      <select id="truthFilter">
        <option value="all">All predictions</option>
        <option value="valid">Matches validation</option>
        <option value="test">Matches test</option>
        <option value="unknown">No held-out match</option>
      </select>
      <button id="apply">Apply filters</button>
      <button id="clear">Clear</button>
    </div>
    <div id="counts">Loading prediction data…</div>
  </section>
  <div class="layout">
    <section>
      <h2>Predicted edges</h2>
      <div class="table-wrap"><table>
        <thead><tr><th>Head</th><th>Predicted relation</th><th>Tail</th><th>Confidence</th><th>Check</th><th></th></tr></thead>
        <tbody id="rows"></tbody>
      </table></div>
      <div class="pager"><button id="previous">Previous</button><span id="page"></span><button id="next">Next</button></div>
    </section>
    <section>
      <h2>Proof graph</h2>
      <div id="selectedEdge">Select a prediction from the table.</div>
      <div id="proofStatus"></div>
      <div class="controls"><button id="proofPrevious" disabled>Previous support path</button><button id="proofNext" disabled>Next support path</button></div>
      <svg id="graph" viewBox="0 0 900 330" role="img" aria-label="Predicted edge and two supporting training edges"></svg>
      <div id="facts"></div>
      <p class="subtitle">Green solid arrows are observed training facts. The orange dashed arrow is the predicted edge. A validation/test match is an observed held-out positive; a prediction with no match is unknown, not necessarily false.</p>
    </section>
  </div>
</main>
<script>
const DATA=__PREDICTION_DATA__;
const nodeNames=DATA.nodes, relNames=DATA.relations, trainEdges=DATA.trainEdges;
const rules=DATA.rules, predictions=DATA.predictions;
const outgoingByNodeRelation=new Map(), outgoingByNode=Array.from({length:nodeNames.length},()=>[]), rulesByFirstRelation=new Map();
for(const [h,r,t] of trainEdges){
  const key=h*relNames.length+r;
  if(!outgoingByNodeRelation.has(key)){outgoingByNodeRelation.set(key,[]);outgoingByNode[h].push([r,outgoingByNodeRelation.get(key)])}
  outgoingByNodeRelation.get(key).push(t);
}
for(const rule of rules){
  if(!rulesByFirstRelation.has(rule[0]))rulesByFirstRelation.set(rule[0],[]);
  rulesByFirstRelation.get(rule[0]).push(rule);
}
const pageSize=100;let matches=[],pageIndex=0,selectedPrediction=-1,proofs=[],proofIndex=0;
const escapeHtml=value=>String(value).replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const statusName=flag=>flag===1?"Validation":flag===2?"Test":"No held-out match";
const statusClass=flag=>flag===1?"valid":flag===2?"test":"unknown";
function matchesFilters(p,q,filter){
  if(filter==="valid"&&p[5]!==1)return false;
  if(filter==="test"&&p[5]!==2)return false;
  if(filter==="unknown"&&p[5]!==0)return false;
  if(!q)return true;
  return nodeNames[p[0]].toLowerCase().includes(q)
    ||relNames[p[1]].toLowerCase().includes(q)
    ||nodeNames[p[2]].toLowerCase().includes(q);
}
function applyFilters(){
  const q=document.querySelector("#query").value.trim().toLowerCase();
  const filter=document.querySelector("#truthFilter").value;
  matches=[];
  for(let i=0;i<predictions.length;i++)if(matchesFilters(predictions[i],q,filter))matches.push(i);
  pageIndex=0;renderRows();
}
function renderRows(){
  const start=pageIndex*pageSize,end=Math.min(start+pageSize,matches.length),tbody=document.querySelector("#rows");
  const chunk=[];
  for(let j=start;j<end;j++){
    const i=matches[j],p=predictions[i],flag=p[5];
    chunk.push(`<tr><td data-label="Head" class="entity">${escapeHtml(nodeNames[p[0]])}</td><td data-label="Relation">${escapeHtml(relNames[p[1]])}</td><td data-label="Tail" class="entity">${escapeHtml(nodeNames[p[2]])}</td><td data-label="Confidence">${p[3].toFixed(3)}</td><td data-label="Dataset match"><span class="status ${statusClass(flag)}">${statusName(flag)}</span></td><td data-label=""><button data-index="${i}">Proof</button></td></tr>`);
  }
  tbody.innerHTML=chunk.join("");
  document.querySelector("#counts").textContent=`Showing ${matches.length.toLocaleString()} of ${predictions.length.toLocaleString()} predicted edges · ${nodeNames.length.toLocaleString()} entities · ${trainEdges.length.toLocaleString()} training edges`;
  document.querySelector("#page").textContent=matches.length?`Page ${pageIndex+1} of ${Math.ceil(matches.length/pageSize)}`:"No matching predictions";
  document.querySelector("#previous").disabled=pageIndex===0;
  document.querySelector("#next").disabled=end>=matches.length;
}
function makeProofs(prediction){
  const [head,targetRelation,tail]=prediction;
  const found=[],seen=new Set();
  for(const [r1,middles] of outgoingByNode[head]){
    for(const middle of middles){
    for(const rule of rulesByFirstRelation.get(r1)||[]){
      const [ruleR1,r2,r3,support,confidence]=rule;
      if(r3!==targetRelation)continue;
      const tails=outgoingByNodeRelation.get(middle*relNames.length+r2)||[];
      for(const candidateTail of tails){
        if(candidateTail!==tail)continue;
        const key=`${r1},${middle},${r2},${r3}`;
        if(seen.has(key))continue;
        seen.add(key);found.push({r1,middle,r2,r3,support,confidence});
      }
    }
    }
  }
  found.sort((a,b)=>b.confidence-a.confidence||b.support-a.support);
  return found;
}
function proofSvg(prediction,proof){
  const [head,targetRelation,tail]=prediction;
  const h=escapeHtml(nodeNames[head]),m=escapeHtml(nodeNames[proof.middle]),t=escapeHtml(nodeNames[tail]);
  const r1=escapeHtml(relNames[proof.r1]),r2=escapeHtml(relNames[proof.r2]),r3=escapeHtml(relNames[targetRelation]);
  return `<defs>
    <marker id="greenArrow" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto"><path d="M0 0 L10 5 L0 10z" fill="#31c978"/></marker>
    <marker id="orangeArrow" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto"><path d="M0 0 L10 5 L0 10z" fill="#f59e0b"/></marker>
  </defs>
  <path d="M190 112 Q450 8 710 112" fill="none" stroke="#f59e0b" stroke-width="4" stroke-dasharray="10 7" marker-end="url(#orangeArrow)"/>
  <text x="450" y="29" fill="#ffd58c" text-anchor="middle" font-size="13">PREDICTED EDGE r3</text>
  <line x1="235" y1="200" x2="365" y2="200" stroke="#31c978" stroke-width="4" marker-end="url(#greenArrow)"/>
  <line x1="535" y1="200" x2="665" y2="200" stroke="#31c978" stroke-width="4" marker-end="url(#greenArrow)"/>
  <circle cx="150" cy="200" r="58" fill="#18304b" stroke="#59b7ff" stroke-width="3"/>
  <circle cx="450" cy="200" r="58" fill="#25334a" stroke="#9aabc2" stroke-width="3"/>
  <circle cx="750" cy="200" r="58" fill="#183b2e" stroke="#31c978" stroke-width="3"/>
  <text x="150" y="195" text-anchor="middle" fill="#e5edf8" font-weight="700">HEAD</text><text x="150" y="216" text-anchor="middle" fill="#e5edf8" font-size="12">${h}</text>
  <text x="450" y="195" text-anchor="middle" fill="#e5edf8" font-weight="700">MIDDLE</text><text x="450" y="216" text-anchor="middle" fill="#e5edf8" font-size="12">${m}</text>
  <text x="750" y="195" text-anchor="middle" fill="#e5edf8" font-weight="700">TAIL</text><text x="750" y="216" text-anchor="middle" fill="#e5edf8" font-size="12">${t}</text>
  <text x="300" y="249" text-anchor="middle" fill="#9ef0c5" font-size="11">TRAINING FACT r1</text>
  <text x="600" y="249" text-anchor="middle" fill="#9ef0c5" font-size="11">TRAINING FACT r2</text>`;
}
function showProof(index){
  selectedPrediction=index;proofs=makeProofs(predictions[index]);proofIndex=0;
  const p=predictions[index],flag=p[5];
  document.querySelector("#selectedEdge").textContent=`Predicted: (${nodeNames[p[0]]}, ${relNames[p[1]]}, ${nodeNames[p[2]]})`;
  document.querySelector("#facts").innerHTML="";
  if(!proofs.length){
    document.querySelector("#proofStatus").textContent="No matching training path was found for this prediction.";
    document.querySelector("#graph").innerHTML="";
    document.querySelector("#proofPrevious").disabled=true;document.querySelector("#proofNext").disabled=true;return;
  }
  drawProof();
  document.querySelector("#proofStatus").textContent=`${proofs.length.toLocaleString()} supporting path${proofs.length===1?"":"s"} found · ${statusName(flag)} · overall confidence ${p[3].toFixed(3)} · reported rule support ${p[4]}`;
}
function drawProof(){
  const p=predictions[selectedPrediction],proof=proofs[proofIndex],head=p[0],tail=p[2];
  document.querySelector("#graph").innerHTML=proofSvg(p,proof);
  const facts=[
    [`Observed training fact 1`,`(${nodeNames[head]}, ${relNames[proof.r1]}, ${nodeNames[proof.middle]})`],
    [`Observed training fact 2`,`(${nodeNames[proof.middle]}, ${relNames[proof.r2]}, ${nodeNames[tail]})`],
    [`Applied rule`,`${relNames[proof.r1]}(X,Y) AND ${relNames[proof.r2]}(Y,Z) => ${relNames[proof.r3]}(X,Z)`],
    [`This path's rule statistics`,`confidence ${proof.confidence.toFixed(3)} · support ${proof.support}`]
  ];
  document.querySelector("#facts").innerHTML=facts.map((f,i)=>`<div class="edge-list ${i<2?"green":""}"><b>${f[0]}:</b><br>${escapeHtml(f[1])}</div>`).join("");
  document.querySelector("#proofPrevious").disabled=proofIndex===0;
  document.querySelector("#proofNext").disabled=proofIndex>=proofs.length-1;
  document.querySelector("#proofStatus").textContent=`Support path ${proofIndex+1} of ${proofs.length.toLocaleString()} · ${statusName(p[5])} · overall prediction confidence ${p[3].toFixed(3)}`;
}
document.querySelector("#apply").addEventListener("click",applyFilters);
document.querySelector("#query").addEventListener("keydown",e=>{if(e.key==="Enter")applyFilters()});
document.querySelector("#clear").addEventListener("click",()=>{document.querySelector("#query").value="";document.querySelector("#truthFilter").value="all";applyFilters()});
document.querySelector("#truthFilter").addEventListener("change",applyFilters);
document.querySelector("#previous").addEventListener("click",()=>{if(pageIndex>0){pageIndex--;renderRows()}});
document.querySelector("#next").addEventListener("click",()=>{if((pageIndex+1)*pageSize<matches.length){pageIndex++;renderRows()}});
document.querySelector("#rows").addEventListener("click",e=>{const button=e.target.closest("button[data-index]");if(button)showProof(Number(button.dataset.index))});
document.querySelector("#proofPrevious").addEventListener("click",()=>{if(proofIndex>0){proofIndex--;drawProof()}});
document.querySelector("#proofNext").addEventListener("click",()=>{if(proofIndex+1<proofs.length){proofIndex++;drawProof()}});
applyFilters();
</script>
</body>
</html>
"""


def write_prediction_explorer(
    triples: Iterable[Triple],
    predictions: dict[Triple, tuple[float, int]],
    rules: Iterable[PathRule],
    validation: Iterable[Triple],
    test: Iterable[Triple],
    output_path: Path,
) -> None:
    """Write all predictions with an on-demand graph of their training proofs."""
    triples = list(triples)
    rules = list(rules)
    validation_set = set(validation)
    test_set = set(test)

    node_names = sorted({entity for head, _, tail in triples for entity in (head, tail)})
    node_ids = {name: index for index, name in enumerate(node_names)}
    relation_names = sorted({relation for _, relation, _ in triples})
    relation_ids = {name: index for index, name in enumerate(relation_names)}

    data = {
        "nodes": node_names,
        "relations": relation_names,
        "trainEdges": [
            [node_ids[head], relation_ids[relation], node_ids[tail]]
            for head, relation, tail in triples
        ],
        "rules": [
            [
                relation_ids[rule.first_relation],
                relation_ids[rule.second_relation],
                relation_ids[rule.predicted_relation],
                rule.support,
                rule.confidence,
            ]
            for rule in rules
        ],
        "predictions": [
            [
                node_ids[head],
                relation_ids[relation],
                node_ids[tail],
                confidence,
                support,
                2 if (head, relation, tail) in test_set else
                1 if (head, relation, tail) in validation_set else 0,
            ]
            for (head, relation, tail), (confidence, support) in sorted(
                predictions.items(),
                key=lambda item: (-item[1][0], item[0]),
            )
        ],
    }
    document = _HTML.replace(
        "__PREDICTION_DATA__",
        json.dumps(data, ensure_ascii=True, separators=(",", ":")),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(document, encoding="utf-8")
