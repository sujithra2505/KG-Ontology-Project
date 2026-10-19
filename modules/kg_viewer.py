"""Generate a standalone browser viewer for the complete training graph."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

Triple = tuple[str, str, str]


_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>FB15K-237 Complete Knowledge Graph</title>
<style>
:root{color-scheme:dark;--bg:#0b1220;--panel:#111c2e;--border:#26354d;--text:#e5edf8;--muted:#9aabc2;--accent:#59b7ff}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 system-ui,"Segoe UI",sans-serif}
header{position:relative;z-index:2;padding:14px 18px;background:var(--panel);border-bottom:1px solid var(--border);display:flex;align-items:center;gap:14px;flex-wrap:wrap}
h1{font-size:17px;margin:0 10px 0 0;white-space:nowrap}.stat{color:var(--muted);white-space:nowrap}.stat b{color:var(--text);font-variant-numeric:tabular-nums}
input,select,button{font:inherit;color:var(--text);background:#0b1423;border:1px solid var(--border);border-radius:7px;padding:8px 10px}
input{width:min(330px,80vw)}select{max-width:min(410px,88vw)}button{cursor:pointer}button:hover{border-color:var(--accent)}
main{height:calc(100vh - 68px);min-height:420px;display:grid;grid-template-columns:minmax(0,1fr) 300px}
#graph{width:100%;height:100%;display:block;touch-action:none}
aside{padding:16px;background:var(--panel);border-left:1px solid var(--border);overflow:auto}
aside h2{font-size:14px;margin:0 0 9px}aside p{color:var(--muted);margin:7px 0 14px;overflow-wrap:anywhere}
.hint{font-size:12px;color:var(--muted);margin-top:12px}.legend{display:flex;align-items:center;gap:8px;margin:9px 0;color:var(--muted);font-size:12px}
.dot{width:10px;height:10px;border-radius:50%;display:inline-block}.edge{width:22px;height:0;border-top:2px solid #59b7ff}.edge.dashed{border-top-style:dashed;border-color:#f59e0b}
.item{border-top:1px solid var(--border);padding:9px 0;overflow-wrap:anywhere}.item small{display:block;color:var(--muted)}
#neighbors{margin-top:12px}.pill{display:inline-block;margin:3px 3px 0 0;padding:3px 6px;background:#1b2c43;border-radius:5px;color:#c6d5e8;font-size:11px}
@media(max-width:800px){main{grid-template-columns:1fr;grid-template-rows:minmax(320px,65vh) auto;height:auto}aside{border-left:0;border-top:1px solid var(--border)}header{gap:8px}h1{width:100%}}
</style>
</head>
<body>
<header>
  <h1>Complete knowledge graph</h1>
  <span class="stat"><b id="nodeCount"></b> entities</span>
  <span class="stat"><b id="edgeCount"></b> training edges</span>
  <input id="search" list="entityList" placeholder="Find an entity (e.g. /m/...)">
  <datalist id="entityList"></datalist>
  <select id="relation"><option value="">All relations</option></select>
  <button id="showAll">Show all</button>
  <button id="fit">Fit graph</button>
</header>
<main>
  <canvas id="graph" aria-label="Interactive graph canvas"></canvas>
  <aside>
    <h2 id="status">All graph edges</h2>
    <p id="summary">Loading graph data…</p>
    <div class="legend"><span class="dot" style="background:#59b7ff"></span>Entities</div>
    <div class="legend"><span class="edge"></span>Observed training fact (arrowheads when focused)</div>
    <div class="legend"><span class="dot" style="background:#f59e0b"></span>Selected entity</div>
    <div id="selection"></div>
    <div class="hint">Search an entity or click a node to inspect its direct incoming and outgoing facts. Choose a relation to isolate its edges. Scroll to zoom; drag to pan. Use “Show all” to return to the full graph.</div>
  </aside>
</main>
<script>
const DATA = __KG_DATA__;
const canvas=document.getElementById("graph"),ctx=canvas.getContext("2d");
const relationSelect=document.getElementById("relation"),search=document.getElementById("search");
const nodes=DATA.nodes,edges=DATA.edges,relations=DATA.relations;
const nodeByName=new Map(nodes.map((n,i)=>[n[0],i])),adj=Array.from({length:nodes.length},()=>[]);
for(let i=0;i<edges.length;i++){const e=edges[i];adj[e[0]].push(i);if(e[1]!==e[0])adj[e[1]].push(i)}
document.getElementById("nodeCount").textContent=nodes.length.toLocaleString();
document.getElementById("edgeCount").textContent=edges.length.toLocaleString();
for(const r of relations){const o=document.createElement("option");o.value=String(r[0]);o.textContent=r[1];relationSelect.appendChild(o)}
const datalist=document.getElementById("entityList");
for(const n of nodes){const o=document.createElement("option");o.value=n[0];datalist.appendChild(o)}
let focus=null, panX=0, panY=0, scale=1, dragging=false, lastX=0,lastY=0;
let visibleEdges=edges.map((_,i)=>i);
const colorFor=(id)=>{let h=(id*137.508)%360;return `hsl(${h},72%,66%)`};
function resize(){const r=canvas.getBoundingClientRect(),dpr=window.devicePixelRatio||1;canvas.width=Math.max(1,Math.floor(r.width*dpr));canvas.height=Math.max(1,Math.floor(r.height*dpr));ctx.setTransform(dpr,0,0,dpr,0,0);fit(false)}
function fit(redraw=true){const w=canvas.clientWidth,h=canvas.clientHeight;scale=Math.max(.02,Math.min(w,h)/2.25);panX=w/2;panY=h/2;if(redraw)draw()}
function xy(i){const n=nodes[i],w=canvas.clientWidth,h=canvas.clientHeight;return [panX+n[2]*scale,panY+n[3]*scale]}
function chooseEdges(){const r=relationSelect.value;visibleEdges=[];if(focus!==null){for(const id of adj[focus])if(!r||String(edges[id][2])===r)visibleEdges.push(id)}else if(r){for(let i=0;i<edges.length;i++)if(String(edges[i][2])===r)visibleEdges.push(i)}else{visibleEdges=edges.map((_,i)=>i)}}
function draw(){const w=canvas.clientWidth,h=canvas.clientHeight;if(!w||!h)return;ctx.clearRect(0,0,w,h);ctx.fillStyle="#0b1220";ctx.fillRect(0,0,w,h);
 const focused=focus!==null,filtered=relationSelect.value!=="",neighborSet=new Set();
 if(focused)for(const id of adj[focus]){const e=edges[id];neighborSet.add(e[0]);neighborSet.add(e[1])}
 ctx.lineWidth=focused?1.5:.65;
 for(const id of visibleEdges){const e=edges[id],a=xy(e[0]),b=xy(e[1]);ctx.beginPath();ctx.moveTo(a[0],a[1]);ctx.lineTo(b[0],b[1]);ctx.strokeStyle=focused||filtered?colorFor(e[2]):"rgba(111,145,183,.10)";ctx.globalAlpha=focused ? .72 : (filtered ? .38 : 1);ctx.stroke();if(focused){const angle=Math.atan2(b[1]-a[1],b[0]-a[0]),size=6;ctx.beginPath();ctx.moveTo(b[0],b[1]);ctx.lineTo(b[0]-size*Math.cos(angle-.45),b[1]-size*Math.sin(angle-.45));ctx.lineTo(b[0]-size*Math.cos(angle+.45),b[1]-size*Math.sin(angle+.45));ctx.closePath();ctx.fillStyle=colorFor(e[2]);ctx.fill()}}
 ctx.globalAlpha=1;
 const pointSize=focused?3.2:(scale>180?1.1:.48);
 for(let i=0;i<nodes.length;i++){if(focused&&i!==focus&&!neighborSet.has(i))continue;const p=xy(i);ctx.beginPath();ctx.arc(p[0],p[1],i===focus?6:pointSize,0,Math.PI*2);ctx.fillStyle=i===focus?"#f59e0b":(focused?"#79c8ff":"#71b8ef");ctx.globalAlpha=focused||filtered?1:.72;ctx.fill()}
 ctx.globalAlpha=1;
 if(focus!==null){const p=xy(focus);ctx.font="12px system-ui";ctx.fillStyle="#fff";ctx.fillText(nodes[focus][0],p[0]+10,p[1]-10)}
 document.getElementById("status").textContent=focus!==null?"Entity neighborhood":(relationSelect.value?"Filtered relation":"Complete graph");
 document.getElementById("summary").textContent=`Showing ${visibleEdges.length.toLocaleString()} of ${edges.length.toLocaleString()} training edges.`;
}
function describe(i){focus=i;chooseEdges();const n=nodes[i],facts=adj[i].map(id=>edges[id]);document.getElementById("selection").replaceChildren();const box=document.createElement("div");box.className="item";const title=document.createElement("strong");title.textContent=n[0];const detail=document.createElement("small");detail.textContent=`${n[1].toLocaleString()} incident edge${n[1]===1?"":"s"}`;box.append(title,detail);
 const list=document.createElement("div");list.id="neighbors";const shown=facts.slice(0,80);for(const e of shown){const row=document.createElement("div");row.className="item";const rel=document.createElement("span");rel.className="pill";rel.textContent=relations[e[2]][1];const direction=document.createElement("small");const other=e[0]===i?e[1]:e[0];direction.textContent=e[0]===i?`outgoing → ${nodes[other][0]}`:`incoming ← ${nodes[other][0]}`;row.append(rel,direction);list.appendChild(row)}if(facts.length>shown.length){const more=document.createElement("small");more.textContent=`Showing first 80 of ${facts.length.toLocaleString()} incident edges.`;list.appendChild(more)}box.appendChild(list);document.getElementById("selection").appendChild(box);draw()}
function findEntity(value){if(!value)return;let i=nodeByName.get(value);if(i===undefined){i=nodes.findIndex(n=>n[0].toLowerCase().includes(value.toLowerCase()))}if(i<0){document.getElementById("summary").textContent="No matching entity found.";return}describe(i);const p=xy(i);panX+=canvas.clientWidth/2-p[0];panY+=canvas.clientHeight/2-p[1];draw()}
search.addEventListener("keydown",e=>{if(e.key==="Enter")findEntity(search.value.trim())});
search.addEventListener("change",()=>findEntity(search.value.trim()));
relationSelect.addEventListener("change",()=>{chooseEdges();draw()});
document.getElementById("showAll").addEventListener("click",()=>{focus=null;search.value="";document.getElementById("selection").replaceChildren();chooseEdges();draw()});
document.getElementById("fit").addEventListener("click",()=>fit());
canvas.addEventListener("pointerdown",e=>{dragging=true;lastX=e.clientX;lastY=e.clientY;canvas.setPointerCapture(e.pointerId)});
canvas.addEventListener("pointermove",e=>{if(!dragging)return;panX+=e.clientX-lastX;panY+=e.clientY-lastY;lastX=e.clientX;lastY=e.clientY;draw()});
canvas.addEventListener("pointerup",e=>{dragging=false;const rect=canvas.getBoundingClientRect(),mx=e.clientX-rect.left,my=e.clientY-rect.top;let best=-1,dist=14;for(let i=0;i<nodes.length;i++){const p=xy(i),d=Math.hypot(p[0]-mx,p[1]-my);if(d<dist){dist=d;best=i}}if(best>=0&&!dragMoved)describe(best);dragMoved=false});
let dragMoved=false;canvas.addEventListener("pointermove",e=>{if(dragging)dragMoved=true});
canvas.addEventListener("wheel",e=>{e.preventDefault();const old=scale,factor=e.deltaY<0?1.12:.89;scale=Math.max(.03,Math.min(1500,scale*factor));const r=canvas.getBoundingClientRect(),x=e.clientX-r.left,y=e.clientY-r.top;panX=x-(x-panX)*(scale/old);panY=y-(y-panY)*(scale/old);draw()},{passive:false});
window.addEventListener("resize",resize);chooseEdges();resize();draw();
</script>
</body>
</html>
"""


def write_full_graph_viewer(triples: Iterable[Triple], output_path: Path) -> None:
    """Write a self-contained HTML canvas viewer with the complete graph data."""
    triples = list(triples)
    names = sorted({entity for head, _, tail in triples for entity in (head, tail)})
    node_ids = {name: index for index, name in enumerate(names)}
    degrees = [0] * len(names)
    for head, _, tail in triples:
        degrees[node_ids[head]] += 1
        degrees[node_ids[tail]] += 1

    # Put high-degree entities toward the center for a useful overview layout.
    order = sorted(range(len(names)), key=lambda i: (-degrees[i], names[i]))
    positions: list[tuple[float, float]] = [(0.0, 0.0)] * len(names)
    golden_angle = math.pi * (3 - math.sqrt(5))
    denominator = max(len(order) - 1, 1)
    for rank, node_id in enumerate(order):
        radius = 0.96 * math.sqrt(rank / denominator)
        angle = rank * golden_angle
        positions[node_id] = (radius * math.cos(angle), radius * math.sin(angle))

    relation_names = sorted({relation for _, relation, _ in triples})
    relation_ids = {name: index for index, name in enumerate(relation_names)}
    data = {
        "nodes": [
            [name, degrees[index], positions[index][0], positions[index][1]]
            for index, name in enumerate(names)
        ],
        "relations": [[index, name] for index, name in enumerate(relation_names)],
        "edges": [
            [node_ids[head], node_ids[tail], relation_ids[relation]]
            for head, relation, tail in triples
        ],
    }
    html = _HTML.replace(
        "__KG_DATA__",
        json.dumps(data, ensure_ascii=True, separators=(",", ":")),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
