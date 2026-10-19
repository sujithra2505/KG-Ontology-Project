# Knowledge graph, ontology, and missing-edge prediction

The project uses the FB15K-237-style `dataset/train.txt`, `valid.txt`, and
`test.txt` files. Run the end-to-end pipeline from the project root with the
workspace's selected Python environment:

```powershell
python -m pip install -r requirements.txt
python modules\kg_pipeline.py
```

The pipeline only uses `train.txt` to build the graph, create ontology
assertions, learn rules, and generate predictions. It writes:

- `results\knowledge_graph.graphml` — directed, relation-labeled NetworkX graph.
- `results\full_kg_viewer.html` — standalone interactive browser view of the
  complete graph. Search/click entities, inspect their neighboring facts, and
  filter by relation. Open this HTML file in a browser; it contains the graph
  data and does not require an internet connection.
- `results\ontology.ttl` — RDF/OWL ontology with a head-role and tail-role
  class for every relation, domain/range declarations, entity types, and
  relation assertions.
- `results\truths.tsv` — training triples asserted in the ontology.
- `results\learned_rules.tsv` — mined rules of the form
  `r1(X,Y) AND r2(Y,Z) => r3(X,Z)`.
- `results\predicted_edges.tsv` — unique predicted triples absent from the
  training graph, with rule confidence and support.
- `results\prediction_explorer.html` — interactive view of all predicted
  edges. Search/filter the candidate list, including validation/test matches,
  and select any prediction to display its two observed supporting training
  edges and applicable rule paths.
- `results\prediction_example.svg` — a visual explanation of one prediction
  that matches held-out data, showing its two training-edge supports and the
  independently observed predicted edge.
- `results\prediction_proof.svg` — an edge-by-edge proof diagram for the
  verified `/m/0226k3` prediction, including both training facts, the applied
  rule, confidence, and its validation match.
- `results\metrics.json` — graph/ontology counts, prediction counts, and
  validation/test overlap, precision, and recall.

By default, rules need support of at least 2 and confidence of at least 0.1;
the pipeline keeps up to 3,000 qualified rules (enough to retain all rules
found in the supplied training data). A pivot entity is skipped if expanding
its incoming/outgoing edge pairs would exceed 10,000 two-hop paths. This
avoids excessive work at very high-degree entities; the number of skipped
pivots and paths is recorded in `metrics.json`. Adjust these settings with
`--min-support`, `--min-confidence`, `--max-rules`, and
`--max-paths-per-pivot`.

Matches against validation/test are known held-out positive triples.
Non-matching predictions are not confirmed false: knowledge graph datasets
are incomplete, so an unobserved edge may still be true.
