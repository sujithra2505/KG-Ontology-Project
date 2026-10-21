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
- `results\evaluation_dashboard.html` — test-set precision/recall, category
  overlap checks, and searchable/paginated lists of recovered and missed test
  triples. Generate it from existing files without retraining:

  ```powershell
  python modules\evaluation_dashboard.py
  ```

Open `results\evaluation_dashboard.html` in a browser to inspect the
calculations and test-triplet tables. It checks the saved `metrics.json`
against freshly counted existing dataset/prediction files and fails clearly
if they disagree; it does not retrain or regenerate predictions.
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

## Evaluation protocol

The current pipeline uses **exact-set matching**, not a ranked
link-prediction evaluation. Let `C` be the set of unique triples generated
from training-only rules, and let `S` be either the validation or test split.
The number of observed matches is `H = |C ∩ S|`.

- **Generated candidate edges** are the distinct head/relation/tail triples in
  `predicted_edges.tsv`, excluding triples already present in training.
- **Known held-out matches** are candidates whose complete triple exactly
  matches a row in the selected validation or test split.
- **Unmatched candidates** are in `C` but not in the selected known-positive
  split. They are not thereby confirmed false. A candidate can also be a known
  match in the other held-out split.
- **Precision for split `S`** is `H / |C|`: the denominator is all unique
  generated candidates, not only candidates matching that split. It is zero
  when there are no candidates.
- **Recall for split `S`** is `H / |S|`: the denominator is all known triples
  in that split. It is zero when that split is empty.

Thus the saved test metrics mean `7,335 / 760,370` exact test matches for
precision and `7,335 / 20,466` test triples recovered for recall. Validation
metrics use the same candidate denominator and the validation split size.
These formulas are implemented in `_evaluate_predictions` in
`modules\kg_pipeline.py`; the dashboard reports related candidate categories
and the recovered/missed split rows.

### Ranked metrics are not currently available

The current prediction output has a confidence/support pair per generated
candidate, with confidence aggregated as the maximum confidence of any
producing rule. It does not score every possible head or tail completion for
each query triple, and candidates are generated only when a learned rule
matches an observed training path. Sorting this partial candidate list would
not produce standard MRR or Hits@K link-prediction metrics: the ranking
universe and ranks of omitted entities would be undefined. The current
precision and recall are set metrics and do not depend on candidate ordering.

Before adding MRR or Hits@K, a separate protocol must be specified and
implemented:

1. **Candidate-ranking method:** define a score for every eligible entity
   replacement for each query, rather than ranking only generated positives.
2. **Candidate universe:** specify whether each query ranks head replacements,
   tail replacements, or both, and exactly which entities are eligible.
3. **Filtered evaluation:** define the known-positive filter set (typically
   training, validation, and test facts), remove other known true completions
   from each query's candidates, and restore the target triple before ranking.
4. **Tie handling:** choose and test a deterministic rule, such as assigning
   average rank across equal scores; Hits@K then checks whether that rank is at
   most `K`, and reciprocal rank is its inverse.
5. **Negative/candidate sampling:** full ranking over the defined entity
   universe needs no negative sampling. If candidates are sampled instead,
   fix and record the sampling strategy, candidate count, random seed, and
   filtering of known positives; sampled-ranking results must be labeled as
   such and not presented as full-ranking MRR/Hits@K.

The repository currently implements none of these ranking-universe,
filtered-ranking, or tie-handling steps. Therefore it reports no MRR or
Hits@K. Validation may be used to select a future protocol or model settings;
test metrics must remain reserved for final reporting.

To explain missed test triples from the current saved run, run:

```powershell
python modules\miss_diagnostics.py
```

This reads the training graph, test split, saved rules, predictions, and
parameters, then writes a uniquely named CSV under
`results\diagnostics\`. It does not overwrite the existing pipeline outputs.
The analysis checks the saved rules and candidates against a read-only
recomputation from the training graph. It reports whether each missed triple
has a two-hop training path, whether the corresponding relation rule was
learned, and whether matching paths were excluded by the configured pivot
limit. It also calculates rule support/confidence with and without that limit
to distinguish threshold failures from rules lost to the limit. Up to ten
matching paths are included per triple; the CSV marks when further paths were
omitted from the examples. If saved artifacts do not agree with recomputation,
the diagnostic cause is marked unknown rather than inferred. Test triples are
used only for this post-hoc explanation, not to select rule settings.

To compare path caps without building the full candidate set in Python memory,
run the disk-backed experiment runner:

```powershell
# Small pilot first:
py -3.13 modules\kg_cap_experiments.py --caps 2000

# After reviewing the pilot, compare additional caps:
py -3.13 modules\kg_cap_experiments.py --caps 5000 10000

# Default comparison in one separate timestamped run directory:
py -3.13 modules\kg_cap_experiments.py
```

The runner reads only `train.txt` and `valid.txt`; it never loads `test.txt`.
All rule settings besides the path cap stay fixed at support 2, confidence 0.1,
and 3,000 maximum rules. Candidate triples are deduplicated in a SQLite
database in bounded insert batches, and every run is written to a new
`results\experiments\path_caps_<UTC timestamp>\` directory. Per-cap
`metrics.json` files include runtime, sampled peak RSS, skipped paths,
candidate count, validation precision/recall, and the on-disk candidate
database. These are exact set-match metrics: precision is validation matches
divided by unique candidates; recall is validation matches divided by known
validation triples. Unmatched candidates remain unknown, not false. The
candidate database is retained for inspection and is not an original output.

The conservative pilot and follow-up commands used for the current comparison
were:

```powershell
py -3.13 modules\kg_cap_experiments.py --caps 2000
py -3.13 modules\kg_cap_experiments.py --caps 5000 10000
```

## Validation-only rule tuning

The support/confidence tuning runner holds the path cap at 10,000 and maximum
rules at 3,000. It varies one threshold at a time: minimum support
`1, 2, 4` with confidence fixed at `0.1`, and minimum confidence
`0.05, 0.1, 0.2` with support fixed at `2`. The shared support-2/confidence-0.1
configuration is the baseline for both sweeps. Each configuration runs in a
separate child process and writes its candidates and metrics to a separate
folder; only `train.txt` and `valid.txt` are read.

Reproduce the sweep with:

```powershell
py -3.13 modules\kg_rule_tuning.py
```

The runner records rule count, candidate count, validation exact-match
precision/recall/F1, skipped pivots and path combinations, elapsed time, and
sampled peak process memory. Candidate triples are deduplicated on disk.
Validation F1 (harmonic mean of exact-set precision and recall) is the
selection metric; ties are broken by precision, then fewer candidates. No test
metrics are read by the runner.

For the recorded run in
`results\experiments\rule_tuning_20261021T080249887658Z\`, support `2` and
confidence `0.2` had the highest validation F1: `0.020327` (precision
`1.0511%`, recall `30.7442%`) across `512,886` candidates and `1,777` rules.
The baseline support `2` / confidence `0.1` yielded F1 `0.016424` (precision
`0.8401%`, recall `36.4300%`) across `760,370` candidates and `2,384` rules.
The selected setting has lower validation recall, but higher precision and
F1 with fewer candidates; it was not selected on recall alone. Lowering
confidence to `0.05` reached the 3,000-rule cap, produced `1,162,296`
candidates, reduced precision, and had a lower F1. The selected thresholds
are recorded as the validation-selected configuration only; no final test
evaluation has been run.

## Candidate scoring experiment

The original candidate score remains the maximum confidence of any learned
rule that produces a candidate. A separate validation-only experiment compares
that baseline with a noisy-OR score over distinct supporting rules:
`1 - product(1 - rule_confidence)`. Evidence is keyed by the candidate and
the rule's two body relations; repeated paths using the same rule are stored
once, so they do not inflate its score. The method assumes distinct rules
provide independent evidence, which may not hold for correlated rules.

Run the experiment using the validation-selected Phase 4 settings:

```powershell
py -3.13 modules\kg_scoring_experiment.py
```

Candidates and per-rule evidence are deduplicated in a separate SQLite
database under a new `results\experiments\candidate_scoring_<UTC timestamp>\`
directory. The run reads training and validation only. Because the score
change does not alter candidate generation, candidate coverage and the
existing exact-set precision, recall, and F1 are identical for both scoring
methods. For the recorded run in
`results\experiments\candidate_scoring_20261021T081816544994Z\`, both scores
cover 512,886 candidates and recover 5,391 validation triples (precision
`1.0511%`, recall `30.7442%`, F1 `0.020327`). The noisy-OR score changed
199,595 candidate scores and increased the mean score by `0.08851` relative
to max-rule confidence. These are score-distribution changes, not evidence of
better ranking: no ranking lift is claimed.

The recorded run used about 286.8 MB peak sampled RSS and wrote a 366 MB
SQLite database. Its measured runtime was 12,164.6 seconds; inspection traced
the excessive time to committing the candidate database once per pivot. The
runner now batches commits at 50,000 generated evidence rows. The saved
metrics retain the original run's measured runtime; reproducing after this
write-batching change will measure the updated implementation.

To compare prioritization, a future evaluation should rank all eligible head
and tail replacements from the training entity universe for each validation
query. Filter other known positives from the train/validation/test fact union,
restore the query target, use average rank for ties, and report MRR/Hits@K.
Test facts must not influence tuning. If full entity ranking is impractical,
the sampled universe, candidate count, negative-sampling method, and random
seed must be fixed and disclosed; the result should be identified as sampled
ranking.

## Experiment run manifests

The experiment runners now write a separate `run_manifest.json` alongside
their outputs. The miss-diagnostics runner writes a timestamp-matched
`test_miss_diagnostics_<timestamp>.manifest.json` beside its CSV. Manifests
record SHA-256 and byte size for the train, validation, and test split files,
whether each split was used for learning or evaluation, configured rule/path
and candidate/scoring settings, measured metrics and evaluation definitions,
and filenames, sizes, and SHA-256 hashes for the run outputs. A test split
hash identifies the input file; it does not mean the test split was used for
tuning or evaluation.

Configured parameters are kept separate from measured results. New runs capture
dataset split hashes, Git commit/worktree state, and runner/source hashes at
run start; output hashes are computed when the run finishes. Manifests added
after an older run are marked `retrospective_from_saved_outputs`: their split
hashes identify the files as they exist now, and the current Git commit is not
claimed as the historical execution version. A manifest hashes run outputs,
but not itself.

Large experiment SQLite databases are intentionally excluded from the source
commit. Their names, sizes, and hashes are retained under `external_outputs`
and can be verified while the local files are present. A checkout without
those external files will report them as missing; their recorded hashes can
verify a separately transferred copy.

Verify the output files for a run from the project root with:

```powershell
py -3.13 -c "import json; from pathlib import Path; from modules.run_provenance import verify_output_hashes; p=Path(r'results\experiments\<run>\run_manifest.json'); errors=verify_output_hashes(Path.cwd(), json.loads(p.read_text(encoding='utf-8'))); print(errors or 'All output hashes verified')"
```

The empty `results\experiments\path_caps_20261021T073532229672Z\cap_2000\`
directory from the earlier interrupted startup attempt remains untouched. It
contains no recorded metrics or outputs to hash and is not represented as a
completed experiment.
