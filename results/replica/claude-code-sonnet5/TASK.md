# DocSEM — task specification

Read this, then produce `predictions.jsonl` in your own working directory.

## What the task is

Each instance gives you a document and a `user_query`. Locate the relevant
quantitative passage in the document, reason **only from that passage**, and
return the final numerical answer together with the identifier of the
supporting block.

Documents contain background prose, tables, dates, numerical facts, and
unrelated questions. Do not combine facts from different passages unless the
target scenario explicitly requires it.

The `user_query` paraphrases the information need. It does not copy the target
question, its numerical inputs, or any identifier from the document.

Every content block begins with a visible identifier followed by the block
text. Use those identifiers when reporting evidence, copied exactly as printed.

## Input

`../input/tasks.jsonl` — one JSON object per line:

```json
{"instance_id": "synth_000001",
 "user_query": "According to the brief's quantitative scenario concerning X and Y, what final value is required?",
 "document": "documents/synth_000001.md"}
```

Paths in `document` are relative to `../input/`.

## Output

`predictions.jsonl` in **your own directory**, one JSON object per line:

```json
{"instance_id": "synth_000001", "answer": "140", "evidence": ["b14"]}
```

- `answer` — the final numerical value only. No unit, no currency symbol, no
  explanation, no "final answer:" prefix, unless the value itself requires a
  unit. Do not round beyond what the computation gives.
- `evidence` — a non-empty list of block identifiers copied verbatim from the
  document. Include every block directly needed to state the target question
  and its numeric inputs. Do not invent identifiers.
- All 300 instance ids must appear, each exactly once. Order does not matter.

## Scoring

Normalised exact match on `answer`; exact set match on `evidence`; the ranking
metric requires both. Citing a superset of the correct blocks scores the same
as citing the wrong block.

## Rules

- Solve from the supplied document. Do not infer answers from filenames,
  instance ids, or an external source-question lookup (e.g. matching a problem
  to a public dataset and reading its answer).
- `../reference/train-with-gold.jsonl` holds 908 training instances **with
  published answers and evidence**. This is a legitimate resource: use it to
  study the task, the document layout, and the annotation conventions, as you
  see fit. It is a different split from the one you are answering, and it was
  produced earlier; do not assume its layout carries over.
- You may write and run whatever code you want inside your own directory.
- Work only within `../input/`, `../reference/`, and your own directory.
  Anything outside those is not part of this task.

## Reporting

Also write `NOTES.md` in your directory: what approach you used, what you tried
and rejected, how long it took, and where you think your predictions are weak.
That file is part of the deliverable.
