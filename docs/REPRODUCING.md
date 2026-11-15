# Reproducing the results

This note says which configuration produced each released output, what can be rebuilt
without calling a model, and where a number recomputed from the released files differs
from the paper.

## Three levels of reproduction

1. **Exact, offline.** `dfm replay` rebuilds every released prediction from the released
   completions with the deterministic part of the pipeline (DSL compilation, validators,
   notebook closing). `dfm tables` recomputes the paper's tables from those predictions.
   `dfm replica build` regenerates the replica from the training extractions. All three are
   covered by `pytest`.
2. **Statistical, with API calls.** `dfm run` repeats an arm with the same prompt, model,
   sampling and routing. All calls sample at temperature 1.0, so new runs differ from ours
   item by item; with 300 tasks the 95% interval of an accuracy is about ±5 points.
3. **Not reproducible here.** Stage 1 (MinerU2.5-Pro transcription) ran once on one RTX 3090;
   its output is released instead of the PDFs. The test-portal scores can only be obtained
   from the organisers' portal; the submitted files are in `results/submissions/`.

## What produced each released output

| Output | Configuration | Notes |
| --- | --- | --- |
| `replica/a0-*`, `replica/c2-*`, `replica/c3-*` | `configs/replica/*.json` | 26 C2 DeepSeek tasks hit a provider rate limit and were rerun with the same configuration on another fp8 provider |
| `replica/c3-card-*` | `configs/replica/c3-card-*.json` | the decoy-only card reported in the paper |
| `replica/c3-card-full-*` | `configs/replica/c3-card-full-*.json` | Appendix E variant; its extra clause is circular on the replica |
| `replica/c3c-deepseek` | `configs/replica/c3c-deepseek.json` | reviewer applied to the C3 DeepSeek completions; 31 of 300 reviewer calls failed or timed out (recorded as `null`), and the C3 answer stands in those tasks |
| `replica/claude-code-sonnet5` | `TASK.md` in that folder | Claude Code with Claude Sonnet 5 (medium effort), run once in an isolated directory; predictions only |
| `dev200/a0`, `c2`, `c3` | `configs/dev200/*.json` | the A0 prompt of this phase still showed `bNN` identifiers (`prompts/a0-bnn.txt`); C3 used an 8,192-token ceiling |
| `dev200/c3c-deepseek` | `configs/dev200/c3c-deepseek.json` | ran with `notebook-dev200.json` (no `single_axis_ok` switch yet) and with reviewer additions enabled |
| `submissions/test-attempt1.jsonl` | C3C on DeepSeek-V4-Flash, one sample | portal joint accuracy 34.62 |
| `submissions/test-attempt2.jsonl` | attempt 1 plus the correction cascade of Appendix B | portal score withheld |
| `submissions/test-attempt3.jsonl` | attempt 2 plus evidence narrowing | portal joint accuracy 45.09, the ranked entry |
| `submissions/val-final.jsonl` | C3C with self-consistency, escalation, deliberation and a separate evidence call | portal 89.86 answer / 89.86 joint |
| `submissions/val-c3-glm.jsonl`, `val-c3-card-glm.jsonl` | C3 and C3 + card on GLM-5.3-Flash, after the competition | portal 88.94 / 74.65 and 89.40 / 88.94 |

The correction cascade of the test submission (targeted re-run, Nemotron-3-Ultra escalation,
deliberation, DeepSeek-R1 fallback) is documented by its prompts
(`prompts/test-*.txt`), by the evidence post-processing in `dfm/blocks.py`, and by the three
submitted files. Its orchestration scripts are not part of this release; the paper reports
that the contribution of each stage cannot be isolated.

## Known behaviour of the released pipeline

- **V2 and V4 on opaque identifiers.** The validators read the evidence text through the
  `b01:` block format of the train and validation splits. On documents with opaque
  identifiers (test and replica) that text is empty, so only V5 can reject a reading there.
  This is how every reported run behaved and it is kept unchanged.
- **C3 adds `base * fraction` variants.** The completion of base-times-fraction readings
  is part of the C3 compiler, so C3 runs include those readings too. They can change the
  answer only when the model's preferred reading does not survive validation.
- **Fallback answers.** When no reading survives validation and the repair call also fails,
  C2 and C3 keep the answer the model stated in its prediction. C3C starts from the compiled
  C3 state, where that fallback does not exist; this is why C3C can have fewer completed
  items than C3.

## Versions of the paper

The LaTeX source in `paper/` is the revised version of November 2026. It corrects the
following figures of the camera-ready version of 27 September 2026, which the poster and
the recorded talk presented at the workshop also showed. The poster and the slides in
`docs/` were regenerated from the revised paper. None of the corrections changes the
paper's conclusions.

| Camera-ready and workshop material | Revised paper and this repository | Explanation |
| --- | --- | --- |
| The replica matches the first test answer accuracy "within one point" (50.5 against 51.4); errors of C3C on 291 tasks: 52.9% selection, 7.6% resolution | C3C on all 300 tasks: 49.0 against 51.4 (2.4 points); 54.0% selection, 7.3% resolution | the earlier numbers were those of the **C3** DeepSeek outputs before 9 parse failures were re-run. `dfm tables` prints both decompositions |
| A decoy is the cited block in 81 of the 154 selection errors | 88 of the 162 selection errors | the earlier count inspected only the selection errors with a wrong number |
| dev200: C3 against C2 is 3:14 on paired items, p = 0.013 | 3:11 on the 194 items both arms completed, p = 0.057 (4:14, p = 0.031, counting missing answers as wrong) | the 3:14 contrast could not be traced to the archived outputs. C3 = 163 is the compiled answer; with the fallback to the stated answer C3 scores 165 |
| Validity gate: keyword 23.4%, longest 32.0%, first 30.6%, random 31.3% (n = 278) | keyword 27.9%, longest 30.1%, first 29.7%, random 29.6% (n = 276) | the original gate script was not preserved; `dfm replica gate` documents its candidate definition. Keyword matching still does not beat chance |
| The val score 88.48 / 88.02 is marked as self-consistency with N = 5 | it is a single sample per task; only the 89.86 submission used N = 5 | the first val submission was scored before the four extra samples were run |
| Replica table: 300 items per arm except 295 for A0 GLM | also 299 for C2 GLM | one task of that arm has no answer |

The revised paper adds a table to Section 2, so the dev200 results are now in Table 4, the
replica results in Table 5 and the portal scores in Table 6. The hard-18 column of Table 4
comes from earlier exploratory runs that are not part of this release; the 18 identifiers
are in `data/splits/hard18.json`.
