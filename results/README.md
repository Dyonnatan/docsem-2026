# Model outputs

Each arm folder holds:

- `predictions.jsonl`: one line per completed task, `{instance_id, answer, evidence, decision}`.
  `decision` is the path that produced the answer (`unique_survivor`,
  `commit_among_survivors`, `notebook_rule`, `max_coverage_among_survivors`,
  `keep_model_commit`, `direct` for A0).
- `completions.jsonl.gz`: the text of every model completion that the answer depends on
  (`analyst`, `repair`, `reviewer`). Reasoning traces are not included.

`dfm replay <config>` rebuilds `predictions.jsonl` from `completions.jsonl.gz`.

| Folder | Arms |
| --- | --- |
| `replica/` | A0, C2, C3, C3 + card (DeepSeek-V4-Flash and GLM-5.3-Flash); C3 + full card (Appendix E); C3C (DeepSeek); A0 with Qwen3.7-Flash (Appendix D); the closed coding agent |
| `dev200/` | A0, C2, C3, C3C (DeepSeek-V4-Flash); C2 with GPT-5.6 Sol |
| `submissions/` | files submitted to the validation and test portals |

## Portal scores of the submitted files

| File | When | Answer | Joint |
| --- | --- | ---: | ---: |
| `test-attempt1.jsonl` | competition | 51.4 | 34.62 |
| `test-attempt2.jsonl` | competition | withheld | withheld |
| `test-attempt3.jsonl` | competition, ranked | not published | **45.09** |
| `val-final.jsonl` | competition | 89.86 | 89.86 |
| `val-c3-glm.jsonl` | after the competition | 88.94 | 74.65 |
| `val-c3-card-glm.jsonl` | after the competition | 89.40 | 88.94 |

The final standings publish joint accuracy only. The best entry reached 85.09.
