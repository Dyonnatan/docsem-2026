# Prompts

All prompts are reproduced verbatim as they were sent. The user message of every analyst
call is the instance id, the query and the stage-1 document (see `USER_TEMPLATE` in
`src/dfm/pipeline.py`).

| File | Used by |
| --- | --- |
| `a0.txt` | A0 on the replica |
| `a0-bnn.txt` | A0 on dev200 (its schema still shows `bNN` identifiers) |
| `c2.txt` | C2, all sets |
| `c3.txt` | C3 and C3C analyst, all sets; test attempt 1 |
| `c3-selection-card.txt` | C3 + selection card (decoy-only variant, the one reported) |
| `c3-selection-card-full.txt` | C3 + card with the outcome-record clause (Appendix E) |
| `repair-c2.txt`, `repair-c3.txt` | the single repair call when no reading survives |
| `c3c-reviewer.txt` | C3C reviewer |
| `notebook.json` | C3C notebook (replica) |
| `notebook-dev200.json` | C3C notebook as used on dev200 |
| `test-targeted-rerun.txt` | test attempt 2, targeted re-run of 113 tasks |
| `test-deliberation.txt` | test attempt 2, deliberation between the DeepSeek and Nemotron derivations |
| `test-r1-fallback.txt` | test attempt 2, DeepSeek-R1 whole-document fallback |

The selection card and the targeted re-run prompt were written after inspecting test
documents (the card after the deadline). Neither was used on dev200, and the card was never
part of a submitted system.
