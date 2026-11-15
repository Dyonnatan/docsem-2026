# Data

| Folder | Content | Source |
| --- | --- | --- |
| `replica/` | the test-like replica: `tasks.jsonl`, `labels.jsonl` and 300 markdown documents | built by `dfm replica build` from the training split |
| `docsem/train/` | official training tasks and labels (908) | DocSEM 2026, Task 1 |
| `docsem/val/`, `docsem/test/` | official validation (217) and test (1,730) tasks, without labels | DocSEM 2026, Task 1 |
| `extraction/train/`, `extraction/val/` | stage-1 markdown of every training and validation document | MinerU2.5-Pro, run once |
| `extraction/test.tar.xz` | stage-1 markdown of the 1,730 test documents | MinerU2.5-Pro, run once |
| `splits/` | instance ids of dev200 and hard-18 (both subsets of train) | this work |

Unpack the test extraction before running on the test split:

```bash
tar -xJf data/extraction/test.tar.xz -C data/extraction
```

## Stage-1 format

Each document is the markdown that MinerU2.5-Pro produced from the rendered PDF pages. Block
identifiers are kept as printed (`b07:` in train and validation; opaque ids such as
`RB2-5E9:` or `2032 ` in test and in the replica). Page breaks appear as `<<<PAGE n>>>`, and
layout comments with box coordinates (`<!-- bbox ... -->`) are not evidence ids. The prompts
tell the model both facts.

## The original PDFs

The PDFs are not redistributed here. They are published by the organisers on Hugging Face
(`amitbcp/docinsights-2026-shared-task-data`, mirror of `oracle-samples/gsm-sem`), and the
validation and test labels are held by the organisers' portal.

## License

The DocSEM data are released under the Universal Permissive License 1.0
([`LICENSE-UPL.txt`](LICENSE-UPL.txt)). The extractions and the replica are derived from them
and are distributed under the same terms. The problems are semantic variants of GSM8K
problems produced by GSM-SEM (Singh et al., 2026, arXiv:2605.07053).
