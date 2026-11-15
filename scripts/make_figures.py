"""Draw the figures used in the README and on the project page.

    python scripts/make_figures.py

Every number is computed from results/ through dfm.tables, so the figures
cannot drift from the released predictions.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dfm import tables  # noqa: E402

OUT = ROOT / "docs" / "assets"
FONT = "Inter, 'Helvetica Neue', Arial, sans-serif"
MONO = "'JetBrains Mono', Menlo, Consolas, monospace"

BLUE, DARK, LIGHT = "#40739e", "#273c75", "#e8f4ff"
SEL, RES = "#eb6834", "#4a3aa7"
NEUTRAL, PALE, GRID, AXIS = "#8f8c84", "#d9d6cd", "#e6e4dd", "#9a978e"
INK, INK2 = "#16181d", "#55534e"


def _svg(w: int, h: int, body: str, title: str) -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
            f'role="img" aria-label="{title}" font-family="{FONT}">\n<title>{title}</title>\n'
            f'<rect width="{w}" height="{h}" fill="#ffffff"/>\n{body}\n</svg>\n')


def replica_chart() -> str:
    rows = {r["arm"]: r for r in tables.replica_table()["rows"]}
    w, h = 760, 360
    left, right, top, bottom = 56, 24, 40, 300
    scale = lambda v: bottom - (bottom - top) * v / 100  # noqa: E731
    parts = []
    for v in range(0, 101, 20):
        y = scale(v)
        parts.append(f'<line x1="{left}" x2="{w - right}" y1="{y:.1f}" y2="{y:.1f}" stroke="{GRID}"/>')
        parts.append(f'<text x="{left - 10}" y="{y + 4:.1f}" font-size="12" fill="{AXIS}" text-anchor="end">{v}</text>')
    parts.append(f'<text x="16" y="{(top + bottom) / 2}" font-size="12" fill="{INK2}" '
                 f'transform="rotate(-90 16 {(top + bottom) / 2})" text-anchor="middle">answer accuracy (%)</text>')

    groups = [("DeepSeek-V4-Flash", "DeepSeek"), ("GLM-5.3-Flash", "GLM")]
    variants = [("A0", "A0"), ("C2", "C2"), ("C3", "C3"), ("C3+card", "C3 + card")]
    gw = (w - left - right) / 2
    bw = 58
    for gi, (gname, key) in enumerate(groups):
        x0 = left + gi * gw + (gw - 4 * bw - 3 * 14) / 2
        for vi, (vk, vlabel) in enumerate(variants):
            r = rows[f"{key} {vk}"]
            x = x0 + vi * (bw + 14)
            y = scale(r["answer"])
            color = SEL if "card" in vk else (NEUTRAL if vk != "A0" else PALE)
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw}" height="{bottom - y:.1f}" rx="3" fill="{color}"/>')
            lo, hi = r["answer_ci95"]
            cx = x + bw / 2
            parts.append(f'<line x1="{cx:.1f}" x2="{cx:.1f}" y1="{scale(hi):.1f}" y2="{scale(lo):.1f}" '
                         f'stroke="{INK}" stroke-width="1.2" opacity="0.55"/>')
            parts.append(f'<text x="{cx:.1f}" y="{scale(hi) - 7:.1f}" font-size="13" font-weight="700" '
                         f'fill="{INK}" text-anchor="middle">{r["answer"]:.1f}</text>')
            parts.append(f'<text x="{cx:.1f}" y="{bottom + 18}" font-size="12.5" fill="{INK2}" '
                         f'text-anchor="middle">{vlabel}</text>')
        parts.append(f'<text x="{left + gi * gw + gw / 2:.1f}" y="{bottom + 42}" font-size="14" font-weight="700" '
                     f'fill="{INK}" text-anchor="middle">{gname}</text>')

    agent = rows["Claude Code (Sonnet 5)"]["answer"]
    y = scale(agent)
    parts.append(f'<line x1="{left}" x2="{w - right}" y1="{y:.1f}" y2="{y:.1f}" stroke="{DARK}" '
                 f'stroke-width="1.5" stroke-dasharray="6 5"/>')
    parts.append(f'<text x="{left + 8}" y="{y - 8:.1f}" font-size="12.5" fill="{DARK}" '
                 f'font-style="italic">closed coding agent (Claude Code, Sonnet 5): {agent:.1f}</text>')
    parts.append(f'<line x1="{left}" x2="{w - right}" y1="{bottom}" y2="{bottom}" stroke="{AXIS}"/>')
    return _svg(w, h, "\n".join(parts), "Replica answer accuracy by variant and model")


def decomposition_chart() -> str:
    d = tables.decomposition()["C3C, all tasks (paper)"]
    n = d["n"]
    sel, res = d["selection_errors"][0], d["resolution_errors"][0]
    ok = d["counts"]["correct"]
    segs = [("selection error", sel, SEL, "#ffffff"), ("resolution error", res, RES, "#ffffff"),
            ("correct", ok, PALE, INK)]
    w, h = 760, 160
    left, width, y, bh = 24, 712, 44, 46
    parts, x = [], left
    for label, k, color, tc in segs:
        sw = width * k / n
        parts.append(f'<rect x="{x:.1f}" y="{y}" width="{sw:.1f}" height="{bh}" fill="{color}"/>')
        if sw > 60:
            parts.append(f'<text x="{x + 12:.1f}" y="{y + 29}" font-size="15" font-weight="700" fill="{tc}">'
                         f'{100 * k / n:.1f}%</text>')
        else:
            parts.append(f'<text x="{x + sw / 2:.1f}" y="{y - 6}" font-size="13" font-weight="700" fill="{color}" '
                         f'text-anchor="middle">{100 * k / n:.1f}%</text>')
        x += sw
    xs = left
    for label, k, color, _ in segs:
        parts.append(f'<rect x="{xs}" y="{y + bh + 22}" width="12" height="12" rx="2" fill="{color}"/>')
        parts.append(f'<text x="{xs + 18}" y="{y + bh + 32}" font-size="13" fill="{INK2}">{label} ({k})</text>')
        xs += 210
    parts.append(f'<text x="{left}" y="18" font-size="13" fill="{INK2}">Outcomes on the replica, '
                 f'DeepSeek-V4-Flash, n = {n}</text>')
    return _svg(w, h, "\n".join(parts), "Selection versus resolution errors on the replica")


def pipeline_diagram() -> str:
    w, h = 760, 560
    parts = []

    def box(x, y, bw, bh, title, sub, kind):
        fill, stroke, tc, sc = {
            "model": (DARK, DARK, "#ffffff", "#c9d6ec"),
            "rule": ("#ffffff", "#8e8c86", INK, INK2),
            "out": (LIGHT, BLUE, DARK, INK2),
        }[kind]
        parts.append(f'<rect x="{x}" y="{y}" width="{bw}" height="{bh}" rx="8" fill="{fill}" stroke="{stroke}" '
                     f'stroke-width="1.4"/>')
        parts.append(f'<text x="{x + bw / 2}" y="{y + (24 if sub else bh / 2 + 5)}" font-size="15" font-weight="700" '
                     f'fill="{tc}" text-anchor="middle">{title}</text>')
        if sub:
            parts.append(f'<text x="{x + bw / 2}" y="{y + 44}" font-size="12.5" fill="{sc}" '
                         f'text-anchor="middle">{sub}</text>')

    def arrow(x1, y1, x2, y2):
        parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2 - 7}" stroke="#6b6964" stroke-width="1.6"/>')
        parts.append(f'<path d="M{x2 - 5},{y2 - 8} L{x2 + 5},{y2 - 8} L{x2},{y2} Z" fill="#6b6964"/>')

    cx, bw = 380, 440
    x = cx - bw / 2
    box(x, 16, bw, 56, "Page transcription (VLM, once per document)", "MinerU2.5-Pro: rendered PDF → markdown with block ids", "model")
    arrow(cx, 72, cx, 96)
    box(x, 96, bw, 56, "Analyst (LLM)", "query + document → two rival readings as DSL steps", "model")
    arrow(cx, 152, cx, 176)
    box(x, 176, bw, 56, "Compile and evaluate", "steps → expression → value (Python, no model)", "rule")
    arrow(cx, 232, cx, 256)
    box(x, 256, bw, 56, "Validate", "V5 evaluates · V4 tiered rates · V2 relative coverage", "rule")
    # fork
    lx, rx = 190, 570
    parts.append(f'<path d="M{cx},312 V330 M{lx},330 H{rx}" fill="none" stroke="#6b6964" stroke-width="1.6"/>')
    arrow(lx, 330, lx, 356)
    arrow(rx, 330, rx, 356)
    parts.append(f'<text x="{lx + 8}" y="346" font-size="12" font-style="italic" fill="{INK2}">one survives</text>')
    parts.append(f'<text x="{rx + 8}" y="346" font-size="12" font-style="italic" fill="{INK2}">several survive (C3C)</text>')
    box(lx - 140, 356, 280, 56, "Commit the survivor", "no further model call", "rule")
    box(rx - 160, 356, 320, 56, "Reviewer (LLM)", "labels each reading: schema + side", "model")
    arrow(rx, 412, rx, 434)
    box(rx - 160, 434, 320, 56, "Decide", "notebook rule if labels separate; else analyst's choice", "rule")
    parts.append(f'<path d="M{lx},412 V508 H{rx} M{rx},490 V508" fill="none" stroke="#6b6964" stroke-width="1.6"/>')
    arrow(cx, 508, cx, 524)
    box(cx - 150, 524, 300, 32, "numeric answer + block id", "", "out")
    legend_y = 16
    for i, (label, fill, stroke) in enumerate([("model call", DARK, DARK), ("deterministic code", "#ffffff", "#8e8c86")]):
        parts.append(f'<rect x="{620}" y="{legend_y + 22 * i + 100}" width="16" height="12" rx="2" fill="{fill}" stroke="{stroke}"/>')
        parts.append(f'<text x="{642}" y="{legend_y + 22 * i + 110}" font-size="12" fill="{INK2}">{label}</text>')
    return _svg(w, h, "\n".join(parts), "The C3C pipeline")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in [("replica-results.svg", replica_chart), ("error-decomposition.svg", decomposition_chart),
                     ("pipeline.svg", pipeline_diagram)]:
        (OUT / name).write_text(fn(), encoding="utf-8")
        print("wrote", OUT / name)


if __name__ == "__main__":
    main()
