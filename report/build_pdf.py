"""Best-effort Markdown -> PDF renderer for the report (no pandoc/LaTeX needed).

Uses ``fpdf2`` to produce a plain but readable PDF: headings, paragraphs,
bullets, monospaced tables/code, and embedded figures.  This is intentionally
simple -- for a polished PDF use pandoc if available.  Run::

    python report/build_pdf.py
"""

from __future__ import annotations

import re
from pathlib import Path

from fpdf import FPDF

HERE = Path(__file__).resolve().parent
MD = HERE / "equity_signal_research_report.md"
PDF = HERE / "equity_signal_research_report.pdf"


def _clean(text: str) -> str:
    # Strip markdown emphasis and links -> plain text; normalize unicode dashes.
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"`(.+?)`", r"\1", text)
    repl = {"–": "-", "—": "-", "≈": "~", "×": "x", "→": "->", "₀": "0",
            "“": '"', "”": '"', "’": "'", "≤": "<=", "≥": ">=", "·": "."}
    for k, v in repl.items():
        text = text.replace(k, v)
    return text.encode("latin-1", "replace").decode("latin-1")


def build() -> Path:
    lines = MD.read_text().splitlines()
    pdf = FPDF(format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_margins(18, 16, 18)

    def mono(line: str, size: float = 7.0) -> None:
        pdf.set_font("Courier", "", size)
        pdf.set_x(pdf.l_margin)
        # Hard-wrap long monospace lines so a single token never overflows.
        width = max(20, int((pdf.w - pdf.l_margin - pdf.r_margin) / (size * 0.6 / 2.83)))
        for i in range(0, max(1, len(line)), width):
            pdf.multi_cell(0, 4.0, _clean(line[i:i + width]))

    def para(text: str, size: float, style: str = "", h: float = 5.0,
             prefix: str = "") -> None:
        pdf.set_font("Helvetica", style, size)
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(0, h, _clean(prefix + text))

    in_front = False
    in_code = False
    in_table = False
    for raw in lines:
        line = raw.rstrip()
        try:
            if line.strip() == "---" and not in_code:
                in_front = not in_front
                continue
            if in_front:
                m = re.match(r'(title|author|date):\s*"?(.+?)"?$', line)
                if m and m.group(1) == "title":
                    para(m.group(2), 16, "B", 8)
                    pdf.ln(2)
                continue

            if line.startswith("```"):
                in_code = not in_code
                continue
            if in_code:
                mono(line)
                continue

            img = re.match(r"!\[(.*?)\]\((.+?)\)", line)
            if img:
                cap, rel = img.group(1), img.group(2)
                path = (HERE / rel).resolve()
                if path.exists():
                    pdf.ln(2)
                    pdf.image(str(path), w=160)
                    para(cap, 8, "I", 4)
                    pdf.ln(2)
                continue

            if line.startswith("|"):
                in_table = True
                mono(line)
                continue
            elif in_table and not line.startswith("|"):
                in_table = False
                pdf.ln(1)

            if line.startswith("# "):
                pdf.ln(2); para(line[2:], 15, "B", 7); pdf.ln(1)
            elif line.startswith("## "):
                pdf.ln(2); para(line[3:], 12, "B", 6); pdf.ln(0.5)
            elif line.startswith("### "):
                para(line[4:], 10, "B", 5.5)
            elif line.startswith("- "):
                para(line[2:], 10, "", 5, prefix="  - ")
            elif line.strip() == "":
                pdf.ln(2)
            else:
                para(line, 10, "", 5)
        except Exception:
            continue

    pdf.output(str(PDF))
    return PDF


if __name__ == "__main__":
    out = build()
    print(f"wrote {out} ({out.stat().st_size/1024:.0f} KB)")
