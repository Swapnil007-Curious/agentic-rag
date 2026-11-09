"""Synthetic PDF fixtures built with PyMuPDF (no real corpus files needed).

* ``build_act_pdf``: Act-like document with a running header, page-number footers, an
  ARRANGEMENT OF SECTIONS block, chapters, bold inline headings, a marginal heading,
  sub-sections, clauses, sub-clauses, an Explanation, a proviso, a superscript footnote
  marker + footnote, a hyphenated line break, a mixed English/Devanagari line, a
  Devanagari-dominant line and a wholly Devanagari page.
* ``build_rules_pdf``: Rules-like Gazette notification that refers to "section 3 of the Act",
  "section 101 of the Act" (which does not exist) and "sections 3 to 5 of the Act".
* ``build_report_pdf``: non-legal document whose headings differ only by font size/weight.
"""

from __future__ import annotations

from pathlib import Path

import pymupdf

PAGE_W, PAGE_H = 595, 842
BODY = 10.0
REG, BOLD = "tiro", "tibo"  # Times-Roman / Times-Bold (PDF base-14)
# Base-14 fonts only encode Latin-1, so dashes are written as "--" and quotes as '"'
# (the cleaner normalises "--" to an em dash, as in the Gazette copies).


class _Page:
    def __init__(self, doc: pymupdf.Document, number: int, header: str | None) -> None:
        self.page = doc.new_page(width=PAGE_W, height=PAGE_H)
        self.y = 90.0
        if header:
            self.page.insert_text((200, 40), header, fontname=REG, fontsize=9)
        self.page.insert_text((PAGE_W / 2 - 3, 815), str(number), fontname=REG, fontsize=9)

    def line(
        self, x: float, parts: list[tuple[str, str]], size: float = BODY, gap: float = 14.0
    ) -> None:
        """Write one visual line made of (text, font) runs."""
        cx = x
        for text, font in parts:
            self.page.insert_text((cx, self.y), text, fontname=font, fontsize=size)
            cx += pymupdf.get_text_length(text, fontname=font, fontsize=size)
        self.y += gap

    def para_gap(self, extra: float = 7.0) -> None:
        self.y += extra

    def superscript(self, x_after: str, x: float, mark: str) -> None:
        """Small raised marker placed right after the text ``x_after`` written at ``x``."""
        cx = x + pymupdf.get_text_length(x_after, fontname=REG, fontsize=BODY)
        self.page.insert_text((cx, self.y - 14.0 - 3.5), mark, fontname=REG, fontsize=6)

    def margin_note(self, x: float, text: str) -> None:
        """Small italic note in the page margin, level with the next body line."""
        self.page.insert_text((x, self.y), text, fontname="tiit", fontsize=8)

    def html(self, x: float, html: str, height: float = 16.0) -> None:
        rect = pymupdf.Rect(x, self.y - 11, PAGE_W - 60, self.y - 11 + height)
        self.page.insert_htmlbox(rect, f'<p style="font-size:10pt;margin:0">{html}</p>')
        self.y += height + 4


ACT_HEADER = "THE TEST PROTECTION ACT, 2024"


def build_act_pdf(path: Path) -> Path:
    doc = pymupdf.open()
    # ---- page 1: arrangement of sections (table of contents)
    p = _Page(doc, 1, ACT_HEADER)
    p.line(230, [("ARRANGEMENT OF SECTIONS", REG)])
    p.para_gap()
    p.line(270, [("CHAPTER I", REG)])
    p.line(260, [("PRELIMINARY", REG)])
    p.line(72, [("SECTIONS", REG)])
    p.line(90, [("1. Short title and commencement.", REG)])
    p.line(90, [("2. Definitions.", REG)])
    p.line(270, [("CHAPTER II", REG)])
    p.line(262, [("COMPLAINTS", REG)])
    p.line(90, [("3. Filing of complaint.", REG)])
    p.line(90, [("4. Appeal.", REG)])
    p.line(
        90, [("5. Powers of the authority to issue guidelines in respect of complaints and", REG)]
    )
    p.line(108, [("appeals.", REG)])

    # ---- page 2: chapter I
    p = _Page(doc, 2, ACT_HEADER)
    p.line(72, [("An Act to provide for the protection of test consumers.", REG)])
    p.para_gap()
    p.line(270, [("CHAPTER I", REG)])
    p.para_gap()
    p.line(260, [("PRELIMINARY", REG)])
    p.para_gap()
    p.line(
        90,
        [
            ("1. Short title and commencement.", BOLD),
            ("--(1) This Act may be called the Test Protection Act, 2024.", REG),
        ],
    )
    p.para_gap()
    p.line(90, [("(2) It shall come into force on such date as the Government may appoint.", REG)])
    p.superscript("(2) It shall come into force on such date", 90, "1")
    p.para_gap()
    p.line(
        90,
        [
            ("2. Definitions.", BOLD),
            ("--In this Act, unless the context otherwise requires,--", REG),
        ],
    )
    p.para_gap()
    p.line(
        108,
        [
            (
                '(1) "consumer" means any person who buys goods for consideration and includes a regis-',
                REG,
            )
        ],
    )
    p.line(90, [("tered user;", REG)])
    p.para_gap()
    p.line(108, [('(2) "goods" means every kind of movable property;', REG)])
    p.para_gap()
    p.line(108, [('(3) "trader" means a person who sells goods and includes--', REG)])
    p.para_gap()
    p.line(126, [("(i) a manufacturer; and", REG)])
    p.para_gap()
    p.line(126, [("(ii) a packer;", REG)])
    p.para_gap()
    p.line(108, [('(4) "Government" means the Government of Testland.', REG)])
    # footnote at the bottom of the page (smaller font)
    p.page.insert_text(
        (72, 770),
        "1. 1st January, 2025, vide Notification No. S.O. 1(E).",
        fontname=REG,
        fontsize=7.5,
    )

    # ---- page 3: chapter II
    p = _Page(doc, 3, ACT_HEADER)
    p.line(270, [("CHAPTER II", REG)])
    p.para_gap()
    p.line(262, [("COMPLAINTS", REG)])
    p.para_gap()
    p.line(
        90,
        [
            ("3. Filing of complaint.", BOLD),
            ("--(1) A complaint may be filed by a consumer referred to in clause (1) of", REG),
        ],
    )
    p.line(72, [("section 2 in the following manner, namely:--", REG)])
    p.para_gap()
    p.line(108, [("(a) in writing; or", REG)])
    p.para_gap()
    p.line(108, [("(b) electronically:", REG)])
    p.para_gap()
    p.line(
        90, [("Provided that no fee shall be charged for a complaint below one lakh rupees.", REG)]
    )
    p.para_gap()
    p.line(
        90,
        [
            (
                "Explanation.--For the purposes of this sub-section, a complaint includes a counter-claim.",
                REG,
            )
        ],
    )
    p.para_gap()
    p.line(
        90,
        [
            (
                "(2) Every complaint shall be accompanied by the fee referred to in clause (a) of sub-section (1).",
                REG,
            )
        ],
    )
    p.para_gap()
    # Gazette-style marginal heading: "Appeal." printed in the left margin beside "4. (1) ..."
    p.margin_note(14, "Appeal.")
    p.line(
        90, [("4. (1) Any person aggrieved by an order made under sections 1 to 3 may appeal", REG)]
    )
    p.line(72, [("within thirty days.", REG)])
    p.para_gap()
    p.line(
        90,
        [("(2) The provisions of sub-section (1) of section 3 shall apply to every appeal.", REG)],
    )
    p.para_gap()
    p.line(
        90, [("5. Powers of the authority to issue guidelines in respect of complaints and", BOLD)]
    )
    p.line(72, [("appeals.", BOLD), ("--The authority may issue guidelines under this Act.", REG)])
    p.para_gap()
    p.html(72, "The authority shall publish an annual report every year in English भारत")
    p.html(72, "भारत सरकार कमल नमक घर report")

    # ---- page 4: Hindi version (dropped as a whole page)
    p = _Page(doc, 4, ACT_HEADER)
    for _ in range(6):
        p.html(72, "भारत सरकार कमल नमक घर सरकार कमल")
    doc.save(path)
    doc.close()
    return path


def build_rules_pdf(path: Path) -> Path:
    doc = pymupdf.open()
    p = _Page(doc, 1, None)
    p.line(72, [("PUBLISHED BY AUTHORITY", BOLD)])
    p.para_gap()
    p.line(200, [("MINISTRY OF TESTING", BOLD)])
    p.line(240, [("NOTIFICATION", BOLD)])
    p.para_gap()
    p.line(
        90,
        [
            (
                "G.S.R. 1(E).--In exercise of the powers conferred by section 5 of the Test Protection Act, 2024,",
                REG,
            )
        ],
    )
    p.line(72, [("the Government hereby makes the following rules, namely:--", REG)])
    p.para_gap()
    p.line(72, [("1. ", REG)])
    p.line(
        100,
        [("Short title.", BOLD), ("--These rules may be called the Test Rules, 2024.", REG)],
        gap=0,
    )
    p.y += 14
    p.para_gap()
    p.line(
        72,
        [
            ("2. Definitions.", BOLD),
            ("--(1) In these rules, unless the context otherwise requires,--", REG),
        ],
    )
    p.para_gap()
    p.line(100, [('(a) "Act" means the Test Protection Act, 2024;', REG)])
    p.para_gap()
    p.line(100, [('(b) "entity" means a trader registered under rule 3.', REG)])
    p.para_gap()
    p.line(
        72,
        [
            ("3. Duties of entities.", BOLD),
            ("--(1) Every entity shall comply with section 101 of the Act.", REG),
        ],
    )
    p.para_gap()
    p.line(
        72,
        [
            (
                "(2) A complaint under sub-section (1) of section 3 of the Act shall be in writing.",
                REG,
            )
        ],
    )
    p.para_gap()
    p.line(72, [("(3) Nothing in sub-rule (1) affects sections 3 to 5 of the Act.", REG)])
    p.para_gap()
    p.line(420, [("[F. No. 1/2024-TEST]", REG)])
    p.line(420, [("A. OFFICER, Jt. Secy.", REG)])
    doc.save(path)
    doc.close()
    return path


def build_report_pdf(path: Path) -> Path:
    doc = pymupdf.open()
    page = doc.new_page(width=PAGE_W, height=PAGE_H)
    y = 80.0

    def put(text: str, size: float, font: str = "helv", gap: float = 18.0) -> None:
        nonlocal y
        page.insert_text((72, y), text, fontname=font, fontsize=size)
        y += gap

    put("Annual Garden Report", 22, "hebo", 34)
    put("Introduction", 16, "hebo", 24)
    put("This report describes how the community garden was planted and maintained", 11)
    put("during the year, including the volunteers who helped every weekend.", 11, gap=26)
    put("Methods", 16, "hebo", 24)
    put("Beds were dug in spring and enriched with compost from the previous season.", 11, gap=26)
    put("Watering Schedule", 13, "hebo", 22)
    put("Plants were watered every morning before nine and again on hot evenings.", 11, gap=26)
    put("Results", 16, "hebo", 24)
    put("The garden produced tomatoes, beans and pumpkins for the local food bank.", 11)
    doc.save(path)
    doc.close()
    return path
