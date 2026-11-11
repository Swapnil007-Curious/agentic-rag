"""Synthetic textbook PDF (PyMuPDF base-14 fonts; no real books needed).

Pages (A4): 1 title, 2 licence, 3 contents (front matter); 4-6 chapter 1 "Basics"; 7 chapter 2
"Graphs"; 8 chapter 3 "Extras" (not selected); 9 index. Printed page numbers are the PDF page
minus 3 and appear in running headers ("2 The Testing Book" / "Basics 3") and in centred
footers on chapter-opening pages. Chapter 1 holds numbered headings (1, 1.1, 1.1.1, 1.2),
unnumbered headings ("Amortized Cost", "Summary"), a Courier code block with indentation, an
equation line with its number and a superscript ("O(n^2)"), cross-references ("Section 1.2",
"Fig. 1.1", "Chapter 2", "[3, Section 2.1]", "Section 1.1.7") and an Exercises section.
Chapter 2 refers to "Section 1.2" and "Section 9.4", repeats the heading "Summary", holds an
"Exercise 2.1." paragraph and a References section. ``with_outline=False`` builds the same PDF
without bookmarks (chapter selection then falls back to heading detection).
"""

from __future__ import annotations

from pathlib import Path

import pymupdf

PAGE_W, PAGE_H = 595, 842
REG, BOLD, MONO = "tiro", "tibo", "cour"
BODY = 10.0
OFFSET = 3  # PDF page - printed page


class _P:
    def __init__(
        self, doc: pymupdf.Document, number: int, header: str | None, footer: bool
    ) -> None:
        self.page = doc.new_page(width=PAGE_W, height=PAGE_H)
        self.y = 90.0
        printed = number - OFFSET
        if header == "even":
            self.text(72, 40, f"{printed} The Testing Book", REG, 9)
        elif header == "odd":
            self.text(72, 40, "Basics", REG, 9)
            self.text(500, 40, str(printed), REG, 9)
        if footer:
            self.text(PAGE_W / 2 - 3, 815, str(printed), REG, 9)

    def text(self, x: float, y: float, s: str, font: str = REG, size: float = BODY) -> None:
        self.page.insert_text((x, y), s, fontname=font, fontsize=size)

    def line(
        self, s: str, font: str = REG, size: float = BODY, x: float = 72, gap: float = 14
    ) -> None:
        self.text(x, self.y, s, font, size)
        self.y += gap

    def heading(self, s: str, size: float) -> None:
        self.y += 10
        self.line(s, BOLD, size, gap=size + 10)

    def para(self, lines: list[str]) -> None:
        for s in lines:
            self.line(s)
        self.y += 8

    def code(self, lines: list[str]) -> None:
        cw = pymupdf.get_text_length("x", fontname=MONO, fontsize=9)
        for s in lines:
            indent = len(s) - len(s.lstrip(" "))
            self.text(72 + indent * cw, self.y, s.lstrip(" "), MONO, 9)
            self.y += 11
        self.y += 8


def build_textbook_pdf(path: Path, with_outline: bool = True) -> Path:
    doc = pymupdf.open()
    # 1 title page
    p = _P(doc, 1, None, False)
    p.y = 200
    p.line("A Small Book of Testing", BOLD, 24, gap=40)
    p.line("Ann Author and Bo Writer")
    # 2 licence page
    p = _P(doc, 2, None, False)
    p.line("License", BOLD, 16, gap=26)
    p.line("This work is licensed under the Creative Commons Attribution-ShareAlike 4.0")
    p.line("International License. To view a copy of this license, visit creativecommons.org.")
    # 3 contents
    p = _P(doc, 3, None, False)
    p.line("Contents", BOLD, 16, gap=26)
    for s in ("1 Basics . . . . . 1", "2 Graphs . . . . . 4", "3 Extras . . . . . 5"):
        p.line(s)
    # 4 chapter 1 opens
    p = _P(doc, 4, None, True)
    p.heading("1 Basics", 20)
    p.para(
        [
            "This chapter introduces lists, arrays and the cost of growing them. It is the",
            "opening chapter of the book and explains how the later material is organised.",
        ]
    )
    p.heading("1.1 Lists and Arrays", 14)
    p.para(
        [
            "An array stores elements in consecutive memory cells. We call this layout the",
            "backing store; see Section 1.2 for linked layouts and Fig. 1.1 for a picture.",
        ]
    )
    p.code(["def grow(a):", "    b = new_array(2 * len(a))", "    return b"])
    p.text(250, p.y, "y = W x + b")
    p.text(480, p.y, "(1.1)")
    p.y += 22
    p.text(72, p.y, "Doubling keeps the running time at O(n")
    w = pymupdf.get_text_length(
        "Doubling keeps the running time at O(n", fontname=REG, fontsize=BODY
    )
    p.text(72 + w, p.y - 3.5, "2", REG, 6)
    p.text(72 + w + 4, p.y, ") over all operations in the worst case.")
    p.y += 22
    # 5 chapter 1 continues
    p = _P(doc, 5, "even", False)
    p.heading("1.1.1 Growing Arrays", 11.5)
    p.para(
        [
            "Growing an array copies every element once. This is explained in Chapter 2 and",
            "in [3, Section 2.1] of another book, while Section 1.1.7 does not exist here.",
        ]
    )
    p.heading("Amortized Cost", 11)
    p.para(["The amortized cost of one addition is constant when the array doubles in size."])
    p.heading("1.2 Linked Structures", 14)
    p.para(
        ["A linked list stores each element in a node that points to the next node of the list."]
    )
    p.heading("Summary", 11)
    p.para(["Arrays give constant-time access; linked lists give constant-time insertion."])
    # 6 exercises
    p = _P(doc, 6, "odd", False)
    p.heading("1.3 Exercises", 14)
    p.para(
        ["1. Show that doubling the array gives linear total copying cost.", "2. Prove it again."]
    )
    # 7 chapter 2
    p = _P(doc, 7, None, True)
    p.heading("2 Graphs", 20)
    p.para(["Graphs connect vertices with edges. This chapter covers traversal of graphs."])
    p.heading("2.1 Traversal", 14)
    p.para(
        [
            "Breadth-first search visits vertices in order of distance. It uses the lists of",
            "Section 1.2 and the queue of Section 9.4 from a chapter that is not included.",
        ]
    )
    p.para(["Exercise 2.1. Prove that breadth-first search visits every reachable vertex."])
    p.heading("Summary", 11)
    p.para(["Traversal visits every vertex that can be reached from the start vertex."])
    p.heading("References", 14)
    p.para(
        ["[1] A. Author. Arrays and Lists. Example Press, 2001.", "[2] B. Writer. Graphs. 2003."]
    )
    # 8 chapter 3 (not selected)
    p = _P(doc, 8, None, True)
    p.heading("3 Extras", 20)
    p.para(["This chapter is not part of the selection and must not be ingested at all."])
    # 9 index
    p = _P(doc, 9, None, False)
    p.heading("Index", 20)
    p.para(["array, 1", "graph, 4", "list, 2"])
    if with_outline:
        doc.set_toc(
            [
                [1, "Contents", 3],
                [1, "Basics", 4],
                [2, "Lists and Arrays", 4],
                [1, "Graphs", 7],
                [1, "Extras", 8],
                [1, "Index", 9],
            ]
        )
    doc.save(path)
    doc.close()
    return path
