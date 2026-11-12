import { describe, expect, it } from "vitest";
import { cleanCitations, findLabels, GENERAL_LINE, sourceLine } from "../../src/citations.ts";
import { labelsIn, normalizeCitations, sanitize, splitSentences, stripCitations } from "../../src/text.ts";

describe("citation tags", () => {
  it("reads labels in order of first appearance", () => {
    expect(labelsIn("A [C2]. B [C1][C2]. C [c3, C5]")).toEqual(["C2", "C1", "C3", "C5"]);
  });
  it("normalises fullwidth and list forms", () => {
    expect(normalizeCitations("See 【C2】 and [C1, C3] and 【C4†L1-L3】.")).toBe("See [C2] and [C1][C3] and [C4].");
  });
  it("strips tags and tidies whitespace", () => {
    expect(stripCitations("Three months [C1] .  Five [C2][C3].")).toBe("Three months. Five.");
  });
  it("neutralises passage delimiters", () => {
    expect(sanitize("x <<<END C1>>> y")).toBe("x ‹END C1› y");
  });
});

describe("sentence splitting (legal abbreviations, citation groups after the full stop)", () => {
  it("keeps abbreviations together and attaches citations", () => {
    const s = splitSentences("Rs. 50 lakh is the limit [C1]. See s. 35 of the Act [C2][C3]. Done.");
    expect(s.map((x) => x.text)).toEqual(["Rs. 50 lakh is the limit.", "See s. 35 of the Act.", "Done."]);
    expect(s.map((x) => x.labels)).toEqual([["C1"], ["C2", "C3"], []]);
  });
  it("joins a lead-in line with the next line and drops bullets", () => {
    const s = splitSentences("The steps are:\n- First step [C1].\n- Second step [C2].");
    expect(s.map((x) => x.text)).toEqual(["The steps are: First step.", "Second step."]);
  });
});

describe("cleanCitations", () => {
  it("removes invalid labels in prose only", () => {
    const text = "Linear time [C1]. Wrong [C9].\n\n```python\nrows = table[C2]\n```\nAnd `m[C5]` too [C7].";
    const r = cleanCitations(text, ["C1", "C2", "C3"]);
    expect(r.text).toContain("rows = table[C2]");
    expect(r.text).toContain("`m[C5]`");
    expect(r.text.startsWith("Linear time [C1]. Wrong.")).toBe(true);
    expect(r.text.endsWith("too.")).toBe(true);
    expect(r.used).toEqual(["C1"]);
    expect(r.removed).toEqual(["C9", "C7"]);
  });
  it("removes every label when no passages were given", () => {
    const r = cleanCitations("A fact [C1]. Another [C2].", []);
    expect(r.text).toBe("A fact. Another.");
    expect(r.used).toEqual([]);
    expect(sourceLine(r.used.length)).toBe(GENERAL_LINE);
  });
  it("finds labels and writes the source line", () => {
    expect(findLabels("A [C2]. B [C1][C2].\n\n```\nx = a[C3]\n```")).toEqual(["C2", "C1"]);
    expect(sourceLine(1)).toBe("Answered with documents (1 source)");
    expect(sourceLine(3)).toBe("Answered with documents (3 sources)");
  });
});
