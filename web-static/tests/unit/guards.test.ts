import { describe, expect, it } from "vitest";
import { citationValidity, extractNumbers, numberForms, numberGrounding, runGuards, type PassageLike } from "../../src/guards.ts";

const passages: PassageLike[] = [
  { label: "C1", heading_path: "Test Act > Section 38", section_ids: ["testact:s38"], text: "Every complaint shall be decided within three months where no testing of goods is required." },
  { label: "C2", heading_path: "Test Act > Section 41", section_ids: ["testact:s41"], text: "An appeal may be filed within forty-five days. The limit is Rs. 50,00,000." },
];

describe("number extraction (by value)", () => {
  it("reads digits, words and multipliers", () => {
    // word numbers run across punctuation ("fifty lakh, one crore" is one mention), as in Python
    const values = extractNumbers("thirty days, Rs. 50,00,000, fifty lakh, one crore, 2.5 million, one hundred and five").map((m) => m.value);
    expect(values).toEqual([30, 5_000_000, 15_000_000, 2_500_000, 105]);
    expect(extractNumbers("fifty lakh. One crore").map((m) => m.value)).toEqual([15_000_000]);
  });
  it("keeps the base before a multiplier", () => {
    const [m] = extractNumbers("fifty lakh");
    expect(m?.base).toBe(50);
    expect(numberForms(["fifty lakh"])).toEqual(new Set([5_000_000, 50]));
  });
});

describe("guards", () => {
  it("passes a supported answer", () => {
    const r = runGuards("A complaint is decided within three months [C1]. An appeal lies within 45 days [C2].", ["C1", "C2"], passages);
    expect(r.passed).toBe(true);
    expect(r.status).toBe("supported");
  });
  it("flags missing citations, unknown labels and ungrounded numbers", () => {
    const r = runGuards("It costs 999 rupees [C1]. No citation here. Unknown [C9].", ["C1"], passages);
    expect(r.passed).toBe(false);
    expect(r.reasons.join("\n")).toContain('number "999" does not occur');
    expect(r.reasons.join("\n")).toContain("sentence 2 has no citation");
    expect(r.reasons.join("\n")).toContain("not in the context: C9");
  });
  it("matches numbers written differently", () => {
    expect(numberGrounding("The limit is 50 lakh rupees [C2].", [passages[1] as PassageLike]).passed).toBe(true);
    expect(numberGrounding("Within 45 days [C2].", [passages[1] as PassageLike]).passed).toBe(true);
    expect(citationValidity("", [], ["C1"]).reasons).toEqual(["the answer is empty"]);
  });
  it("ignores a bare 'one'", () => {
    expect(numberGrounding("Any one of them may file [C1].", [passages[0] as PassageLike]).passed).toBe(true);
  });
});
