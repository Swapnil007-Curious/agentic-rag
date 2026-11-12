import { describe, expect, it } from "vitest";
import { chunkLines, isHeading, type Line } from "../../src/chunker.ts";
import { cleanName, checkFile, UploadError } from "../../src/uploads.ts";

describe("numbered-heading rule (generic parser)", () => {
  it("accepts numbered headings and rejects sentences", () => {
    expect(isHeading("2 Findings")).toEqual({ number: "2", title: "Findings", level: 1 });
    expect(isHeading("3.1.2 Method of Analysis")).toMatchObject({ level: 3 });
    expect(isHeading("3. Staff")).toMatchObject({ number: "3", title: "Staff" });
    expect(isHeading("3 staff lower case")).toBeNull();
    expect(isHeading("2 Findings were made.")).toBeNull();
    expect(isHeading("In 2023 the station recorded 4.2 cm.")).toBeNull();
  });
});

describe("chunking uploaded lines", () => {
  const lines: Line[] = [
    { text: "Station Report", page: 1 },
    { text: "1 Introduction", page: 1 },
    { text: "The station was founded in 2019. ".repeat(3), page: 1 },
    { text: "2 Findings", page: 2 },
    { text: "The melt rate was 4.2 cm per day.", page: 2 },
    { text: "2.1 Details", page: 2 },
    { text: "x ".repeat(2000), page: 3 },
  ];
  it("builds heading paths, pages and ids", () => {
    const chunks = chunkLines(lines, "Station Report", "up_abc");
    expect(chunks[0]?.heading_path).toBe("Station Report");
    expect(chunks[0]?.text).toBe("Station Report");
    expect(chunks[1]?.heading_path).toBe("Station Report > 1 Introduction");
    expect(chunks[1]?.section_ids).toEqual(["up_abc:h1"]);
    expect(chunks[1]?.pages).toBe("1");
    const findings = chunks.find((c) => c.heading_path.endsWith("2 Findings"));
    expect(findings?.text).toContain("4.2 cm");
    const details = chunks.filter((c) => c.heading_path.endsWith("2 Findings > 2.1 Details"));
    expect(details.length).toBeGreaterThan(1); // a long section is split
    expect(details.every((c) => c.n_tokens <= 384 * 1.25)).toBe(true);
    expect(chunks.every((c, i) => c.chunk_id === `structural:up_abc:${String(i).padStart(5, "0")}`)).toBe(true);
    expect(chunks.every((c) => c.embedText.startsWith(c.heading_path))).toBe(true);
  });
});

describe("upload checks", () => {
  it("cleans names and refuses non-PDFs and large files", () => {
    expect(cleanName("../../etc/passwd")).toBe("passwd");
    expect(cleanName('a<script>"x".pdf')).toBe("a script x .pdf");
    expect(cleanName("")).toBe("document.pdf");
    const limits = { max_files: 3, max_mb: 1, max_pages: 300, max_chunks_per_file: 4000 };
    expect(() => checkFile("a.txt", new TextEncoder().encode("hello"), limits)).toThrow(UploadError);
    expect(() => checkFile("a.pdf", new Uint8Array(1024 * 1024 + 1), limits)).toThrow(/larger than 1 MB/);
    expect(() => checkFile("a.pdf", new TextEncoder().encode("%PDF-1.7 ok"), limits)).not.toThrow();
  });
});
