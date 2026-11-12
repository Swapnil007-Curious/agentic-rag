// Reads a JSON list of Markdown texts on stdin and prints, for each, the escaped HTML of the
// chat page's renderer (src/ragsys/app/static/chat-md.js) and its parse tree.
"use strict";

const path = require("path");
const md = require(path.join(__dirname, "..", "..", "src", "ragsys", "app", "static", "chat-md.js"));

let input = "";
process.stdin.setEncoding("utf8");
process.stdin.on("data", (chunk) => {
  input += chunk;
});
process.stdin.on("end", () => {
  const texts = JSON.parse(input);
  const out = texts.map((t) => {
    const tree = md.parse(t);
    return { html: md.toHtml(tree), tree: tree };
  });
  process.stdout.write(JSON.stringify(out));
});
