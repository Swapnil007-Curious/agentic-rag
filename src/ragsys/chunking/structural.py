"""Structure-aware chunking: one chunk per section/rule, split along the legal hierarchy.

* a unit (section / rule / schedule / top-level heading) of <= ``max_tokens`` is one chunk;
* larger units split at sub-section, then clause boundaries (recursively); a leaf that is
  still too large is split at sentence boundaries (word-snapped windows as a last resort);
* the introductory words of a split provision ("35. Manner ...—") are prepended to its first
  piece; consecutive tiny siblings (< ``min_tokens``) under the same parent are merged;
* in a definitions section every definition clause is its own chunk (definitions are never
  merged with each other);
* chunks never cross a unit boundary. ``text`` (embedded) = heading path + newline +
  ``display_text``; ``display_text`` is the exact cleaned text slice;
* with ``orphan_text`` (textbooks), text outside every unit (a chapter's introduction before
  its first section) is chunked as well, split by the same rules, with the enclosing node as
  primary node; chunks are then emitted in document order.
"""

from __future__ import annotations

from dataclasses import dataclass

from ragsys.chunking.base import Chunk, DocInput, deepest_containing, heading_path, make_chunk
from ragsys.chunking.fixed import token_windows
from ragsys.chunking.semantic import split_units
from ragsys.chunking.tokenizer import TokenCounter
from ragsys.structure.base import Node


@dataclass
class Piece:
    start: int
    end: int
    node: Node  # primary node
    parent: str  # parent tag for sibling merging
    is_def: bool
    tokens: int


class StructuralChunker:
    name = "structural"

    def __init__(
        self,
        tokenizer: TokenCounter,
        max_tokens: int = 384,
        min_tokens: int = 48,
        contextual_header: bool = True,
        *,
        orphan_text: bool = False,
    ) -> None:
        self.tok = tokenizer
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens
        self.contextual_header = contextual_header
        self.orphan_text = orphan_text

    # ------------------------------------------------------------------ helpers

    def _count(self, doc: DocInput, a: int, b: int) -> int:
        return self.tok.count(doc.text[a:b])

    @staticmethod
    def _trim(doc: DocInput, a: int, b: int) -> tuple[int, int]:
        seg = doc.text[a:b]
        return a + (len(seg) - len(seg.lstrip())), b - (len(seg) - len(seg.rstrip()))

    def _has_inner_defs(self, doc: DocInput, node: Node) -> bool:
        stack = list(node.children)
        while stack:
            child = doc.by_id.get(stack.pop())
            if child is None:
                continue
            if child.defined_term:
                return True
            stack.extend(child.children)
        return False

    def _split_text(self, doc: DocInput, a: int, b: int, node: Node, parent: str) -> list[Piece]:
        """Split a text span without structural children: sentences, then windows."""
        spans: list[tuple[int, int]] = []
        for ua, ub in split_units(doc.text[a:b]):
            ua, ub = a + ua, a + ub
            if self._count(doc, ua, ub) <= self.max_tokens:
                spans.append((ua, ub))
                continue
            seg = doc.text[ua:ub]
            offs = self.tok.offsets(seg)
            spans.extend(
                (ua + offs[s][0], ua + offs[e - 1][1])
                for s, e in token_windows(seg, offs, self.max_tokens, 0)
            )
        pieces: list[Piece] = []
        cur: tuple[int, int] | None = None
        for sa, sb in spans:
            if cur is not None and self._count(doc, cur[0], sb) <= self.max_tokens:
                cur = (cur[0], sb)
                continue
            if cur is not None:
                pieces.append(Piece(cur[0], cur[1], node, parent, False, self._count(doc, *cur)))
            cur = (sa, sb)
        if cur is not None:
            pieces.append(Piece(cur[0], cur[1], node, parent, False, self._count(doc, *cur)))
        return pieces

    # ------------------------------------------------------------------ recursion

    def _pieces(self, doc: DocInput, node: Node) -> list[Piece]:
        a, b = self._trim(doc, node.char_start, node.char_end)
        tokens = self._count(doc, a, b)
        parent_tag = node.parent_id or node.node_id
        if tokens <= self.max_tokens and not self._has_inner_defs(doc, node):
            return [Piece(a, b, node, parent_tag, bool(node.defined_term), tokens)]
        children = [doc.by_id[c] for c in node.children if c in doc.by_id]
        if not children:
            return self._split_text(doc, a, b, node, node.node_id)

        pieces: list[Piece] = []
        intro: Piece | None = None
        cursor = node.char_start
        for i, ch in enumerate(children):
            ga, gb = self._trim(doc, cursor, ch.char_start)
            if gb > ga:
                own = self._split_text(doc, ga, gb, node, node.node_id)
                if i == 0 and len(own) == 1:
                    intro = own[0]
                else:
                    pieces.extend(own)
            child_pieces = self._pieces(doc, ch)
            if intro is not None and child_pieces:
                first = child_pieces[0]
                if self._count(doc, intro.start, first.end) <= self.max_tokens:
                    child_pieces[0] = Piece(
                        intro.start,
                        first.end,
                        first.node,
                        first.parent,
                        first.is_def,
                        self._count(doc, intro.start, first.end),
                    )
                else:
                    pieces.append(intro)
                intro = None
            pieces.extend(child_pieces)
            cursor = ch.char_end
        ga, gb = self._trim(doc, cursor, node.char_end)
        if gb > ga:
            pieces.extend(self._split_text(doc, ga, gb, node, node.node_id))
        return self._merge_tiny(doc, pieces)

    def _merge_tiny(self, doc: DocInput, pieces: list[Piece]) -> list[Piece]:
        changed = True
        while changed:
            changed = False
            for i in range(len(pieces) - 1):
                p, q = pieces[i], pieces[i + 1]
                if p.parent != q.parent or (p.is_def and q.is_def):
                    continue
                if p.tokens >= self.min_tokens and q.tokens >= self.min_tokens:
                    continue
                total = self._count(doc, p.start, q.end)
                if total > self.max_tokens:
                    continue
                common = doc.by_id.get(p.parent, p.node)
                node = p.node if p.node is q.node else common
                pieces[i : i + 2] = [
                    Piece(p.start, q.end, node, p.parent, p.is_def or q.is_def, total)
                ]
                changed = True
                break
        return pieces

    # ------------------------------------------------------------------ public

    def _orphan_pieces(self, doc: DocInput, units: list[Node]) -> list[Piece]:
        """Pieces of the text that lies outside every unit."""
        covered = sorted((u.char_start, u.char_end) for u in units)
        gaps: list[tuple[int, int]] = []
        cursor = 0
        for a, b in covered:
            if a > cursor:
                gaps.append((cursor, a))
            cursor = max(cursor, b)
        gaps.append((cursor, len(doc.text)))
        pieces: list[Piece] = []
        for a, b in gaps:
            a, b = self._trim(doc, a, b)
            if b <= a:
                continue
            owner = deepest_containing(doc, a, b)
            if owner is None:
                continue
            parent = owner.parent_id or owner.node_id
            pieces.extend(self._merge_tiny(doc, self._split_text(doc, a, b, owner, parent)))
        return pieces

    def chunk_document(self, doc: DocInput) -> list[Chunk]:
        chunks: list[Chunk] = []
        units = [n for n in doc.nodes if n.is_unit]
        pieces = [p for unit in units for p in self._pieces(doc, unit)]
        if self.orphan_text:
            pieces = sorted(
                pieces + self._orphan_pieces(doc, units), key=lambda p: (p.start, p.end)
            )
        for piece in pieces:
            path = heading_path(doc, piece.node, include_sub=True)
            display = doc.text[piece.start : piece.end]
            embed = f"{path}\n{display}" if self.contextual_header else display
            chunks.append(
                make_chunk(
                    doc,
                    self.name,
                    len(chunks),
                    piece.start,
                    piece.end,
                    self.tok.count(embed),
                    primary=piece.node,
                    embed_text=embed,
                    path=path,
                )
            )
        return chunks
