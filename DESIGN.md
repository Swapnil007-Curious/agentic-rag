# Veridex: design plan

Veridex is the public web app of this repository (research codename Agentic-RAG). It runs
retrieval in the visitor's browser and streams answers from the LLM proxy. This document is the
plan the redesign was built to: tokens, layout, components, motion, and the map from every UI
element to the real data behind it. The app stays vanilla TypeScript + Vite; nothing in the
retrieval or LLM code changed (see "What changed in the engine" at the end).

Concept: **Lapis & Brass**. A notary's seal for knowledge. Deep lapis surfaces, scholarly type,
brass marks of proof. Brass is spent only on proof: citations, the verification trace, the seal,
and the primary action.

## Tokens

CSS custom properties in `web-static/src/styles/tokens.css`. Dark is the default; the theme follows
`prefers-color-scheme` until the visitor picks one (`html[data-theme]`).

| token | dark | light |
|---|---|---|
| `--bg` | #0B1230 | #F5F7FC |
| `--surface` | #121A3D | #FFFFFF |
| `--surface-2` | #1A2350 | #EEF1F9 |
| `--line` | rgba(231,236,255,.08) | rgba(11,18,48,.08) |
| `--text` | #E7ECFF | #0B1230 |
| `--text-muted` | #8C96C0 | #5A6390 |
| `--brass` (marks, seal, bars, focus ring) | #D9A441 | #A87414 |
| `--brass-text` (brass used as text) | #D9A441 | #8B5F0B |
| `--brass-soft` | rgba(217,164,65,.14) | rgba(168,116,20,.12) |
| `--on-brass` (text on the primary button) | #0B1230 | #0B1230 |
| `--supported` | #4FD1A5 | #1C7A5B |
| `--partial` | #F2C14E | #8A5A00 |
| `--unsupported` | #F07A7A | #B9353A |

`--brass-text` exists because the brief's light brass (#A87414) reaches only 3.79:1 on the light
background, enough for marks (3:1) but not for text (4.5:1). Signal colours are tuned per theme
for the same reason. The ratios below come from `node web-static/scripts/contrast.mjs`.

Radius 10 / 14 / 20 / 28 px. Spacing in rem on a 4 px base. Shadows are layered and tinted lapis
(`rgba(4,8,30,…)`), never grey. Glass: `backdrop-filter: blur(20px) saturate(160%)`, a
semi-transparent surface, a 1 px brighter top edge; bigger surfaces get stronger blur and a deeper
shadow; glass never sits on glass. Ambient background: two lapis radial-gradient layers (max
opacity 0.25) drifting over 48 s and 56 s; static under `prefers-reduced-motion`. The drift
advances in 240 steps per cycle (under a pixel each), so it reads as continuous while the glass
above re-blurs 5 times a second instead of 60. Where content meets the top bar and the composer,
static gradient fades soften the edge (no divider line); an earlier CSS mask on the scrolling
layer was replaced because a mask over a moving background is re-rendered every frame (see
Performance).

Typography (self-hosted from the fontsource packages, latin subsets only, `font-display: swap`,
3 font files, the body weight preloaded):

- Display: Instrument Serif 400 for the wordmark, hero headline, section titles. Large sizes use
  letter-spacing −0.02em and line-height 1.05.
- UI/body: Geist (variable) with system-ui fallback. 16 px / 1.6, letter-spacing 0.
- Mono: Geist Mono (variable) for citation IDs, scores, timings, keyboard hints; small sizes
  +0.01em.
- Fluid scale with `clamp()`: `--fs-display`, `--fs-h1`, `--fs-h2`, `--fs-body`, `--fs-small`,
  `--fs-micro`.

## Layout

Desktop (≥1100 px): `[ rail 264 px, collapsible to 64 px ] [ conversation, reading width 760 px ] [ evidence 380 px ]`.
The evidence panel springs in from the right and out to the right; the conversation column keeps
its layout with a FLIP transform so nothing jumps.
Tablet (700–1099 px): the rail is a slide-over drawer; the evidence panel is a right drawer.
Mobile (<700 px): compact glass top bar (menu, collection pill, theme); composer pinned to the
bottom with `env(safe-area-inset-bottom)`; evidence is a bottom sheet with two detents (40 % and
90 %), 1:1 drag tracking, momentum projection, rubber-banding and swipe-down to dismiss.

## Components → real data

Every element below shows data the pipeline really produces (`web-static/src/engine.ts`,
`gate.ts`, `grounded.ts`, `verify.ts`, `public/data/manifest.json`). Nothing is simulated.

| UI element | Real data behind it |
|---|---|
| Collection tiles and rail list: name, documents, passages | `manifest.collections[].label`, `.sources.length`, `.count` |
| Collection load status in the pill/popover | `CollectionStore.load` byte progress (`passages.bytes + vectors.bytes`) |
| Hero seal assembling + "Preparing the search index – x/y MB" | `Embedder.onProgress(loaded, total, cached)` from the model worker |
| Suggested questions | Written for this redesign and checked against each collection's `passages.json` (grep of the supporting sections) |
| Mode control: Chat / Grounded only | `ChatRequest.groundedOnly` (the two modes that exist today) |
| Attach PDF, upload list and progress | `UploadStore.add` (`reading` / `embedding` progress) |
| Verification trace: Embed, Search | `GateDecision.embedMs`, `.searchMs` (Chat); timed around `embedder.queries` / `mergeHits` (Grounded only) |
| Trace: Draft, first token | the streamed proxy call; `ttft_ms` |
| Trace: Synthesize, Guards, Critic, Revise | the `synth` / `critic` proxy calls the grounded pipeline makes (timed in `engine.ts`); `GroundedResult.guards`, `.critic`, `.llmCalls` |
| Trace: Verify | `verifyAnswer` (guards here, critic through the proxy), timed |
| Seal verdict, Chat mode | `VerifyResult.supported / total` after the visitor clicks Verify: all → supported, some → partial, none → unsupported. Before Verify the answer shows only its source count. |
| Seal verdict, Grounded only | `outcome === "answered"` with `guards.passed` and `critic.verdict === "accept"` → supported; `outcome === "abstained"` → not enough evidence; answered without a critic call → source count only |
| Sentence list under the seal | `VerifyResult.sentences[]` with `status` and `reasons` |
| Inline citation chips `[Cn]` | `Passage.label` of the passages the reply actually cites (`cleanCitations`) |
| Source card: title, section, pages | `manifest.collections[].sources[].title` by `doc_id` (or the upload's name), `Passage.heading_path`, `.pages` |
| Source card: relevance bar + number | `Passage.score` (cosine of the int8 index) |
| Source card: highlighted terms | the words of the visitor's question (≥4 letters) found in `Passage.text` |
| Abstention card | `outcome === "abstained"`, `abstentionMessage`, `closest[]` sections |
| Rate limit countdown | `ProxyError.code` (`rate_limited`, `busy`, `daily_cap`) and `retryAfter` from the Worker's JSON |
| General-knowledge line | `grounded === false` (the gate found no passage above the threshold) |
| Share link `?q=&c=` | pre-fills the composer and selects the collection on load; never sends |
| History: Today / Earlier | `Conversation.updated` |

## Motion spec

`motion/mini` `animate` with the `spring` generator (6.3 KB gzip). CSS transitions only for
hover and colour.

| motion | spring / timing |
|---|---|
| UI movement (panel, drawer, popover, indicator) | `bounce: 0`, `visualDuration: 0.35` (damping 1.0, response 0.35) |
| Momentum gestures (sheet flick, drawer throw) | `bounce: 0.2`, `visualDuration: 0.3`, release velocity handed to the spring |
| Seal stamp | scale 1.15 → 1, `bounce: 0.3` (≈ damping 0.7); one brass ripple ring to opacity 0; check and ring draw with `stroke-dashoffset`; `navigator.vibrate(10)` on coarse pointers; everything starts on the same frame |
| Streaming tokens | opacity 0 → 1, translateY 4 px → 0, 120 ms; blinking brass caret |
| Press feedback | `:active` scale(0.97), 100 ms, on pointer-down |
| Page load | aurora fades in → seal draws → wordmark and tagline rise (transform only, opaque from the first frame) → tiles stagger at 40 ms; under 700 ms in total |
| Seal idle | the tick ring turns 2° every half second, like a watch hand (90 s per turn) |
| Collection switch | suggested questions cross-fade with a 6 px slide; the mode indicator slides |
| Command palette | scale 0.96 → 1 with a blur-to-sharp materialise |
| Reduced motion | every slide/spring/stagger becomes a 150 ms opacity fade; aurora static; the stamp fades in |

Only `transform`, `opacity` and SVG `stroke-dashoffset` are animated (the source card's expand
height is the one requested exception). Every animation is interruptible: it starts from the
current on-screen value and never locks input. Enter and exit share a path.

## Accessibility floor

WCAG 2.2 AA pairs in both themes (table below). Focus ring 2 px brass, 2 px offset. Full keyboard
use; Esc closes any overlay; focus is trapped in dialogs and the mobile sheet and returns to the
trigger. The answer region is `aria-live="polite"` and announces on completion, not per token.
Verdicts and errors are announced. Touch targets ≥ 44 px. Shortcuts: Ctrl/Cmd+K palette,
`/` focuses the composer, Ctrl/Cmd+Shift+O new chat, Esc stops streaming or closes overlays; all
listed under "Keyboard shortcuts" in the palette. `prefers-reduced-transparency`: solid surfaces,
no blur. `prefers-contrast: more`: solid backgrounds and 1 px contrasting borders.

## Contrast (node web-static/scripts/contrast.mjs)

| theme | use | pair | ratio | AA (4.5:1) |
|---|---|---|---|---|
| dark | body text | #E7ECFF on #0B1230 | 15.60:1 | 4.5:1 | pass |
| dark | text on cards | #E7ECFF on #121A3D | 14.38:1 | 4.5:1 | pass |
| dark | text in user bubble | #E7ECFF on #1A2350 | 12.74:1 | 4.5:1 | pass |
| dark | muted text (hints, timings) | #8C96C0 on #0B1230 | 6.33:1 | 4.5:1 | pass |
| dark | muted text on cards | #8C96C0 on #121A3D | 5.83:1 | 4.5:1 | pass |
| dark | brass text: citation chips, labels | #D9A441 on #0B1230 | 8.17:1 | 4.5:1 | pass |
| dark | brass text on cards | #D9A441 on #121A3D | 7.52:1 | 4.5:1 | pass |
| dark | brass marks (seal, bars, focus ring): non-text 3:1 | #D9A441 on #0B1230 | 8.17:1 | 3:1 | pass |
| dark | text on the primary (brass) button | #0B1230 on #D9A441 | 8.17:1 | 4.5:1 | pass |
| dark | verified label | #4FD1A5 on #0B1230 | 9.62:1 | 4.5:1 | pass |
| dark | verified label on cards | #4FD1A5 on #121A3D | 8.86:1 | 4.5:1 | pass |
| dark | partially supported label | #F2C14E on #0B1230 | 10.94:1 | 4.5:1 | pass |
| dark | partial label on cards | #F2C14E on #121A3D | 10.08:1 | 4.5:1 | pass |
| dark | not enough evidence label | #F07A7A on #0B1230 | 6.80:1 | 4.5:1 | pass |
| dark | abstained label on cards | #F07A7A on #121A3D | 6.26:1 | 4.5:1 | pass |
| light | body text | #0B1230 on #F5F7FC | 17.13:1 | 4.5:1 | pass |
| light | text on cards | #0B1230 on #FFFFFF | 18.37:1 | 4.5:1 | pass |
| light | text in user bubble | #0B1230 on #EEF1F9 | 16.25:1 | 4.5:1 | pass |
| light | muted text (hints, timings) | #5A6390 on #F5F7FC | 5.41:1 | 4.5:1 | pass |
| light | muted text on cards | #5A6390 on #FFFFFF | 5.80:1 | 4.5:1 | pass |
| light | brass text: citation chips, labels | #8B5F0B on #F5F7FC | 5.24:1 | 4.5:1 | pass |
| light | brass text on cards | #8B5F0B on #FFFFFF | 5.62:1 | 4.5:1 | pass |
| light | brass marks (seal, bars, focus ring): non-text 3:1 | #A87414 on #F5F7FC | 3.79:1 | 3:1 | pass |
| light | text on the primary (brass) button | #0B1230 on #A87414 | 4.52:1 | 4.5:1 | pass |
| light | verified label | #1C7A5B on #F5F7FC | 4.92:1 | 4.5:1 | pass |
| light | verified label on cards | #1C7A5B on #FFFFFF | 5.27:1 | 4.5:1 | pass |
| light | partially supported label | #8A5A00 on #F5F7FC | 5.53:1 | 4.5:1 | pass |
| light | partial label on cards | #8A5A00 on #FFFFFF | 5.93:1 | 4.5:1 | pass |
| light | not enough evidence label | #B9353A on #F5F7FC | 5.39:1 | 4.5:1 | pass |
| light | abstained label on cards | #B9353A on #FFFFFF | 5.78:1 | 4.5:1 | pass |

## Performance

Model and index loading are unchanged, except that the model worker starts one frame after the
hero has painted (the downloads begin a few milliseconds later; nothing else moved). KaTeX and
highlight.js load on first use (an answer with maths or a code block); the command palette and
How it works are dynamic imports. Three font files; only the body face is preloaded. Space for the
trace strip and the seal is reserved so the answer does not shift while streaming. The wordmark,
tagline and suggested questions paint from static data before any download starts, and text that
updates during loading is only rewritten when it changes, so the largest paint lands at first
render.

JS added by the redesign (gzip level 9, excluding fonts, KaTeX and the unchanged pdf.js and
embedding-worker chunks): **21.9 KB**, within the 60 KB budget.

| bundle | before | after |
|---|---|---|
| main bundle (`index-*.js`) | 22.6 KB | 40.7 KB |
| command palette (loaded on Ctrl/Cmd+K) | n/a | 1.6 KB |
| How it works (loaded on open) | n/a | 2.2 KB |
| stylesheet | 4.5 KB | 9.4 KB |

The old page also loaded KaTeX and highlight.js on every visit; they now load only when an answer
needs them.

Lighthouse (mobile, `vite preview` of the production build, `node web-static/scripts/lighthouse.mjs`):

| category | Veridex | original app (same machine, same script) |
|---|---|---|
| Performance | **65** | 63 |
| Accessibility | **100** | 100 |
| Best Practices | **100** | 100 |
| SEO | **100** | 91 |

| metric | Veridex | original app |
|---|---|---|
| First Contentful Paint | 0.8 s | |
| Largest Contentful Paint | 0.8 s | 2.2 s |
| Total Blocking Time | 7,100 ms | 6,380 ms |
| Cumulative Layout Shift | 0.011 | |
| Speed Index | 5.6 s | 6.2 s |

The scores are in `docs/lighthouse.json`. Total Blocking Time varies a lot between runs on this
machine (3–7 s for the same build), so Performance moves between about 63 and 68.

**The Performance target of 85 is not met while the glass and the aurora are on.** Almost all of
the blocking time is one main-thread task of 1–2 s with no script, style or layout work traced
inside it, and it goes away when the blur and the aurora are removed. The likely cause is that
Lighthouse's lab browser runs headless without a GPU, so `backdrop-filter` and the large aurora
layers are composited in software. Measured on the same build:

| variant | Performance | Total Blocking Time |
|---|---|---|
| as designed (glass on the bars, aurora) | 63–65 | 6.3–7.3 s |
| bars without blur, aurora kept | 65–66 | 4.3–5.6 s |
| aurora static | 67–68 | 3.0–3.9 s |
| no aurora, glass kept | 71 | 1.9 s |
| no blur and no aurora | **86–91** | 0.4–0.55 s |

So the brief's materials (translucent glass, ambient aurora) and the 85 target conflict in this
lab. Changes that keep the look were made: the scroll-edge mask became static gradients, the
aurora drift and the seal's idle ticks advance in discrete steps, the desktop rail and evidence
panel (which sit beside the content, over nothing but the ambient gradient) do not blur, and the
model worker starts after the first paint, which brought Largest Contentful Paint from 8.0 s to
0.8 s. Reaching 85 needs a design decision: drop the blur on the bars and the aurora (or reserve
them for devices with GPU compositing), which the brief currently rules out.

## What changed in the engine

`web-static/src/engine.ts` gained an optional `onStage` callback that reports the existing
stages as they start and finish, with their real durations, and `ChatOutcome` now also carries
the grounded pipeline's `guards`, `critic`, `reason`, `detail` and `llmCalls` (fields
`groundedAnswer` already returned) plus `embed_ms` / `search_ms`. No decision, prompt, threshold
or call order changed. `deploy/worker/`, `site-config.ts`, the Worker URL resolution and the
Pages workflow are untouched.
