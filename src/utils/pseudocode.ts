import katex from 'katex'
import '../styles/pseudocode.css'

/*
 * pseudocode.js 2.4.1 is patched (see patches/pseudocode@2.4.1.patch).
 *
 * `HTMLBuilder._beginTag` assigns to `attrVal` without declaring it, which is
 * only legal in sloppy mode: the assignment creates an implicit global. Vite
 * bundles the package to ESM, which is always strict, so that path throws
 * `ReferenceError: attrVal is not defined`.
 *
 * It is reached whenever a style is passed as an object rather than a string,
 * which is not an edge case: the caption, every pre-condition line
 * (`\Require`/`\Ensure`), and the `ps-linenum` span of every numbered line all
 * take that branch. Without the patch nothing rendered at all — the caption is
 * emitted first, so the crash happened before any output existed.
 *
 * 2.4.1 is the newest release, so there is nothing to upgrade to.
 */

type PseudocodeModule = typeof import('pseudocode')

/**
 * pseudocode.js and its KaTeX dependency are only needed by articles that
 * actually carry an algorithm, so the whole thing is loaded on demand.
 */
let modulePromise: Promise<PseudocodeModule> | null = null

function loadPseudocode(): Promise<PseudocodeModule> {
  if (!modulePromise) {
    // pseudocode.js picks its math backend by probing for a global `katex`,
    // falling back to `require('katex')`. That fallback cannot work in an ESM
    // bundle — there is no `require`, and the resulting throw is swallowed by
    // the library's own try/catch — so without this it finds no backend at all
    // and throws `No math backend found` on the first `$...$`. Publish ours
    // before its module body evaluates (the assignment is read at import time).
    ;(globalThis as unknown as Record<string, unknown>).katex = katex
    modulePromise = import('pseudocode')
  }
  return modulePromise
}

/**
 * Reduce a LaTeX fragment to the form pseudocode.js expects.
 *
 * Authors naturally paste a complete document — it is what they compile to get
 * the reference PDF — so most of this is shedding what is valid LaTeX but
 * outside the algorithm environment. The last two rewrites are different in
 * kind: they are idioms that are valid LaTeX *and* valid input to
 * pseudocode.js, but that it renders differently than the author wrote them.
 */
export function normalizeAlgorithmSource(latex: string): string {
  let src = latex

  // Keep only the document body, if there is one.
  const body = src.match(/\\begin\{document\}([\s\S]*?)\\end\{document\}/)
  if (body) src = body[1]

  // Drop the preamble that lives outside the body.
  src = src
    .replace(/\\documentclass\b(\[[^\]]*\])?\{[^}]*\}/g, '')
    .replace(/\\usepackage\b(\[[^\]]*\])?\{[^}]*\}/g, '')

  // `\begin{algorithmic}[1]` — the optional argument selects LaTeX's own line
  // numbering, which pseudocode.js does not accept; numbering is a renderer
  // option here (see lineNumber below).
  src = src.replace(/\\begin\{algorithmic\}\s*\[[^\]]*\]/g, '\\begin{algorithmic}')

  // Display math. pseudocode.js's lexer knows only `$...$` and `\(...\)` — there
  // is no `$$` rule, so `$$` lexes as an *empty* inline math atom and the TeX
  // that follows it is then read as text, which fails on the first `_` or `^`.
  //
  // Demoting to inline math parses, and lands the formula where the author
  // meant: the lexer skips whitespace *including newlines*, so a `$$...$$` on
  // its own line behind a `\State` is absorbed into that statement and renders
  // as a continuation of it.
  src = src.replace(/\$\$/g, '$')

  // `\State \Return x` is the usual way to spell a return in `algorithmic`, but
  // `\Return` is already a statement keyword here. The leading `\State` opens a
  // statement of its own and renders as a stray empty numbered line.
  src = src.replace(/\\State\s+(?=\\Return\b)/g, '')

  return src.trim()
}

/**
 * Replace every `pre.pseudocode` element under `root` with rendered algorithm
 * markup.
 *
 * The element keeps its LaTeX source as text until this runs, so a block that
 * fails to parse stays readable (and gets `pseudocode--error`) instead of
 * vanishing. Each block is rendered independently for the same reason.
 */
export async function renderPseudocodeBlocks(root: ParentNode): Promise<void> {
  const nodes = Array.from(
    root.querySelectorAll<HTMLElement>('pre.pseudocode:not([data-processed])')
  )
  if (nodes.length === 0) return

  const { default: pseudocode } = await loadPseudocode()

  for (const node of nodes) {
    // Mark first so a re-run (route change, hot reload) never double-renders.
    node.dataset.processed = 'true'
    const source = node.textContent ?? ''
    if (!source.trim()) continue

    try {
      // No `mathRenderer` option: 2.4.1 documents and reads no such thing, it
      // just uses whichever backend `loadPseudocode` published above.
      node.innerHTML = pseudocode.renderToString(normalizeAlgorithmSource(source), {
        lineNumber: true,
      })
      node.classList.add('pseudocode--rendered')
    } catch (e) {
      node.classList.add('pseudocode--error')
      console.error('[pseudocode] failed to render algorithm:', e)
    }
  }
}
