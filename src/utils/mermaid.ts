import type { Mermaid } from 'mermaid'
import katex from 'katex'

/**
 * Mermaid is ~680 kB of JS and only a handful of articles use a diagram, so it
 * is loaded on demand: the chunk is fetched the first time a page actually
 * contains a ```mermaid block, and never on pages that do not.
 */
let mermaidPromise: Promise<Mermaid> | null = null

function loadMermaid(): Promise<Mermaid> {
  if (!mermaidPromise) {
    mermaidPromise = import('mermaid').then(({ default: mermaid }) => {
      mermaid.initialize({
        startOnLoad: false,
        securityLevel: 'strict',
        // On a parse error, throw instead of injecting mermaid's error graphic
        // into the page — we show the source instead.
        suppressErrorRendering: true,
        theme: 'base',
        fontFamily:
          "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif",
        themeVariables: {
          fontSize: '12px',
          primaryColor: '#f1f6fd',
          primaryBorderColor: '#1a73e8',
          primaryTextColor: '#1a1a2e',
          secondaryColor: '#eef2f7',
          tertiaryColor: '#ffffff',
          lineColor: '#5f6368',
          textColor: '#1a1a2e',
        },
      })
      return mermaid
    })
  }
  return mermaidPromise
}

let renderCount = 0

/**
 * Inline math inside a diagram label, using the same `$...$` convention as the
 * article markdown: `$$` is not a delimiter, and a formula cannot span a line
 * break. The single capture group is what `String.split` keeps in its result.
 */
const INLINE_MATH = /(?<!\$)\$(?!\$)([^$\n]+?)\$(?!\$)/g

function katexSpan(tex: string): HTMLElement | null {
  const span = document.createElement('span')
  span.className = 'mermaid-math'
  try {
    span.innerHTML = katex.renderToString(tex.trim(), {
      displayMode: false,
      throwOnError: false,
    })
  } catch {
    return null
  }
  return span
}

/**
 * Replace `$...$` with KaTeX markup inside a rendered diagram.
 *
 * This runs *after* mermaid has laid the diagram out, and only touches HTML
 * labels: mermaid measures its label boxes before we get here, and its lexer
 * rejects the `{`/`}` in KaTeX's markup, so the formulas cannot be baked into
 * the diagram source. Labels are therefore sized against the literal `$...$`
 * text — usually wider than the typeset formula, so it fits. `.mermaid-math`
 * and the `foreignObject` overflow rule in ProjectDetail.vue cover the rest.
 */
function renderMathInLabels(root: ParentNode): void {
  const walker = document.createTreeWalker(root as Node, NodeFilter.SHOW_TEXT)
  const targets: Text[] = []
  for (let n = walker.nextNode(); n; n = walker.nextNode()) {
    const text = n as Text
    // SVG <text> cannot host KaTeX's markup — only HTML labels can.
    if (text.nodeValue?.includes('$') && text.parentElement?.closest('foreignObject')) {
      targets.push(text)
    }
  }

  for (const node of targets) {
    // Odd indices are the captured formulas, even ones the surrounding text.
    const parts = (node.nodeValue ?? '').split(INLINE_MATH)
    if (parts.length < 3) continue

    const frag = document.createDocumentFragment()
    for (let i = 0; i < parts.length; i++) {
      if (i % 2 === 1) {
        frag.appendChild(katexSpan(parts[i]) ?? document.createTextNode(`$${parts[i]}$`))
      } else if (parts[i]) {
        frag.appendChild(document.createTextNode(parts[i]))
      }
    }
    node.replaceWith(frag)
  }
}

/**
 * Replace every `.mermaid` element under `root` with its rendered SVG.
 *
 * Renders are sequential (mermaid's DOM-based measurement is not safe to run
 * concurrently) and each block is isolated: a diagram that fails to parse
 * keeps its original source and gets `mermaid--error`, so one bad diagram
 * cannot blank out the rest of the article.
 */
export async function renderMermaidBlocks(root: ParentNode): Promise<void> {
  const nodes = Array.from(root.querySelectorAll<HTMLElement>('.mermaid:not([data-processed])'))
  if (nodes.length === 0) return

  const mermaid = await loadMermaid()

  for (const node of nodes) {
    // Mark first so a re-run (route change, hot reload) never double-renders.
    node.dataset.processed = 'true'
    const source = node.textContent ?? ''
    if (!source.trim()) continue

    try {
      const { svg } = await mermaid.render(`mermaid-${renderCount++}`, source)
      node.innerHTML = svg
      renderMathInLabels(node)
      node.classList.add('mermaid--rendered')
    } catch (e) {
      node.classList.add('mermaid--error')
      console.error('[mermaid] failed to render diagram:', e)
    }
  }
}
