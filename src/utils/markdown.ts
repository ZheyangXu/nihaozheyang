import { marked } from 'marked'
import katex from 'katex'

const PLACEHOLDER_PREFIX = '\x00MD_'

/**
 * Escape a raw string so it survives as inert text inside the HTML we emit.
 * Used for Mermaid sources so the browser does not parse `<br/>` (and friends)
 * into real elements before mermaid.js reads them back via textContent.
 */
function escapeHtml(text: string): string {
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
}

/**
 * Custom marked renderer that:
 * - Transforms asset paths from ../public/... to /...
 * - Adds lazy loading to images
 */
function createRenderer(): marked.Renderer {
  const renderer = new marked.Renderer()

  renderer.image = function ({ href, title, text }: { href: string; title?: string | null; text: string }) {
    const src = href.replace(/^(\.\.\/)+public\//, '/')
    const titleAttr = title ? ` title="${title}"` : ''
    const alt = text || ''
    return `<img src="${src}" alt="${alt}"${titleAttr} loading="lazy" />`
  }

  renderer.html = function ({ text }: { text: string }) {
    return text.replace(/(src|data-src)="(\.\.\/)+public\//g, '$1="/')
  }

  renderer.link = function ({ href, title, text }: { href: string; title?: string | null; text: string }) {
    const titleAttr = title ? ` title="${title}"` : ''
    return `<a href="${href}"${titleAttr}>${text}</a>`
  }

  return renderer
}

/**
 * Render markdown to HTML with KaTeX math support.
 *
 * Pipeline:
 * 0. Extract ```mermaid fences → <div class="mermaid"> placeholders
 * 0b. Extract <pre class="pseudocode"> → placeholders (LaTeX kept verbatim)
 * 1. Protect fenced code blocks (```...```)
 * 2. Protect inline code (`...`)
 * 3. Extract $$...$$ display math → render with KaTeX
 * 4. Extract $...$ inline math → render with KaTeX
 * 5. Parse remaining markdown with marked
 * 6. Restore KaTeX + Mermaid + pseudocode + code blocks
 * 7. Unwrap block placeholders that marked wrapped in <p>
 */
export function renderMarkdown(markdown: string): string {
  const tokens: string[] = []

  function save(html: string): string {
    const idx = tokens.length
    tokens.push(html)
    return `${PLACEHOLDER_PREFIX}${idx}\x00`
  }

  let text = markdown

  // ── Step 0: Extract ```mermaid fences ──
  // Emitted as a placeholder div whose textContent is the untouched diagram
  // source; mermaid.js reads it back at mount time. This must run before the
  // generic fence rule below, which would otherwise swallow these blocks.
  text = text.replace(/```mermaid[ \t]*\n([\s\S]*?)```/g, (_match, diagram: string) => {
    return save(`<div class="mermaid">${escapeHtml(diagram.trim())}</div>`)
  })

  // ── Step 0b: Extract <pre class="pseudocode"> blocks ──
  // Same treatment as mermaid: keep the LaTeX source verbatim for
  // pseudocode.js to parse at mount time. Without this the `$...$` inside the
  // algorithm would be pulled out as article math and the whole block would
  // arrive at the renderer pre-mangled.
  text = text.replace(
    /<pre class="pseudocode">([\s\S]*?)<\/pre>/g,
    (_match, latex: string) => save(`<pre class="pseudocode">${escapeHtml(latex)}</pre>`)
  )

  // ── Step 1: Protect fenced code blocks ──
  text = text.replace(/```[\s\S]*?```/g, (match) => save(match))

  // ── Step 2: Protect inline code ──
  text = text.replace(/`[^`]+`/g, (match) => save(match))

  // ── Step 3: Extract $$...$$ display math ──
  text = text.replace(/\$\$([\s\S]*?)\$\$/g, (_match, formula: string) => {
    try {
      return save(
        katex.renderToString(formula.trim(), { displayMode: true, throwOnError: false })
      )
    } catch {
      return save(`<pre>${formula.trim()}</pre>`)
    }
  })

  // ── Step 4: Extract $...$ inline math (single-line only) ──
  // [^$] ensures we never cross into another $ delimiter;
  // the lookbehind/lookahead prevent matching $$ display blocks
  text = text.replace(/(?<!\$)\$(?!\$)([^$\n]+?)\$(?!\$)/g, (_match, formula: string) => {
    try {
      return save(
        katex.renderToString(formula.trim(), { displayMode: false, throwOnError: false })
      )
    } catch {
      return save(`<code>${formula.trim()}</code>`)
    }
  })

  // ── Step 4.5: Escape any remaining stray $ signs ──
  text = text.replace(/\$/g, '&#36;')

  // ── Step 5: Parse markdown ──
  const renderer = createRenderer()
  let html = marked.parse(text, { renderer, breaks: false }) as string

  // ── Step 6: Restore tokens in reverse order (code + math) ──
  // Restore in reverse so indices don't shift. The replacement must be a
  // function: a string replacement would interpret `$&`, `$'`, `` $` `` and
  // `$1` inside the token (code and Mermaid sources routinely contain `$&`).
  for (let i = tokens.length - 1; i >= 0; i--) {
    html = html.replace(`${PLACEHOLDER_PREFIX}${i}\x00`, () => tokens[i])
  }

  // ── Step 7: Unwrap block placeholders ──
  // Restored block elements land inside the <p> marked created around their
  // placeholder, which is invalid nesting (browsers would split it out and
  // leave empty paragraphs behind). Lift them up to the top level.
  html = html.replace(/<p>\s*(<div class="mermaid">[\s\S]*?<\/div>)\s*<\/p>/g, '$1')
  html = html.replace(/<p>\s*(<pre class="pseudocode">[\s\S]*?<\/pre>)\s*<\/p>/g, '$1')

  return html
}

/**
 * Extract title from markdown (first # heading)
 */
export function extractTitle(markdown: string): string {
  const match = markdown.match(/^#\s+(.+)$/m)
  return match ? match[1].trim() : ''
}
