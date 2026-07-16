import { marked } from 'marked'
import katex from 'katex'

const PLACEHOLDER_PREFIX = '\x00MD_'

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
 * 1. Protect fenced code blocks (```...```)
 * 2. Protect inline code (`...`)
 * 3. Extract $$...$$ display math → render with KaTeX
 * 4. Extract $...$ inline math → render with KaTeX
 * 5. Parse remaining markdown with marked
 * 6. Restore KaTeX HTML
 * 7. Restore code blocks
 */
export function renderMarkdown(markdown: string): string {
  const tokens: string[] = []

  function save(html: string): string {
    const idx = tokens.length
    tokens.push(html)
    return `${PLACEHOLDER_PREFIX}${idx}\x00`
  }

  let text = markdown

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
  // Restore in reverse so indices don't shift
  for (let i = tokens.length - 1; i >= 0; i--) {
    html = html.replace(`${PLACEHOLDER_PREFIX}${i}\x00`, tokens[i])
  }

  return html
}

/**
 * Extract title from markdown (first # heading)
 */
export function extractTitle(markdown: string): string {
  const match = markdown.match(/^#\s+(.+)$/m)
  return match ? match[1].trim() : ''
}
