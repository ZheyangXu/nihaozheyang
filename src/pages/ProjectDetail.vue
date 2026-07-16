<template>
  <div class="project-detail">
    <!-- Back navigation -->
    <div class="project-detail__nav">
      <div class="project-detail__nav-inner">
        <router-link to="/" class="project-detail__back">
          <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
            <path d="M12 4L6 10L12 16" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
          </svg>
          Back to Research
        </router-link>
      </div>
    </div>

    <!-- Loading -->
    <div v-if="loading" class="project-detail__loading">
      <div class="project-detail__spinner"></div>
    </div>

    <!-- Error -->
    <div v-else-if="error" class="project-detail__error">
      <h2>Unable to load project</h2>
      <p>{{ error }}</p>
      <router-link to="/" class="project-detail__error-link">Return to Research</router-link>
    </div>

    <!-- Content -->
    <template v-else>
      <!-- Cover Section -->
      <section class="cover">
        <div class="cover__inner">
          <div class="cover__meta">
            <span class="cover__category">Research</span>
            <span class="cover__separator">·</span>
            <span class="cover__date">{{ projectDate }}</span>
          </div>
          <h1 class="cover__title">{{ projectTitle }}</h1>
        </div>
      </section>

      <!-- Article Body -->
      <section class="article">
        <div class="article__inner">
          <article
            class="article__content"
            v-html="renderedContent"
          ></article>

          <!-- Back link at bottom -->
          <div class="article__footer-nav">
            <router-link to="/" class="project-detail__back">
              <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
                <path d="M12 4L6 10L12 16" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
              </svg>
              Back to Research
            </router-link>
          </div>
        </div>
      </section>
    </template>

    <SiteFooter />
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, watch } from 'vue'
import SiteFooter from '../components/SiteFooter.vue'
import { renderMarkdown, extractTitle } from '../utils/markdown'

const props = defineProps<{
  projectId: string
}>()

const rawMarkdown = ref('')
const loading = ref(true)
const error = ref('')

const projectTitle = computed(() => {
  return extractTitle(rawMarkdown.value) || 'Project'
})

const projectDate = computed(() => {
  return '2025'
})

const renderedContent = computed(() => {
  if (!rawMarkdown.value) return ''
  // Strip the first # heading since we display it in the cover
  const content = rawMarkdown.value.replace(/^#\s+.+$/m, '').trimStart()
  return renderMarkdown(content)
})

async function loadContent() {
  loading.value = true
  error.value = ''
  try {
    const module = await import(`../content/projects/${props.projectId}.md?raw`)
    rawMarkdown.value = module.default
  } catch (e) {
    error.value = `Could not load project "${props.projectId}".`
  } finally {
    loading.value = false
  }
}

onMounted(loadContent)
watch(() => props.projectId, loadContent)
</script>

<style scoped>
.project-detail {
  display: flex;
  flex-direction: column;
  min-height: 100vh;
  background: #fafafa;
}

/* ── Nav bar ── */
.project-detail__nav {
  position: sticky;
  top: 0;
  z-index: 100;
  background: rgba(255, 255, 255, 0.92);
  backdrop-filter: blur(12px);
  border-bottom: 1px solid rgba(0, 0, 0, 0.05);
  padding: 12px 24px;
}

.project-detail__nav-inner {
  max-width: 1200px;
  margin: 0 auto;
}

.project-detail__back {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  font-size: 14px;
  font-weight: 500;
  color: #5f6368;
  text-decoration: none;
  transition: color 0.2s;
}

.project-detail__back:hover {
  color: #1a73e8;
}

/* ── Loading ── */
.project-detail__loading {
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: center;
}

.project-detail__spinner {
  width: 36px;
  height: 36px;
  border: 3px solid #e0e0e0;
  border-top-color: #1a73e8;
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

/* ── Error ── */
.project-detail__error {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  text-align: center;
  padding: 48px 24px;
  color: #5f6368;
}

.project-detail__error h2 {
  font-size: 24px;
  color: #1a1a2e;
  margin: 0 0 8px;
}

.project-detail__error-link {
  display: inline-block;
  margin-top: 16px;
  color: #1a73e8;
  font-weight: 500;
  text-decoration: none;
}

/* ── Cover Section ── */
.cover {
  padding: 64px 24px 48px;
  text-align: center;
  background: #fff;
  border-bottom: 1px solid rgba(0, 0, 0, 0.04);
}

.cover__inner {
  max-width: 800px;
  margin: 0 auto;
}

.cover__meta {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  margin-bottom: 20px;
  font-size: 14px;
  color: #5f6368;
}

.cover__category {
  font-weight: 600;
  letter-spacing: 1.5px;
  text-transform: uppercase;
  color: #1a73e8;
}

.cover__separator {
  opacity: 0.4;
}

.cover__title {
  font-size: clamp(28px, 5vw, 48px);
  font-weight: 700;
  line-height: 1.15;
  letter-spacing: -0.02em;
  color: #1a1a2e;
  margin: 0;
}

/* ── Article ── */
.article {
  flex: 1;
  padding: 48px 24px 80px;
}

.article__inner {
  max-width: 800px;
  margin: 0 auto;
}

/* ── Rendered Markdown Content ── */
.article__content :deep(h2) {
  font-size: 26px;
  font-weight: 700;
  color: #1a1a2e;
  margin: 48px 0 16px;
  letter-spacing: -0.01em;
  padding-bottom: 8px;
  border-bottom: 1px solid rgba(0, 0, 0, 0.06);
}

.article__content :deep(h3) {
  font-size: 20px;
  font-weight: 600;
  color: #1a1a2e;
  margin: 32px 0 12px;
}

.article__content :deep(h4) {
  font-size: 17px;
  font-weight: 600;
  color: #333;
  margin: 24px 0 8px;
}

.article__content :deep(p) {
  font-size: 16px;
  line-height: 1.75;
  color: #333;
  margin: 0 0 16px;
}

.article__content :deep(a) {
  color: #1a73e8;
  text-decoration: none;
}

.article__content :deep(a:hover) {
  text-decoration: underline;
}

/* Images */
.article__content :deep(img) {
  max-width: 100%;
  height: auto;
  border-radius: 12px;
  margin: 24px 0;
  display: block;
}

/* Videos */
.article__content :deep(video) {
  max-width: 100%;
  border-radius: 12px;
  margin: 24px 0;
  display: block;
}

/* Tables */
.article__content :deep(table) {
  width: 100%;
  border-collapse: collapse;
  margin: 24px 0;
  font-size: 14px;
  background: #fff;
  border-radius: 8px;
  overflow: hidden;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.06);
}

.article__content :deep(thead) {
  background: #f5f5f7;
}

.article__content :deep(th) {
  padding: 10px 14px;
  text-align: left;
  font-weight: 600;
  color: #1a1a2e;
  border-bottom: 2px solid #e0e0e0;
}

.article__content :deep(td) {
  padding: 10px 14px;
  border-bottom: 1px solid #f0f0f0;
  color: #333;
}

.article__content :deep(tr:last-child td) {
  border-bottom: none;
}

/* Code blocks */
.article__content :deep(pre) {
  background: #1e1e2e;
  color: #cdd6f4;
  border-radius: 10px;
  padding: 20px 24px;
  overflow-x: auto;
  font-size: 13px;
  line-height: 1.6;
  margin: 24px 0;
}

.article__content :deep(code) {
  font-family: 'SF Mono', 'Fira Code', 'Fira Mono', Menlo, monospace;
}

.article__content :deep(p code),
.article__content :deep(li code) {
  background: rgba(0, 0, 0, 0.05);
  padding: 2px 6px;
  border-radius: 4px;
  font-size: 0.9em;
  color: #d63384;
}

/* Lists */
.article__content :deep(ul),
.article__content :deep(ol) {
  padding-left: 24px;
  margin: 12px 0 20px;
}

.article__content :deep(li) {
  font-size: 16px;
  line-height: 1.75;
  color: #333;
  margin-bottom: 6px;
}

/* Blockquotes */
.article__content :deep(blockquote) {
  border-left: 3px solid #1a73e8;
  margin: 24px 0;
  padding: 12px 20px;
  background: rgba(26, 115, 232, 0.04);
  border-radius: 0 8px 8px 0;
}

.article__content :deep(blockquote p) {
  margin: 0;
  color: #555;
}

/* Horizontal rules */
.article__content :deep(hr) {
  border: none;
  border-top: 1px solid rgba(0, 0, 0, 0.08);
  margin: 48px 0;
}

/* KaTeX math */
.article__content :deep(.katex-display) {
  margin: 24px 0;
  overflow-x: auto;
  overflow-y: hidden;
}

.article__content :deep(.katex) {
  font-size: 1.1em;
}

/* Article footer nav */
.article__footer-nav {
  margin-top: 64px;
  padding-top: 32px;
  border-top: 1px solid rgba(0, 0, 0, 0.06);
}

/* ── Responsive ── */
@media (max-width: 640px) {
  .cover {
    padding: 40px 16px 32px;
  }

  .article {
    padding: 32px 16px 64px;
  }

  .article__content :deep(h2) {
    font-size: 22px;
  }

  .article__content :deep(h3) {
    font-size: 18px;
  }

  .article__content :deep(p),
  .article__content :deep(li) {
    font-size: 15px;
  }

  .article__content :deep(table) {
    font-size: 12px;
  }

  .article__content :deep(th),
  .article__content :deep(td) {
    padding: 8px 10px;
  }
}
</style>
