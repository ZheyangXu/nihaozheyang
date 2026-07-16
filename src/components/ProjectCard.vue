<template>
  <article class="project-card">
    <router-link
      v-if="isInternalLink"
      :to="link"
      :aria-label="`Learn more about ${title}`"
      class="project-card__overlay-link"
      tabindex="-1"
    >
    </router-link>
    <a
      v-else-if="link"
      :href="link"
      :aria-label="`Learn more about ${title}`"
      class="project-card__overlay-link"
      tabindex="-1"
    >
    </a>
    <div class="project-card__inner">
      <!-- Media / Image -->
      <div class="project-card__media">
        <div class="project-card__image" :style="{ background: imageColor }">
          <span class="project-card__image-text">{{ title }}</span>
        </div>
      </div>

      <!-- Content -->
      <div class="project-card__content">
        <h3 class="project-card__title">{{ title }}</h3>
        <p v-if="description" class="project-card__text">{{ description }}</p>
      </div>

      <!-- Footer -->
      <div v-if="link" class="project-card__footer">
        <router-link v-if="isInternalLink" :to="link" class="project-card__link">
          Learn more
          <svg class="project-card__link-icon" width="16" height="16" viewBox="0 0 16 16" fill="none">
            <path d="M6 4L10 8L6 12" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
          </svg>
        </router-link>
        <a v-else :href="link" class="project-card__link">
          Learn more
          <svg class="project-card__link-icon" width="16" height="16" viewBox="0 0 16 16" fill="none">
            <path d="M6 4L10 8L6 12" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
          </svg>
        </a>
      </div>
    </div>
  </article>
</template>

<script setup lang="ts">
import { computed } from 'vue'

interface ProjectCardProps {
  /** Card title */
  title: string
  /** Card description */
  description?: string
  /** CSS background value for the placeholder image */
  imageColor?: string
  /** Link URL for the card */
  link?: string
}

const props = withDefaults(defineProps<ProjectCardProps>(), {
  description: '',
  imageColor: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
  link: '',
})

const isInternalLink = computed(() => {
  return props.link.startsWith('/')
})
</script>

<style scoped>
.project-card {
  position: relative;
  border-radius: 16px;
  overflow: hidden;
  background: #fff;
  border: 1px solid rgba(0, 0, 0, 0.06);
  transition: transform 0.3s ease, box-shadow 0.3s ease;
  display: flex;
  flex-direction: column;
}

.project-card:has(.project-card__overlay-link:hover) {
  transform: translateY(-4px);
  box-shadow: 0 12px 40px rgba(0, 0, 0, 0.1);
}

.project-card__overlay-link {
  position: absolute;
  inset: 0;
  z-index: 2;
}

.project-card__inner {
  display: flex;
  flex-direction: column;
  height: 100%;
}

/* Media / Image */
.project-card__media {
  position: relative;
  aspect-ratio: 16 / 9;
  overflow: hidden;
}

.project-card__media::after {
  content: '';
  position: absolute;
  inset: 0;
  background: linear-gradient(
    to bottom,
    transparent 50%,
    rgba(0, 0, 0, 0.35) 100%
  );
  pointer-events: none;
  z-index: 1;
}

.project-card__image {
  width: 100%;
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
}

.project-card__image-text {
  font-size: 28px;
  font-weight: 700;
  color: rgba(255, 255, 255, 0.9);
  text-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
  letter-spacing: 1px;
}

/* Content */
.project-card__content {
  padding: 20px 24px 12px;
  flex: 1;
  position: relative;
  z-index: 1;
}

.project-card__title {
  font-size: 20px;
  font-weight: 600;
  color: #1a1a2e;
  margin: 0 0 8px;
  line-height: 1.3;
  letter-spacing: -0.01em;
}

.project-card__text {
  font-size: 15px;
  color: #5f6368;
  line-height: 1.55;
  margin: 0;
}

/* Footer */
.project-card__footer {
  padding: 8px 24px 20px;
  position: relative;
  z-index: 1;
}

.project-card__link {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  font-size: 14px;
  font-weight: 500;
  color: #1a73e8;
  position: relative;
  z-index: 3;
  text-decoration: none;
  transition: gap 0.2s ease;
}

.project-card:has(.project-card__overlay-link:hover) .project-card__link {
  gap: 8px;
}

.project-card__link-icon {
  transition: transform 0.2s ease;
}

.project-card:has(.project-card__overlay-link:hover) .project-card__link-icon {
  transform: translateX(2px);
}
</style>
