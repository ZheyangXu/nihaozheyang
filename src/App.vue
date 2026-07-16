<template>
  <div id="app">
    <router-view />
    <SiteFooter />
  </div>
</template>

<script setup lang="ts">
import SiteFooter from './components/SiteFooter.vue'
</script>

<style>
/* ── Reset ── */
*,
*::before,
*::after {
  box-sizing: border-box;
}
html,
body,
#app {
  margin: 0;
  padding: 0;
  min-height: 100vh;
}

body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto,
    'Helvetica Neue', Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  color: #1a1a2e;
  background: #fafafa;
}

#app {
  display: flex;
  flex-direction: column;
  min-height: 100vh;
}
</style>
