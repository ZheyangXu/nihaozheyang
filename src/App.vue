<template>
  <div id="app" style="display:flex;min-height:100vh;flex-direction:column;">
    <main style="flex:1;display:flex;align-items:center;justify-content:center;font-size:32px;">
      你好晢旸
      <br/>
      寻寻觅觅, 冷冷清清, 凄凄惨惨戚戚.
    </main>
    <SiteFooter />
  </div>
</template>

<script setup lang="ts">
import SiteFooter from './components/SiteFooter.vue'
</script>

<style>
/* minimal reset */
html,body,#app { margin:0; padding:0; height:100%; }
</style>
