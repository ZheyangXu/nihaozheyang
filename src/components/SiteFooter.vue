<template>
  <footer class="site-footer">
    <div class="container">
        <span>© {{ year }} 你好晢旸</span>
        <span class="sep">|</span>
        <span>备案号：
          <a href="https://beian.miit.gov.cn" target="_blank" rel="noopener noreferrer">闽ICP备2022004009号-1</a>
        </span>
      </div>
  </footer>
</template>

<script setup lang="ts">
import { ref } from 'vue'
const year = new Date().getFullYear()
</script>

<style scoped>
.site-footer {
  background: #f7f7f8;
  padding: 12px 16px;
  text-align: center;
  font-size: 13px;
  color: #555;
}
.site-footer .container { display:inline-flex; gap:8px; align-items:center }
.site-footer .sep { opacity:0.5 }
</style>
