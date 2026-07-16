import { createRouter, createWebHistory } from 'vue-router'

const router = createRouter({
  history: createWebHistory(),
  routes: [
    {
      path: '/',
      name: 'home',
      component: () => import('../pages/ResearchPage.vue'),
    },
    {
      path: '/projects/ame',
      name: 'ame',
      component: () => import('../pages/ProjectDetail.vue'),
      props: { projectId: 'ame' },
    },
    {
      path: '/projects/amp',
      name: 'amp',
      component: () => import('../pages/ProjectDetail.vue'),
      props: { projectId: 'amp' },
    },
    {
      path: '/projects/mimic',
      name: 'mimic',
      component: () => import('../pages/ProjectDetail.vue'),
      props: { projectId: 'tracking' },
    },
    {
      path: '/projects/luwu',
      name: 'luwu',
      component: () => import('../pages/ProjectDetail.vue'),
      props: { projectId: 'luwu' },
    },
    {
      path: '/projects/mpc',
      name: 'mpc',
      component: () => import('../pages/ProjectDetail.vue'),
      props: { projectId: 'mpc' },
    },
  ],
})

export default router
