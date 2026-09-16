import { createRouter, createWebHistory } from 'vue-router'

const router = createRouter({
  history: createWebHistory(),
  routes: [
    {
      path: "/",
      name: "home",
      component: () => import("../pages/ResearchPage.vue"),
    },
    {
      path: "/projects/ame",
      name: "ame",
      component: () => import("../pages/ProjectDetail.vue"),
      props: { projectId: "ame" },
    },
    {
      path: "/projects/amp",
      name: "amp",
      component: () => import("../pages/ProjectDetail.vue"),
      props: { projectId: "amp" },
    },
    {
      path: "/projects/gmr",
      name: "gmr",
      component: () => import("../pages/ProjectDetail.vue"),
      props: { projectId: "gmr" },
    },
    {
      path: "/projects/beyondmimic",
      name: "beyondmimic",
      component: () => import("../pages/ProjectDetail.vue"),
      props: { projectId: "beyondmimic" },
    },
    {
      path: "/projects/luwu",
      name: "luwu",
      component: () => import("../pages/ProjectDetail.vue"),
      props: { projectId: "luwu" },
    },
    {
      path: "/projects/mpc",
      name: "mpc",
      component: () => import("../pages/ProjectDetail.vue"),
      props: { projectId: "mpc" },
    },
    {
      path: "/projects/smp",
      name: "smp",
      component: () => import("../pages/ProjectDetail.vue"),
      props: { projectId: "smp" },
    },
    {
      path: "/projects/humanoid",
      name: "humanoid",
      component: () => import("../pages/ProjectDetail.vue"),
      props: { projectId: "humanoid" },
    },
  ],
});

export default router
