<template>
  <div class="research-page">
    <!-- Cover / Hero Section -->
    <section class="cover">
      <div class="cover__inner">
        <span class="cover__eyebrow">X</span>
        <h1 class="cover__title">
          人形和四足机器人的运动控制
        </h1>
      </div>
    </section>

    <!-- Projects Section -->
    <section class="projects">
      <div class="projects__container">
        <div class="projects__header">
          <h2 class="projects__title">项目</h2>
          <p class="projects__subtitle">
            机器人的运动控制
          </p>
        </div>

        <div class="projects__grid">
          <ProjectCard
            v-for="project in projects"
            :key="project.title"
            :title="project.title"
            :description="project.description"
            :image-color="project.imageColor"
            :link="project.link"
          />
        </div>
      </div>
    </section>
  </div>
</template>

<script setup lang="ts">
import ProjectCard from '../components/ProjectCard.vue'

interface Project {
  title: string
  description: string
  imageColor: string
  link: string
}

const projects: Project[] = [
  {
    title: 'AME',
    description:
      'AME 算法通过基于注意力的地图编码架构来处理高程地图以提取地形感知特征, 用于控制决策, 来提升机器人在复杂地形上的运动能力.',
    imageColor: 'linear-gradient(135deg, #e8734a 0%, #f59e6b 50%, #f7c9a8 100%)',
    link: '/projects/ame',
  },
  {
    title: 'AMP',
    description:
      'AMP(Adversarial Motion Priors, 对抗性运动先验) 是一种基于 GAN 和强化学习RL的运动控制框架, 核心是通过对抗模仿从非结构化运动数据中学习风格特征, 让物理模拟角色或真实机器人在完成任务的同时, 呈现自然, 风格化的运动.',
    imageColor: 'linear-gradient(135deg, #4a90d9 0%, #67b8f7 50%, #a0d2ff 100%)',
    link: '/projects/amp',
  },
  {
    title: 'DeepMimic',
    description:
      'DeepMimic: 一个基于引导扩散的现实世界框架, 旨在从人类动作中学习并实现多样化, 自然的人形机器人全身控制.',
    imageColor: 'linear-gradient(135deg, #2ecc71 0%, #58d68d 50%, #82e0aa 100%)',
    link: '/projects/mimic',
  },
  {
    title: ' PID-DE-MPC',
    description:
      '通过 PID 事件驱动触发的减少 MPC 中的 QP 求解次数, 来提升 MPC 的运行速度. 同时通过抗扰动模块来提升机器人的抗扰动能力.',
    imageColor: 'linear-gradient(135deg, #e74c3c 0%, #f0736a 50%, #f5a097 100%)',
    link: '/projects/mpc',
  },
  {
    title: '陆吾',
    description:
      '使用 IsaacLab 来实现 Cyberdog2 的四足机器人 Locomotion 控制, 通过强化学习来提升四足机器人的运动能力.',
    imageColor: 'linear-gradient(135deg, #7b4fbf 0%, #9b7fd4 50%, #c4b0e8 100%)',
    link: '/projects/luwu',
  },
  
]
</script>

<style scoped>
.research-page {
  flex: 1;
}

/* ── Cover Section ── */
.cover {
  padding: 80px 24px 56px;
  text-align: center;
  background: #fff;
  border-bottom: 1px solid rgba(0, 0, 0, 0.04);
}

.cover__inner {
  max-width: 800px;
  margin: 0 auto;
}

.cover__eyebrow {
  display: block;
  font-size: 14px;
  font-weight: 600;
  letter-spacing: 2px;
  text-transform: uppercase;
  color: #1a73e8;
  margin-bottom: 24px;
}

.cover__title {
  font-size: clamp(28px, 5vw, 52px);
  font-weight: 700;
  line-height: 1.15;
  letter-spacing: -0.02em;
  color: #1a1a2e;
  margin: 0 auto;
  max-width: 720px;
}

/* ── Projects Section ── */
.projects {
  padding: 64px 24px 80px;
}

.projects__container {
  max-width: 1280px;
  margin: 0 auto;
}

.projects__header {
  margin-bottom: 48px;
}

.projects__title {
  font-size: 32px;
  font-weight: 700;
  color: #1a1a2e;
  margin: 0 0 12px;
  letter-spacing: -0.01em;
}

.projects__subtitle {
  font-size: 17px;
  color: #5f6368;
  margin: 0;
  line-height: 1.5;
}

/* ── Cards Grid ── */
.projects__grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 24px;
}

@media (max-width: 1024px) {
  .projects__grid {
    grid-template-columns: repeat(2, 1fr);
  }

  .cover {
    padding: 64px 20px 48px;
  }
}

@media (max-width: 640px) {
  .projects__grid {
    grid-template-columns: 1fr;
  }

  .cover {
    padding: 48px 16px 40px;
  }

  .projects {
    padding: 40px 16px 64px;
  }
}
</style>
