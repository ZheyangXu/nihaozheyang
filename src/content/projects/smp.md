# SMP

## 一、基本原理

Score-Matching Motion Priors（SMP）是一种可重用、模块化的运动先验构造方法，其核心思想是利用预训练的运动扩散模型，通过分数蒸馏采样（Score Distillation Sampling, SDS）将其转化为强化学习中的风格奖励函数。给定一个非结构化的参考运动数据集，首先训练一个与任务和策略无关的运动扩散模型，该模型学习参考运动数据的分布 $p(\mathbf{x})$。扩散模型通过前向扩散过程逐步加噪，并训练去噪网络 $f$ 预测所加噪声，其训练目标为
$$
\mathcal{L}_{\mathrm{simple}} = \mathbb{E}_{i, \mathbf{x}^0, \epsilon}\left[\left\| \epsilon - f(\mathbf{x}^i) \right\|_2^2\right], 
$$
其中 $\mathbf{x}^i = \sqrt{\alpha_i}\mathbf{x}^0 + \sqrt{1 - \alpha_i}\epsilon$，$\epsilon \sim \mathcal{N}(0, \mathbf{I})$，$\alpha_i$ 由噪声调度决定。训练完成后，扩散模型参数被冻结，不再更新。在策略训练阶段，对于仿真角色产生的运动片段 $\hat{\mathbf{x}}$，将其扩散至某个噪声层级 $i$，得到含噪样本 $\mathbf{x}^i$，扩散模型预测噪声 $\hat{\epsilon} = f(\mathbf{x}^i)$。前向扩散过程中实际加入的噪声 $\epsilon$ 与预测噪声 $\hat{\epsilon}$ 之间的残差 $(\epsilon - \hat{\epsilon})$ 指示了将仿真运动拉回参考分布所需的修正方向。SMP 奖励定义为该残差的指数负形式：
$$
r^{\mathrm{smp}} = \exp \left(-\mathbf{w}_s \| \hat{\epsilon} - \epsilon \|_2^2\right), 
$$
其中 $\mathbf{w}_s$ 为权重系数。该奖励在仿真运动与参考分布一致时取得最大值，从而鼓励策略产生自然、类人的行为。

## 二、方法设计

### 2.1 集成分数匹配

由于 SDS 目标对扩散时间步 $i$ 较为敏感，单一时间步的评估会引入较大方差，不利于强化学习中的值函数与优势估计。为此，SMP 采用集成分数匹配（Ensemble Score-Matching, ESM），在预先定义的扩散时间步集合 $\mathbb{K}$ 上聚合多个 SDS 评估。ESM 奖励为
$$
r^{\mathrm{smp}} = \exp \left(-\frac{\mathbf{w}_s}{\| \mathbb{K} \|}\sum_{i \in \mathbb{K}} \| \hat{\epsilon}_i - \epsilon_i \|_2^2\right), 
$$
其中 $\hat{\epsilon}_i = f\left(\sqrt{\alpha_i}\hat{\mathbf{x}}^0 + \sqrt{1 - \alpha_i}\epsilon_i\right)$。论文中采用 $\mathbb{K} = \{22, 15, 8\}$，以平衡不同噪声层级提供的指导信息。此外，对每个时间步的 SDS 误差使用运行均值进行自适应归一化，以降低不同预训练模型和运动风格之间的尺度差异，减少人工调参负担。

### 2.2 生成式状态初始化

参考状态初始化（RSI）通常需要从原始运动数据集中采样初始状态，而 SMP 通过扩散模型的生成能力实现生成式状态初始化（Generative State Initialization, GSI）。具体而言，在训练新策略时，初始状态直接从预训练扩散模型中采样得到，无需保留原始运动数据集。GSI 与 SMP 奖励共享同一个扩散模型，从而使该模型同时充当奖励函数和初始状态分布，显著提升了框架的模块化程度。

### 2.3 风格条件与风格组合

SMP 支持在训练扩散模型时引入风格标签 $c$，从而得到风格条件扩散模型 $f(\mathbf{x}^i, c)$。通过无分类器引导（Classifier-Free Guidance, CFG），可以将通用先验调整为特定风格的先验：
$$
f_{\mathrm{style}} = f(\mathbf{x}^i, \mathbf{0}) + w_{\mathrm{cfg}}(f(\mathbf{x}^i, c_{\mathrm{style}}) - f(\mathbf{x}^i, \mathbf{0})), 
$$
其中 $w_{\mathrm{cfg}}$ 为引导权重。此外，可以通过混合不同风格在 $\epsilon$ 空间的预测来构造新风格，例如对上半身和下半身分别使用不同风格的条件预测：
$$
f_{\mathrm{comp}} = M_{\mathrm{upper}} \odot f(\mathbf{x}^i, c_{\mathrm{upper}}) + M_{\mathrm{lower}} \odot f(\mathbf{x}^i, c_{\mathrm{lower}}), 
$$
其中 $M_{\mathrm{upper}}$ 和 $M_{\mathrm{lower}}$ 为二值掩码。对于差异较大的风格组合，可采用多步分数匹配（Multi-Step Score Matching, MSM），通过部分去噪过程获得更连贯的复合风格。

### 2.4 运动表示与网络结构

运动片段由 $H$ 帧连续状态组成，$\mathbf{x} := (\mathbf{s}_{t-H+2}, \ldots, \mathbf{s}_{t+1})$。运动特征包括根节点在角色局部坐标系下的线速度与角速度、各关节的局部旋转（球关节采用 6D 表示）、以及末端执行器在局部坐标系下的 3D 位置。扩散模型采用两层 Transformer 编码器，使用自适应归一化注入噪声层级和风格条件，窗口长度 $H=10$，扩散步数 $N=50$，参数量约 3M。策略网络与值函数网络均为多层感知机，策略输出高斯动作分布，动作为各关节 PD 控制器的目标位置，球关节目标以 3D 指数映射表示。

## 三、训练流程

训练分为两个阶段。第一阶段，在参考运动数据集上训练运动扩散模型，该阶段完全任务无关，仅学习运动数据的分布。第二阶段，冻结扩散模型，将其作为 SMP 奖励模型，结合任务奖励训练控制策略。在每个时间步 $t$，策略与环境交互产生状态转移，并计算 SMP 奖励 $r_t^{\mathrm{smp}}$ 与任务奖励 $r_t^g$。二者线性组合为复合奖励：
$$
r_t = w^{\mathrm{prior}} r_t^{\mathrm{smp}} + w^g r_t^g.
$$
策略采用近端策略优化（PPO）更新，优势函数通过 GAE($\lambda$) 估计，值函数通过 TD($\lambda$) 更新。训练过程中，扩散模型保持固定，无需持续更新，也无需访问原始运动数据集。初始状态由生成式状态初始化提供。整体流程如算法 1 所示。

## 四、伪代码

<pre class="pseudocode">
\documentclass{article}
\usepackage{algorithm}
\usepackage{algpseudocode}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{bm}

\begin{document}

\begin{algorithm}
\caption{Policy Training with SMP}
\begin{algorithmic}[1]
\Require 可选的风格标签 $c$
\State $f \leftarrow$ 加载预训练扩散模型
\State $\pi \leftarrow$ 初始化策略
\State $V \leftarrow$ 初始化值函数
\State $\mathcal{B} \leftarrow \emptyset$ 初始化回放缓冲区
\While{未结束}

    \For{轨迹 $j = 1, \dots, m$}
        \State 使用策略 $\pi$ 收集轨迹 $\tau^j \leftarrow \{(s_t, \bar{x}_t, a_t, r_t^j)_{t=0}^{T-1}, \bar{s}_T^j, \bar{x}_T\}$
        \For{$t = 0, \dots, T-1$}
            \For{扩散时间步 $i \in \mathbb{K}$}
                \State $\epsilon_i \sim \mathcal{N}(0, I)$
                \State $\hat{\epsilon}_i \leftarrow f(\sqrt{\alpha_i}\bar{x}_{t+1} + \sqrt{1 - \alpha_i}\epsilon_i, c)$
            \EndFor
            \State 根据式 (8) 使用 $\{\epsilon_i, \hat{\epsilon}_i\}_{i \in \mathbb{K}}$ 计算先验奖励 $r_t^{\mathrm{smp}}$
            \State $r_t \leftarrow w^{\mathrm{prior}} r_t^{\mathrm{smp}} + w^g r_t^g$
            \State 将 $r_t$ 记录到 $\tau^j$ 中
        \EndFor
        \State 将 $\tau^j$ 存入 $\mathcal{B}$
    \EndFor
    \State 使用轨迹集合 $\{\tau^j\}_{j=1}^m$ 更新 $V$ 与 $\pi$

\EndWhile
\end{algorithmic}
\end{algorithm}
\end{document}
</pre>
