# CMoE

## 一、基本原理

CMoE（Contrastive Mixture of Experts）是一种面向人形机器人多地形运动控制的单阶段强化学习框架。其核心思想在于将混合专家（Mixture of Experts, MoE）结构与对比学习相结合，以解决传统 Vanilla MoE 中门控网络对不同地形激活分布近乎均匀、专家专业化不足的问题。在 Vanilla MoE 中，门控网络往往无法根据环境特征动态地分配专家，导致专家在不同地形上表现出相似的激活模式，从而削弱了模型的表达能力与泛化能力。CMoE 通过引入对比学习约束，最大化同一地形内专家激活分布的一致性，同时最小化不同地形间专家激活分布的相似性，迫使专家网络针对不同地形类型形成专门化分工。

具体而言，给定多地形运动控制任务，将其建模为马尔可夫决策过程 \(\langle S, A, T, R, \gamma \rangle\)，其中 \(S\) 为状态空间，\(A\) 为动作空间，\(T\) 为状态转移概率，\(R(s, a)\) 为奖励函数，\(\gamma \in [0, 1]\) 为折扣因子。采用近端策略优化（PPO）学习最优策略 \(\pi^*\)，其目标为最大化期望折扣回报：
$$
\pi^* = \arg \max \mathbb{E}\left[\sum_{t = 0}^{\infty}\gamma^t R(s_t, a_t)\right].
$$
CMoE 在 PPO 框架下，将 actor 与 critic 网络均扩展为 MoE 结构，并通过地形对比学习优化门控网络，使专家能够根据地形特征自适应激活。

## 二、方法设计

### 2.1 信息编码与状态估计

机器人的本体感知观测为
$$
\mathbf{o}_t = [\omega_t, g_t, c_v^t, \theta_t, \dot{\theta}_t, a_{t - 1}], 
$$
其中包括角速度 \(\omega_t\)、重力方向 \(g_t\)、速度指令 \(c_v^t\)、关节角度 \(\theta_t\)、关节速度 \(\dot{\theta}_t\) 以及上一时刻动作 \(a_{t-1}\)。为从历史本体感知中估计机器人身体状态并提取环境特征，CMoE 设计了一个上下文状态蒸馏模型，包含两个独立估计器。第一个估计器采用 \(\beta\)-变分自编码器（\(\beta\)-VAE），将历史观测 \(\mathbf{o}_t^H\) 编码为机器人身体速度 \(\mathbf{v}_t\) 与隐表示 \(\mathbf{z}_t^H\)，再解码为下一时刻观测 \(\tilde{\mathbf{o}}_{t+1}\)。其混合损失函数为
$$
\mathcal{L}_{\mathrm{CS}} = \mathrm{MSE}(\tilde{\mathbf{v}}_t, \mathbf{v}_t) + \mathcal{L}_{\mathrm{VAE}}, 
$$
其中身体速度损失促使估计器准确预测机器人运动速度，VAE 重构损失为
$$
\mathcal{L}_{\mathrm{VAE}} = \mathrm{MSE}(\tilde{\mathbf{o}}_{t + 1}, \mathbf{o}_{t + 1}) + \beta D_{\mathrm{KL}}(q(\mathbf{z}_t^H\mid \mathbf{o}_t^H)\parallel p(\mathbf{z}_t^H)), 
$$
这里 \(\tilde{\mathbf{o}}_{t+1}\) 为重构观测，\(q(\mathbf{z}_t^H\mid \mathbf{o}_t^H)\) 为后验分布，\(p(\mathbf{z}_t^H)\) 为标准高斯先验。第二个估计器使用自编码器（AE）从高程图中提取地形特征，其损失函数为
$$
\mathcal{L}_{\mathrm{AE}} = \mathrm{MSE}(\tilde{\mathbf{e}}_t, \mathbf{e}_t), 
$$
其中 \(\mathbf{e}_t\) 为仿真器提供的真实高程图。通过上述编码，机器人能够同时获得显式的身体状态估计与隐式的地形环境表示。

### 2.2 混合专家策略

为缓解多任务强化学习中的梯度冲突，CMoE 在 actor 与 critic 网络中均引入 MoE 架构。每个专家模块包含独立的 actor-critic 对，且每个 critic 仅使用特权观测评估其对应的 actor。所有专家接收估计的身体速度 \(\mathbf{v}_t^p\)、隐式上下文状态变量 \(\mathbf{z}_t^E\) 与 \(\mathbf{z}_t^H\)、当前观测 \(\mathbf{o}_t^c\) 以及高程图 \(\mathbf{e}_t\)，并输出动作或价值估计。为保证策略评估与动作生成的一致性，actor 与 critic 的 MoE 组件共享同一个门控网络。最终输出为各专家输出的加权和，经 softmax 归一化后得到：
$$
\mu_{\mathrm{weighted}} = \sum_{i = 1}^{N}\mathrm{softmax}(g_i)\cdot \mu_i, 
$$
其中 \(\mu_i\) 为第 \(i\) 个专家的输出，\(g_i\) 为对应的专家激活值。通过门控网络，机器人能够根据当前地形特征动态地组合不同专家的能力。

### 2.3 地形对比学习

地形对比学习是 CMoE 的核心创新，旨在将地形信息编码为隐特征，并增强其与 MoE 门控网络之间的关联。具体地，首先采用两个多层感知机分别将门控输出与高程图变换到相同维度，得到 \(g_t^z\) 与 \(e_t^z\)。在对比学习过程中，若一对 \(\langle g^z, e^z\rangle\) 属于同一轨迹，则视为正样本；否则视为负样本。优化过程受 SwAV 启发，通过原型向量计算聚类分配概率。对原型进行 \(\mathcal{L}_2\) 归一化得到矩阵 \(\mathbf{E} = \{\tilde{\mathbf{e}}_1, \dots, \tilde{\mathbf{e}}_K\}\)，然后对源向量与目标向量同所有原型的点积做 softmax：
$$
\mathbf{p}_t^g = \frac{\exp(\frac{1}{2}g_t^z{}^\top\mathbf{e}_k)}{\sum_{k'}\exp(\frac{1}{2}g_t^z{}^\top\mathbf{e}_{k'})}, \quad
\mathbf{p}_t^e = \frac{\exp(\frac{1}{2}e_t^z{}^\top\mathbf{e}_k)}{\sum_{k'}\exp(\frac{1}{2}e_t^z{}^\top\mathbf{e}_{k'})}, 
$$
其中 \(\mathbf{p}_t^g\) 与 \(\mathbf{p}_t^e\) 分别为门控输出与地形特征映射到第 \(k\) 个聚类簇的预测概率。为避免平凡解，采用 Sinkhorn-Knopp 算法对两个编码器的预测概率进行归一化，得到聚类分配 \(\mathbf{q}_t^g\) 与 \(\mathbf{q}_t^e\)。最终对比学习目标为最大化匹配精度：
$$
\mathcal{I}^{\mathrm{SwAV}} = -\frac{1}{2H}\sum_{t = 1}^{H}(\mathbf{q}_t^{\mathrm{g}}\log \mathbf{p}_t^{\mathrm{e}} + \mathbf{q}_t^{\mathrm{e}}\log \mathbf{p}_t^{\mathrm{g}}).
$$
该损失促使门控网络输出的专家激活分布与地形特征在聚类空间中保持一致，从而使专家能够针对不同地形形成专门化响应。

## 三、训练流程

CMoE 采用单阶段端到端训练。训练开始时，在仿真环境中并行运行大量环境实例，收集机器人交互轨迹。对于每条轨迹，首先利用 VAE 与 AE 估计身体速度、隐状态与地形特征，并将这些信息与当前观测融合后输入 MoE 策略网络。MoE 门控网络根据地形特征输出专家激活值，各专家输出加权融合后得到动作分布。与此同时，门控输出与地形特征被送入对比学习模块，计算 SwAV 对比损失。策略优化采用 PPO，使用 GAE(\(\lambda\)) 估计优势函数，值函数通过 TD(\(\lambda\)) 更新。总损失由 PPO 损失、值函数损失、VAE/AE 重构损失以及对比学习损失组成，通过梯度下降联合更新策略网络、值函数网络、门控网络、VAE 编码器、AE 编码器与对比学习原型。训练过程中，采用课程学习机制逐步增加地形难度与速度指令范围，并施加域随机化以增强 sim-to-real 迁移能力。整个训练无需分阶段蒸馏，直接在多种地形上同时学习，最终得到单一策略即可在混合地形上实现稳健运动。

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
\caption{CMoE 训练流程}
\begin{algorithmic}[1]
\Require 仿真环境集合 $\mathcal{E}$，地形集合 $\mathcal{T}$，专家数量 $N$，原型数量 $K$
\State 初始化 VAE 编码器 $q(\mathbf{z}_t^H|\mathbf{o}_t^H)$、AE 编码器、MoE actor-critic 网络、门控网络、对比学习原型 $\mathbf{E}$
\State 初始化 PPO 策略 $\pi$ 与值函数 $V$
\While{未收敛}

    \State 在 $\mathcal{E}$ 中并行采样多条轨迹 $\{\tau_j\}$
    \For{每条轨迹 $\tau_j$ 中的每个时间步 $t$}
        \State 从历史观测 $\mathbf{o}_t^H$ 经 VAE 估计身体速度 $\mathbf{v}_t$ 与隐状态 $\mathbf{z}_t^H$
        \State 从高程图 $\mathbf{e}_t$ 经 AE 提取地形特征
        \State 将 $\mathbf{v}_t, \mathbf{z}_t^H, \mathbf{o}_t^c, \mathbf{e}_t$ 输入 MoE 网络
        \State 门控网络输出专家激活 $g_i$，计算加权动作 $\mu_{\mathrm{weighted}} = \sum_{i=1}^N \mathrm{softmax}(g_i)\cdot \mu_i$
        \State 执行动作，获得奖励 $r_t$ 与下一观测
        \State 将门控输出 $g_t^z$ 与地形特征 $e_t^z$ 映射至同一维度
        \State 计算聚类概率 $\mathbf{p}_t^g, \mathbf{p}_t^e$，经 Sinkhorn-Knopp 得到 $\mathbf{q}_t^g, \mathbf{q}_t^e$
        \State 计算对比损失 $\mathcal{I}^{\mathrm{SwAV}} = -\frac{1}{2H}\sum_{t=1}^H (\mathbf{q}_t^g \log \mathbf{p}_t^e + \mathbf{q}_t^e \log \mathbf{p}_t^g)$
    \EndFor
    \State 计算 PPO 损失、值函数损失、VAE 损失 $\mathcal{L}_{\mathrm{VAE}}$、AE 损失 $\mathcal{L}_{\mathrm{AE}}$ 与对比损失
    \State 联合更新策略、值函数、门控网络、VAE 编码器、AE 编码器与对比原型

\EndWhile
\State \Return 训练完成的 CMoE 策略
\end{algorithmic}
\end{algorithm}
\end{document}
</pre>
