# 运动跟踪 (Motion Tracking)

基于 PPO 强化学习的人形机器人全身运动跟踪任务。策略从动作捕捉数据中学习关节级控制，驱动机器人（Unitree G1）实时复现参考运动轨迹。

## 训练演示

<video width="1080" controls src="../public/projects/tracking_train_100k.mp4"></video>

---

## 1. 符号约定

- 参考 anchor body 世界位姿：$\mathbf{p}_a^{\text{ref}}, \mathbf{q}_a^{\text{ref}}$
- 机器人 anchor body 世界位姿：$\mathbf{p}_a^{\text{robot}}, \mathbf{q}_a^{\text{robot}}$
- 参考 body 世界位姿/速度：$\mathbf{p}_b^{\text{ref}}, \mathbf{q}_b^{\text{ref}}, \mathbf{v}_b^{\text{ref}}, \boldsymbol{\omega}_b^{\text{ref}}$
- 机器人 body 世界位姿/速度：$\mathbf{p}_b^{\text{robot}}, \mathbf{q}_b^{\text{robot}}, \mathbf{v}_b^{\text{robot}}, \boldsymbol{\omega}_b^{\text{robot}}$
- 相对 body 参考位姿（在 anchor 系下表示）：$\mathbf{p}_b^{\text{rel}}, \mathbf{q}_b^{\text{rel}}$
- 关节位置/速度：$\mathbf{q}, \dot{\mathbf{q}}$
- 默认关节位置/速度：$\mathbf{q}_0, \dot{\mathbf{q}}_0$

强化学习每步总回报为加权和：

$$
R_t = \sum_k w_k \cdot r_k
$$

其中 $w_k$ 来自 `RewardTermCfg` 的 `weight`，跟踪类奖励均采用指数平方误差形式 $r = \exp(-e^2 / \sigma^2)$。

---

## 2. Commands（命令）

### 2.1 MotionCommand

从 `.npz` 动作捕捉文件加载参考运动轨迹，提供每时刻的关节位置/速度、body 位姿/速度作为跟踪目标。

| 参数 | 值 | 含义 |
| --- | --- | --- |
| `anchor_body_name` | `torso_link` | 根部参考坐标系原点 |
| `body_names` | 14 个关键 body | pelvis / 左右髋、膝、踝 / torso / 左右肩、肘、腕 |
| `resampling_time_range` | $(10^9, 10^9)$ | 命令永不超时自动重采样 |

命令输出为参考关节状态的拼接向量：

$$
\mathbf{c} = [\mathbf{q}_{\text{ref}}, \dot{\mathbf{q}}_{\text{ref}}]
$$

### 2.2 运动采样模式

| 模式 | 采样分布 | 说明 |
| --- | --- | --- |
| `uniform` | $t \sim \text{Uniform}(0, T)$ | 均匀随机采样，最大熵 |
| `adaptive` | $P(i) \propto \text{bin\_failed}[i] + \lambda_u / K$ | 根据历史失败率自适应重采样困难片段 |
| `start` | $t = 0$ | 始终从起始帧开始（play/eval 模式） |

**自适应采样公式：**

$$
P(i) = \frac{\text{bin\_failed}[i] + \lambda_u / K}{\sum_{j=1}^K \left(\text{bin\_failed}[j] + \lambda_u / K\right)}
$$

其中 $\lambda_u = 0.1$ 为均匀混合比，$K$ 为总 bin 数。采样概率经卷积核 $[\lambda^k]$（$\lambda = 0.8$）平滑后归一化：

$$
P_{\text{smooth}}(i) = \frac{(P * \kappa)(i)}{\sum_j (P * \kappa)(j)}, \quad \kappa = [1, \lambda, \lambda^2, \dots]
$$

失败直方图以指数移动平均更新：

$$
\text{bin\_failed}[i] \leftarrow \alpha \cdot \text{current\_bin\_failed}[i] + (1 - \alpha) \cdot \text{bin\_failed}[i], \quad \alpha = 0.001
$$

### 2.3 重采样时的域随机化

采样新的时间步后，对参考状态施加均匀噪声以增强鲁棒性：

**根位姿扰动：**

$$
\mathbf{p}_{\text{root}}' = \mathbf{p}_{\text{root}} + \boldsymbol{\epsilon}_p, \quad \boldsymbol{\epsilon}_p \sim \mathcal{U}(\text{pose\_range}_{xyz})
$$

$$
\mathbf{q}_{\text{root}}' = \mathbf{q}(\boldsymbol{\epsilon}_r) \otimes \mathbf{q}_{\text{root}}, \quad \boldsymbol{\epsilon}_r \sim \mathcal{U}(\text{pose\_range}_{rpy})
$$

**根速度扰动：**

$$
\mathbf{v}_{\text{root}}' = \mathbf{v}_{\text{root}} + \boldsymbol{\epsilon}_v, \quad \boldsymbol{\omega}_{\text{root}}' = \boldsymbol{\omega}_{\text{root}} + \boldsymbol{\epsilon}_\omega
$$

$$
\boldsymbol{\epsilon}_{v,\omega} \sim \mathcal{U}(\text{velocity\_range})
$$

**关节位置扰动与限位：**

$$
\mathbf{q}' = \text{clip}\left(\mathbf{q} + \boldsymbol{\epsilon}_q,\; \mathbf{q}_{\text{min}},\; \mathbf{q}_{\text{max}}\right), \quad \boldsymbol{\epsilon}_q \sim \mathcal{U}(-0.1, 0.1)
$$

---

## 3. Observations（观测）

环境包含两个观测组：`actor` 与 `critic`。Actor 组启用 `enable_corruption=True` 进行观测噪声增强；Critic 组使用无噪声的干净状态并额外包含 body 位姿信息。

### 3.1 Actor 观测（带噪声增强）

| 观测项 | 物理含义 | 噪声分布 |
| --- | --- | --- |
| `command` | 参考关节状态 $[\mathbf{q}_{\text{ref}}, \dot{\mathbf{q}}_{\text{ref}}]$ | 无 |
| `motion_anchor_pos_b` | anchor 相对位置误差（robot base 系） | $\mathcal{U}(-0.25, 0.25)$ |
| `motion_anchor_ori_b` | anchor 相对姿态（旋转矩阵前 2 列，6D） | $\mathcal{U}(-0.05, 0.05)$ |
| `base_lin_vel` | 基座线速度 $\mathbf{v}_{\text{base}}$（IMU） | $\mathcal{U}(-0.5, 0.5)$ |
| `base_ang_vel` | 基座角速度 $\boldsymbol{\omega}_{\text{base}}$（IMU） | $\mathcal{U}(-0.5, 0.5)$ |
| `joint_pos` | 相对关节位置 $\mathbf{q} - \mathbf{q}_0$（biased） | $\mathcal{U}(-0.01, 0.01)$ |
| `joint_vel` | 相对关节速度 $\dot{\mathbf{q}} - \dot{\mathbf{q}}_0$ | $\mathcal{U}(-0.5, 0.5)$ |
| `actions` | 上一控制步动作 $\mathbf{a}_{t-1}$ | 无 |

### 3.2 Critic 观测（无噪声）

Critic 观测与 Actor 共享基础项，并额外包含：

| 额外观测项 | 物理含义 |
| --- | --- |
| `body_pos` | 所有 key body 在 robot base 系下的位置 $\mathbf{p}_{\text{body}}^{\text{robot}, b}$ |
| `body_ori` | 所有 key body 在 robot base 系下的姿态 $\mathbf{R}_{\text{body}}^{\text{robot}, b}$（取前 2 列，6D） |

### 3.3 坐标变换

观测中的相对量通过坐标变换计算，以 anchor 位置为例：

$$
\mathbf{p}_a^{\text{error}, b} = \text{transform}^{-1}\left(\mathbf{p}_a^{\text{robot}}, \mathbf{q}_a^{\text{robot}} \;\middle|\; \mathbf{p}_a^{\text{ref}}, \mathbf{q}_a^{\text{ref}}\right)
$$

即将世界系下参考与机器人 anchor 的位姿差，变换到机器人 base 系下表示。

---

## 4. Actions（动作）

动作项为 `JointPositionAction`，控制机器人全部关节。

### 4.1 动作映射

RL 策略输出原始动作 $\mathbf{a}_{\text{raw}}$，经仿射变换得到目标关节位置：

$$
\mathbf{a}_{\text{proc}} = \mathbf{a}_{\text{raw}} \odot \mathbf{s} + \mathbf{o}
$$

本任务配置：

- `scale = 0.25`（G1_23Dof 使用 `G1_23DOF_ACTION_SCALE`）
- `use_default_offset = True`，即 $\mathbf{o} = \mathbf{q}_0$

每个关节目标位置为：

$$
q_i^{\text{target}} = q_{0, i} + s_i \cdot a_i^{\text{raw}}
$$

目标位置通过 PD 控制器下发到 MuJoCo 仿真器。

### 4.2 控制频率

| 参数 | 值 | 说明 |
| --- | --- | --- |
| MuJoCo 物理步长 | 0.005s | 200Hz 物理仿真 |
| 控制间隔 (decimation) | 4 | 每 4 个物理步执行一次控制 |
| 实际控制频率 | 50Hz | $\Delta t = 0.02\text{s}$ |

---

## 5. Rewards（奖励函数）

所有跟踪奖励采用指数平方误差形式 $r = \exp(-e^2 / \sigma^2)$。

| 奖励项 | 权重 $w_k$ | $\sigma$ | 物理公式 |
| --- | ---: | ---: | --- |
| `motion_global_root_pos` | $+0.5$ | $0.3$ | $\exp\!\left(-\frac{\|\mathbf{p}_a^{\text{ref}} - \mathbf{p}_a^{\text{robot}}\|^2}{\sigma^2}\right)$ |
| `motion_global_root_ori` | $+0.5$ | $0.4$ | $\exp\!\left(-\frac{e_{\text{quat}}(\mathbf{q}_a^{\text{ref}}, \mathbf{q}_a^{\text{robot}})^2}{\sigma^2}\right)$ |
| `motion_body_pos` | $+1.0$ | $0.3$ | $\exp\!\left(-\frac{\frac{1}{N_B}\sum_{b}\|\mathbf{p}_b^{\text{rel}} - \mathbf{p}_b^{\text{robot}}\|^2}{\sigma^2}\right)$ |
| `motion_body_ori` | $+1.0$ | $0.4$ | $\exp\!\left(-\frac{\frac{1}{N_B}\sum_{b} e_{\text{quat}}(\mathbf{q}_b^{\text{rel}}, \mathbf{q}_b^{\text{robot}})^2}{\sigma^2}\right)$ |
| `motion_body_lin_vel` | $+1.0$ | $1.0$ | $\exp\!\left(-\frac{\frac{1}{N_B}\sum_{b}\|\mathbf{v}_b^{\text{ref}} - \mathbf{v}_b^{\text{robot}}\|^2}{\sigma^2}\right)$ |
| `motion_body_ang_vel` | $+1.0$ | $3.14$ | $\exp\!\left(-\frac{\frac{1}{N_B}\sum_{b}\|\boldsymbol{\omega}_b^{\text{ref}} - \boldsymbol{\omega}_b^{\text{robot}}\|^2}{\sigma^2}\right)$ |
| `action_rate_l2` | $-0.1$ | — | $-\|\mathbf{a}_t - \mathbf{a}_{t-1}\|^2$ |
| `joint_limit` | $-10.0$ | — | $-\sum_i\big[\max(q_i^{\text{min}} - q_i, 0) + \max(q_i - q_i^{\text{max}}, 0)\big]$ |
| `self_collisions` | $-10.0$ | — | $-\sum_{h} \mathbb{1}\!\left(\|\mathbf{F}_{\text{contact}}^{(h)}\| > 10\text{N}\right)$ |

其中 $e_{\text{quat}}(\cdot, \cdot)$ 为四元数误差幅度，$N_B = 14$ 为 key body 数量。

### 5.1 相对 body 位姿的构造

为避免全局漂移带来的奖励信号失真，body 位置/姿态跟踪在 anchor-relative 坐标系下计算。相对参考位姿的构造如下：

$$
\mathbf{p}_b^{\text{rel}} \gets \Delta\mathbf{p} + \mathbf{R}(\Delta\mathbf{q}_{\text{yaw}}) \cdot \left(\mathbf{p}_b^{\text{ref}} - \mathbf{p}_a^{\text{ref}}\right)
$$

$$
\mathbf{q}_b^{\text{rel}} \gets \mathbf{q}(\Delta\mathbf{q}_{\text{yaw}}) \otimes \mathbf{q}_b^{\text{ref}}
$$

其中 $\Delta\mathbf{p}$ 和 $\Delta\mathbf{q}_{\text{yaw}}$ 为 robot 与参考 anchor 之间的水平面内位姿差（仅保留 yaw 旋转分量和 $(x, y, z_{\text{ref}})$ 位置偏移），确保只惩罚姿态误差而非全局路径偏差。

---

## 6. Terminations（终止条件）

| 终止条件 | 阈值 | 物理公式 |
| --- | ---: | --- |
| `time_out` | — | $\text{ep\_len} \ge T_{\text{max}} = 10\text{s}$ |
| `anchor_pos` | $0.25\text{m}$ | $\|z_a^{\text{ref}} - z_a^{\text{robot}}\| > 0.25$ |
| `anchor_ori` | $0.8$ | $\|g_{\text{proj}}^{\text{ref}}[2] - g_{\text{proj}}^{\text{robot}}[2]\| > 0.8$ |
| `ee_body_pos` | $0.25\text{m}$ | $\frac{1}{N_{ee}}\sum_{b \in \mathcal{E}} \|z_b^{\text{rel}} - z_b^{\text{robot}}\| > 0.25$ |

其中 $g_{\text{proj}}[2]$ 为重力向量在 body 系下的 z 分量（通过四元数逆旋转投影），$\mathcal{E}$ 为末端执行器集合（左右脚踝 + 左右手腕）。

---

## 7. Metrics（评估指标）

### 7.1 内部 Metrics（MotionCommand 中计算）

| 指标 | 物理公式 |
| --- | --- |
| `error_anchor_pos` | $\|\mathbf{p}_a^{\text{ref}} - \mathbf{p}_a^{\text{robot}}\|$ |
| `error_anchor_rot` | $e_{\text{quat}}(\mathbf{q}_a^{\text{ref}}, \mathbf{q}_a^{\text{robot}})$ |
| `error_anchor_lin_vel` | $\|\mathbf{v}_a^{\text{ref}} - \mathbf{v}_a^{\text{robot}}\|$ |
| `error_anchor_ang_vel` | $\|\boldsymbol{\omega}_a^{\text{ref}} - \boldsymbol{\omega}_a^{\text{robot}}\|$ |
| `error_body_pos` | $\frac{1}{N_B}\sum_b \|\mathbf{p}_b^{\text{rel}} - \mathbf{p}_b^{\text{robot}}\|$ |
| `error_body_rot` | $\frac{1}{N_B}\sum_b e_{\text{quat}}(\mathbf{q}_b^{\text{rel}}, \mathbf{q}_b^{\text{robot}})$ |
| `error_joint_pos` | $\|\mathbf{q}_{\text{ref}} - \mathbf{q}_{\text{robot}}\|$ |
| `error_joint_vel` | $\|\dot{\mathbf{q}}_{\text{ref}} - \dot{\mathbf{q}}_{\text{robot}}\|$ |
| `sampling_entropy` | $H_{\text{norm}} = -\sum_i P(i) \log P(i) \;/\; \log K$ |
| `sampling_top1_prob` | $\max_i P(i)$ |
| `sampling_top1_bin` | $\arg\max_i P(i) \;/\; K$ |

### 7.2 外部 Metrics

| 指标 | 物理公式 | 说明 |
| --- | --- | --- |
| **MPKPE** | $\frac{1}{N_B}\sum_b \|\mathbf{p}_b^{\text{ref}} - \mathbf{p}_b^{\text{robot}}\|$ | 平均关键 body 位置误差（世界系） |
| **R-MPKPE** | $\frac{1}{N_B}\sum_b \|(\mathbf{p}_b^{\text{ref}} - \mathbf{p}_a^{\text{ref}}) - (\mathbf{p}_b^{\text{robot}} - \mathbf{p}_a^{\text{robot}})\|$ | 根相对 MPKPE，排除全局漂移 |
| **Joint Vel Error** | $\|\dot{\mathbf{q}}_{\text{ref}} - \dot{\mathbf{q}}_{\text{robot}}\|$ | 关节速度误差 |
| **EE Pos Error** | $\frac{1}{N_{ee}}\sum_{b \in \mathcal{E}} \|\mathbf{p}_b^{\text{rel}} - \mathbf{p}_b^{\text{robot}}\|$ | 末端执行器位置误差 |
| **EE Ori Error** | $\frac{1}{N_{ee}}\sum_{b \in \mathcal{E}} e_{\text{quat}}(\mathbf{q}_b^{\text{rel}}, \mathbf{q}_b^{\text{robot}})$ | 末端执行器姿态误差 |

---

## 8. Domain Randomization（域随机化）

### 8.1 根状态扰动（重采样时）

见第 2.3 节。pose_range 和 velocity_range 的扰动范围为：

| 参数 | 范围 |
| --- | --- |
| $\Delta x, \Delta y$ | $\mathcal{U}(-0.05, 0.05)$ |
| $\Delta z$ | $\mathcal{U}(-0.01, 0.01)$ |
| $\Delta\text{roll}, \Delta\text{pitch}$ | $\mathcal{U}(-0.1, 0.1)$ |
| $\Delta\text{yaw}$ | $\mathcal{U}(-0.2, 0.2)$ |
| $\Delta v_x, \Delta v_y, \Delta v_z$ | $\mathcal{U}(-0.5, 0.5)$ |
| $\Delta \omega_{\text{roll}}, \Delta \omega_{\text{pitch}}$ | $\mathcal{U}(-0.52, 0.52)$ |
| $\Delta \omega_{\text{yaw}}$ | $\mathcal{U}(-0.78, 0.78)$ |

### 8.2 动力学随机化

| 随机化项 | 方式 | 范围 |
| --- | --- | --- |
| 基座质心偏移 | `startup` 事件，叠加偏移 | $\Delta x, \Delta y \sim \mathcal{U}(-0.05, 0.05),\; \Delta z \sim \mathcal{U}(-0.05, 0.05)$ |
| 编码器偏置 | `startup` 事件，关节位置偏置 | $\mathcal{U}(-0.01, 0.01)$ |
| 足部摩擦 | `startup` 事件，绝对值随机化 | $\mathcal{U}(0.3, 1.2)$，所有足部几何体共享同一随机值 |

### 8.3 外力扰动

| 扰动项 | 方式 | 范围 |
| --- | --- | --- |
| `push_robot` | `interval` 事件，每 $1 \sim 3\text{s}$ 注入一次根速度脉冲 | 同 velocity_range 范围 |

$$
\mathbf{v}_{\text{root}} \leftarrow \mathbf{v}_{\text{root}} + \Delta\mathbf{v}, \quad \Delta\mathbf{v} \sim \mathcal{U}(\text{velocity\_range})
$$

### 8.4 观测噪声（Actor 观测）

对 Policy 观测施加均匀噪声以模拟传感器和状态估计误差，具体噪声区间见第 3.1 节。

---

## 9. 训练算法

### 9.1 算法：PPO + GAE

采用 PPO-Clip 变体，基于 [RSL-RL](https://github.com/leggedrobotics/rsl_rl) 库实现。

| 超参数 | 值 | 说明 |
| --- | ---: | --- |
| $\gamma$ | $0.99$ | 折扣因子 |
| $\lambda$ (GAE) | $0.95$ | GAE 平滑参数 |
| $\epsilon$ (clip) | $0.2$ | PPO 裁剪参数 |
| $c_v$ | $1.0$ | Value loss 系数 |
| $c_e$ | $0.005$ | Entropy 系数 |
| $N_{\text{steps}}$ | $24$ | 每环境每轮 rollout 步数 |
| $N_{\text{epochs}}$ | $5$ | 每轮更新 epoch 数 |
| $N_{\text{minibatches}}$ | $4$ | 小批次数 |
| learning rate | $10^{-3}$ | 初始学习率（自适应调节） |
| `desired_kl` | $0.01$ | KL 散度目标 |
| `max_grad_norm` | $1.0$ | 梯度裁剪 |
| `max_iterations` | $30001$ | 最大训练迭代数 |

### 9.2 模型架构

```text
Actor (策略网络):                     Critic (价值网络):
  obs (actor obs dim)                   obs (critic obs dim)
       │                                       │
  Linear(N, 512) + ELU                   Linear(M, 512) + ELU
       │                                       │
  Linear(512, 256) + ELU                 Linear(512, 256) + ELU
       │                                       │
  Linear(256, 128) + ELU                 Linear(256, 128) + ELU
       │                                       │
  Linear(128, action_dim)                Linear(128, 1)
       │                                       │
  Gaussian(μ, σ_init=1.0)                V(s)
       │
  a ~ N(μ, σ)
```

- 激活函数：ELU
- 分布：GaussianDistribution，scalar std（init_std=1.0）
- 观测归一化：启用（running mean/std）

### 9.3 损失函数

**Policy Loss（Clipped）：**

$$
\mathcal{L}_{\text{clip}}(\theta) = \hat{\mathbb{E}}_t\left[\min\!\left(r_t(\theta)\,\hat{A}_t,\; \text{clip}\!\left(r_t(\theta), 1-\epsilon, 1+\epsilon\right)\hat{A}_t\right)\right]
$$

其中重要性采样比 $r_t(\theta) = \dfrac{\pi_\theta(\mathbf{a}_t \mid \mathbf{s}_t)}{\pi_{\theta_{\text{old}}}(\mathbf{a}_t \mid \mathbf{s}_t)}$。

**Value Loss（Clipped）：**

$$
\mathcal{L}_{\text{value}}(\phi) = \hat{\mathbb{E}}_t\left[\max\!\left((V_\phi(\mathbf{s}_t) - V_t^{\text{targ}})^2,\; (V_{\phi_{\text{old}}}(\mathbf{s}_t) + \text{clip}(V_\phi(\mathbf{s}_t) - V_{\phi_{\text{old}}}(\mathbf{s}_t), -\epsilon, \epsilon) - V_t^{\text{targ}})^2\right)\right]
$$

**Entropy Bonus：**

$$
\mathcal{L}_{\text{entropy}}(\theta) = -\hat{\mathbb{E}}_t\left[H\!\left(\pi_\theta(\cdot \mid \mathbf{s}_t)\right)\right]
$$

**总损失：**

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{clip}} + c_v \cdot \mathcal{L}_{\text{value}} + c_e \cdot \mathcal{L}_{\text{entropy}}
$$

### 9.4 GAE 优势估计

$$
\hat{A}_t = \sum_{l=0}^{\infty} (\gamma\lambda)^l \, \delta_{t+l}, \qquad \delta_t = r_t + \gamma V(\mathbf{s}_{t+1}) - V(\mathbf{s}_t)
$$

**价值目标：**

$$
V_t^{\text{targ}} = \hat{A}_t + V(\mathbf{s}_t)
$$

### 9.5 自适应学习率调度

每轮更新后，根据当前 KL 散度与目标值的比较调整学习率：

$$
\text{lr} \leftarrow
\begin{cases}
\text{lr} \;/\; 1.5, & \text{if } D_{\text{KL}}(\pi_{\text{old}} \parallel \pi) > 0.01 \times 1.5 \\[4pt]
\text{lr} \times 1.5, & \text{if } D_{\text{KL}}(\pi_{\text{old}} \parallel \pi) < 0.01 \;/\; 1.5
\end{cases}
$$

### 9.6 训练伪代码

```pseudo {line-numbers}
Algorithm: PPO with Adaptive Motion Sampling

Hyperparameters: γ=0.99, λ=0.95, ε=0.2, c_v=1.0, c_e=0.005
                 N_steps=24, N_epochs=5, N_minibatches=4
                 lr=1e-3 (adaptive, desired_kl=0.01)
                 max_grad_norm=1.0

1:  Initialize policy π_θ, value function V_φ
2:  Initialize observation normalizer (running μ, σ²)
3:  Initialize MotionCommand with adaptive sampling
4:  Initialize N parallel simulation environments
5:
6:  for iteration ← 1 to 30001 do
7:      // ===== Phase 1: Rollout =====
8:      for step ← 1 to N_steps do
9:          o_t  ← get_observations()               ▷ with noise for actor
10:         a_t  ← π_θ(· | o_t)                     ▷ sample from Gaussian
11:         apply a_t to simulator, step (dt = 0.02s)
12:         r_t  ← Σ_k w_k · r_k(state)             ▷ sum of reward terms
13:         d_t  ← any(termination_condition)       ▷ 4 termination criteria
14:         store (o_t, a_t, r_t, d_t, V_φ(o_t))
15:
16:         if time_step ≥ motion_length then
17:             resample motion time step            ▷ adaptive / uniform / start
18:             apply domain randomization to ref state
19:         end if
20:     end for
21:
22:     // ===== Phase 2: Compute GAE & Returns =====
23:     for step ← N_steps down to 1 do
24:         δ_t    ← r_t + γ · V(s_{t+1}) · (1 − d_t) − V(s_t)
25:         A_t    ← δ_t + γ · λ · A_{t+1} · (1 − d_t)
26:         R_t    ← A_t + V(s_t)
27:     end for
28:
29:     // ===== Phase 3: Policy Update =====
30:     for epoch ← 1 to N_epochs do
31:         shuffle and split rollout data into N_minibatches mini-batches
32:         for each mini-batch (o, a, A, R) do
33:             ρ_t ← π_θ(a|o) / π_θ_old(a|o)       ▷ importance ratio
34:
35:             L_clip    ← mean(min(ρ_t · A, clip(ρ_t, 1−ε, 1+ε) · A))
36:             L_value   ← mean(clipped_value_loss(V_φ(o), V_old(o), R, ε))
37:             L_entropy ← −mean(H(π_θ(·|o)))
38:             L_total   ← L_clip + c_v · L_value + c_e · L_entropy
39:
40:             θ, φ ← Adam(L_total, lr=lr)
41:             clip gradients by max_grad_norm = 1.0
42:         end for
43:     end for
44:
45:     // ===== Phase 4: Adaptive LR Schedule =====
46:     KL ← D_KL(π_θ_old || π_θ)
47:     if KL > desired_kl × 1.5 then  lr ← lr / 1.5
48:     elif KL < desired_kl / 1.5 then  lr ← lr × 1.5
49:
50:     // ===== Phase 5: Logging & Checkpoint =====
51:     if iteration mod save_interval = 0 then
52:         save checkpoint, export ONNX model
53:     end if
54: end for
```

---

## 10. 仿真与部署配置

| 参数 | 值 |
| --- | --- |
| 仿真器 | MuJoCo |
| 物理步长 | $0.005\text{s}$ (200Hz) |
| 控制频率 | $50\text{Hz}$ (decimation = 4) |
| Episode 长度 | $10\text{s}$ = 500 控制步 |
| 地形 | 平面 (plane) |
| 机器人 | Unitree G1 / G1_23Dof |
| 训练环境数 | 可配置（`num_envs`，典型值 4096） |
| 日志工具 | W&B / TensorBoard |

---

## 11. Play 模式

Play/eval 模式下关闭以下随机化以评估策略的确定性表现：

- `enable_corruption = False`（关闭 Actor 观测噪声）
- `push_robot` 事件移除（无外力扰动）
- `pose_range` 和 `velocity_range` 置空（无参考运动扰动）
- `sampling_mode = "start"`（始终从第 0 帧开始）
- `episode_length_s = 10^9$（几乎无时间限制）

---

## 12. 参考

- RSL-RL: [https://github.com/leggedrobotics/rsl_rl](https://github.com/leggedrobotics/rsl_rl)
- MJLab: 基于 MuJoCo 的机器人强化学习框架
- DeepMimic / PHC: 基于参考运动的全身控制方法
