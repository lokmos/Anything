---
marp: true
theme: default
paginate: true
footer: '中期答辩 - 周予恺 2025.06.03'
---

<!-- _class: lead, align-center -->

# 中期答辩

<p align="center">周予恺  <br>2025年6月3日</p>

---

## 量子动力学研究背景

- **量子多体动力学**（Quantum Many-body Dynamics）研究关注多个相互作用粒子组成的量子系统随时间的演化行为。

- 深入理解量子纠缠、非平衡态动力学、热化（thermalization）、多体局域化（many-body localization, MBL）等核心物理现象。

- 揭示量子统计力学、量子信息理论中尚未完全解决的基础问题。

---

## 经典数值模拟方法

### 一、精确对角化方法（Exact Diagonalization, ED）
- 直接对量子多体系统哈密顿量（Hamiltonian）进行完整对角化的方法。
- 可精确求解系统的所有能级与本征态。

### 二、张量网络方法（Tensor Network）
- 通过高维量子多体态的低维表示进行近似求解。
- 包括矩阵乘积态（MPS）和投影纠缠对态（PEPS）等。

### 三、量子蒙特卡罗方法（Quantum Monte Carlo, QMC）
- 基于随机抽样估计量子多体系统的各种统计物理量。
- 广泛用于平衡态统计特性的研究。

---

## 经典数值模拟方法的局限性

### 精确对角化（ED）
- 对整个希尔伯特空间进行完全对角化。
- 空间维数随粒子数指数增长，存储需求巨大。

### 张量网络方法（Tensor Network）
- 纠缠快速增长，存在纠缠障碍。
- 高维扩展困难，计算复杂度高。
- 长期演化的精度显著下降。

### 量子蒙特卡罗方法（QMC）
- 存在严重的符号问题（Sign Problem）。
- 非平衡态动力学表现较差，精度难以保证。

