# p1 

# p2

1. 为什么 ED 方法只能处理小系统？
因为对于一个有 $L$ 个自旋-$1/2$ 粒子的系统，希尔伯特空间的维数是 $2^L$。即使是 $L = 20$，就已经需要 $2^{20} \approx 10^6$ 个复数维度，而 Hamiltonian 是一个 $10^6 \times 10^6$ 的稠密矩阵。对角化这个矩阵的时间复杂度是 $\mathcal{O}(N^3)$，即便在 HPC 上也非常困难。

2. 什么是纠缠障碍（Entanglement Barrier）？
张量网络方法依赖于低纠缠的假设。比如 MPS 的截断维度 $\chi$ 决定了它能表示多少纠缠。如果系统演化后纠缠熵快速增长，比如量子涨落、热化等过程，$\chi$ 需要指数增长才能保留准确性，这会导致计算失效。

3. QMC 的“符号问题”是指什么？
在 QMC 中，我们用随机抽样路径的方式计算期望值。某些情况下（如费米系统、非平衡系统）这些路径的概率权重会变成负数或复数，导致无法解释为概率分布。最终模拟结果中，误差会随样本数增长而非减少，这就是“符号问题”。

4. CUT 的潜在优势？
通过构造一个流动参数 $l$，将哈密顿量逐步对角化；

不依赖初始态，可以计算系统整体的结构；

可用于任意时间点的算符演化，不需要储存整个波函数；

理论上可以分析长期动力学、谱性质、热化过程。

1. **哈密顿量的定义**  
   在经典力学中，哈密顿量是一个函数 $H(p, q)$，表示系统在相空间中的总能量。在量子力学中，它变为一个作用在希尔伯特空间的**厄米算符**，通常记为 $H$，主导量子态随时间的变化。

2. **薛定谔方程中的作用**  
   系统随时间演化由：
   $$
   i \frac{d}{dt} |\psi(t)\rangle = H |\psi(t)\rangle
   $$
   或等价地：
   $$
   |\psi(t)\rangle = e^{-iHt} |\psi(0)\rangle
   $$
   即哈密顿量通过指数映射控制系统的时间演化。

3. **实际结构**  
   在多体系统中，$H$ 常常由如下形式组成：
   $$
   H = \sum_i h_i + \sum_{i,j} J_{ij} O_i O_j + \dots
   $$
   其中 $h_i$ 是单体项（如磁场作用），$O_i O_j$ 是双体或多体相互作用（如耦合、自旋交换等）。

4. **计算上的挑战**  
   哈密顿量维度随粒子数指数增长，$n$ 个自旋-$1/2$ 粒子系统的 $H$ 是一个 $2^n \times 2^n$ 的矩阵，导致计算上瓶颈明显。

5. **在 CUT 方法中的作用**  
   CUT 的目标是通过构造流动参数 $l$ 的演化方程 $\frac{dH}{dl} = [\eta, H]$，将 $H$ 不断变换为一个对角化的 $H(l\to\infty)$，便于后续处理。

1. **张量网络的定义**  
   张量网络是一种用来压缩与组织多体量子态或算符的图结构表达方式。它将一个高维张量拆解为多个低维张量的连接，通过 contraction 表达全局结构。

2. **典型结构：MPS、MPO、PEPS、TTN、MERA**  
   - MPS 适合 1D 系统；
   - PEPS 适合 2D 系统；
   - MPO 是算符的张量网络；
   - MERA 用于临界系统的多尺度建模。

3. **压缩机制：Bond Dimension $\chi$**  
   若两部分子系统之间的纠缠熵为 $S$，则张量网络需满足：
   $$
   \chi \gtrsim \exp(S)
   $$
   才能保留全部信息。

4. **常见压缩方法**  
   - 使用 SVD 保留最大的 $\chi$ 个奇异值；
   - 对张量归一化（canonical form）；
   - 使用递归/自动剪枝策略移除非主导路径。

5. **压缩失败场景：纠缠障碍**  
   在 volume-law 的纠缠结构下（如热化态），$S \sim L$，压缩所需的 $\chi$ 呈指数增长，导致方法失效。

6. **与 CUT 的比较**  
   CUT 方法不显式表示波函数，而是直接演化算符，因此不依赖低纠缠假设，适合处理热化、非平衡过程，但也可能产生项数爆炸。

# p3 

1. **流动参数 $l$ 是什么？**

- 在 CUT 中，我们将哈密顿量 $H$ 视作 $l$ 的函数：$H(l)$；
- 起始点 $l = 0$ 是原始哈密顿量；
- 终点 $l \to \infty$ 是理想的对角哈密顿量；
- 它是一个辅助的“虚拟演化时间”，不对应物理世界中的时间，而是酉变换的过程变量；
- 类似于重整化群（RG）中的“尺度参数”概念。

2. **生成元 $\eta(l)$ 是什么？**

- 数学上，它是一个反厄米（anti-Hermitian）算符，$\eta^\dagger = -\eta$；
- 用于构造哈密顿量在 $l$ 下的导数：
  $$
  \frac{dH(l)}{dl} = [\eta(l), H(l)]
  $$
- 这个对易子形式保证变换是酉的，因为对于任意酉演化 $U(l)$ 有：
  $$
  H(l) = U(l) H(0) U^\dagger(l) \quad \Rightarrow \quad \frac{dH}{dl} = [\eta, H]
  $$
  其中 $\eta = \frac{dU}{dl} U^\dagger$。

3. **生成元的选择影响什么？**

- 不同 $\eta(l)$ 会给出不同的变换路径；
- 比如 Wegner 的生成元选择是：
  $$
  \eta = [H_\text{diag}, H_\text{off}]
  $$
  它的目标是将非对角项压缩为零。

- 而 White 类型生成元会更快收敛于低能子空间，对某些模型更有效率；
- 实际实现中通常需要平衡“压缩效率”与“数值稳定性”。

4. **直觉类比**

- $\eta$ 是变换“风向”，$l$ 是变换“时间”，$H$ 是随风演化的物体；
- 所以整套 CUT 流程是一个“受 $\eta$ 导向的连续变换”。

# p5

1. **CUT 的思想与目标**

- 连续酉变换流方程由 Wegner、Glazek-Wilson 等人提出；
- 本质是通过酉变换 $U(l)$ 将哈密顿量 $H$ 转化为对角或块对角形式；
- 满足 $H(l) = U^\dagger(l) H U(l)$，目标是去除非对角项。

---

2. **微分变换推导过程**

$$
\frac{dH}{dl} = [\eta(l), H(l)], \quad \eta(l) = -\frac{dU}{dl} U^\dagger
$$

常选 $\eta(l) = [H_\text{diag}(l), H_\text{off}(l)]$，可以系统抑制非对角项。

---

3. **数值存储与复杂度**

- $H^{(2)}$：$L \times L$ 矩阵，$\mathcal{O}(L^2)$；
- $H^{(4)}$：四阶张量，$\mathcal{O}(L^4)$；
- 若不截断，对易可能产生 $O(L^6)$ 项，难以承受。

---

4. **截断的重要性**

- 对易后产生更高阶 operator，如六体、八体；
- 实际中通常保留至四体；
- 截断是 CUT 数值实现可行的核心。

# p6

1. **为什么要一维展开？**

- 原始高维系统（如二维晶格）不容易直接做张量运算；
- 高维 CUT 需要实现任意对易子组合，受限于邻接结构；
- 展开为一维后，所有 operator 可统一处理，简化张量表示和合并。

---

2. **展开带来的长程效应**

- 例如二维 $N \times N$ 晶格编号为 $i = x + Ny$，展开后最近邻在编号上差距不再是 ±1；
- 导致 operator 支撑在更远距离上，形成“长程哈密顿量”；
- 但 CUT 处理的是算符代数，不依赖于“距离”的概念。

---

3. **数学与实现层面的优势**

- 所有 operator 都表示为 Pauli 字符串或 Fermionic monomial，对易结构封闭；
- 一维展开后可统一使用张量列（Tensor List）处理；
- 对于 CUT-GPU 实现尤为重要 —— 一维结构更利于并行。

---

4. **对几何结构不敏感的优势**

- 不需要依赖几何邻接关系；
- 即便是具有拓扑结构的系统，如拓扑绝缘体、量子霍尔系统，也可以以这种方式统一处理；
- 在处理随机图、量子神经网络等结构中同样适用。

## 🔍 原始模型中的耦合特点

- 多体系统中的哈密顿量一般具有**局域性**：
  - 即每个粒子（或格点）只与其空间上的**有限个邻居**发生作用；
  - 在哈密顿量中体现为：$H^{(2)}_{ij}$ 和 $H^{(4)}_{ijkq}$ 仅在 $i,j,k,q$ 相邻或接近时非零；
  - 实际物理过程如电子跳跃、库伦作用等通常也只发生在一定范围内。

- 局域性使得数值模拟具备可行性：
  - 相互作用稀疏；
  - 张量存储、对易过程等可以有效裁剪；
  - 是量子多体物理的一个重要假设。

---

## 🔄 一维展开中的“非局域”现象

- 将 $d$ 维晶格展开成线性编号时，原本的近邻格点在编号上可能相隔很远；
  - 例如二维 $N \times N$ 晶格中，$(x,y)$ 可能变成 $i = x + Ny$；
  - $(x,y)$ 和 $(x+1,y)$ 是最近邻，但编号差为 1；
  - 而 $(x,y)$ 和 $(x,y+1)$ 在编号上差为 $N$。

- 展开后的哈密顿量中出现了编号上的“长程项”：
  - 这些项并非物理长程作用，而是编号带来的；
  - 所以仍保留原系统的局域性本质。

- **结论**：
  - 虽然一维化后看起来耦合“非局域”，但 CUT 方法只关注算符的代数结构；
  - 对 CUT 而言是否“长程”并不重要，操作仍基于 operator 支撑集的对易关系。

---

## ✅ 总结

> 在原始模型中，格点之间的作用是局域的；
> 一维展开不会改变这一点，只是表现形式上引入了长程跳跃项。

> CUT 计算依赖代数结构，与空间距离无关，因此仍然适用。

# p7

# p8&p9

### 1. 固定点本质

- CUT 是一个哈密顿量的流动系统；
- 若存在 $dH/dl = 0$ 的状态，就是固定点；
- 如果 $\eta(l)$ 接近 0，系统就卡在此状态附近，演化极慢。

---

### 2. Wegner 生成元机制

- $\eta(l) = [H_0, V]$；
- 可得：
  $$
  \frac{d}{dl} \|V(l)\|^2 = -2 \|\eta(l)\|^2 \leq 0
  $$
- 理论上 $V \to 0$，系统对角化；
- 若 $|H_{ii} - H_{jj}| \ll 1$，则对应 $\eta_{ij} \to 0$，导致演化“冻结”。

---

### 3. 示例推导

- 考虑 $H_{11} = E + \epsilon, H_{22} = E - \epsilon$；
- 则 $\eta_{12} = 2\epsilon \cdot V_{12}$；
- $\epsilon \to 0$ 时 $\eta_{12} \to 0$，对角化停滞。

---

### 4. 可行改进方向

- 使用替代生成元（White 等）；
- 局部 CUT；
- 多通道联合流；
- 动态能量差调节。

---

# p10

### 1. 引入动机

- CUT 会在“近简并”结构中停滞；
- 基底不友好 → 流动方向受限；
- 需要预处理系统，使其更易对角化。

---

### 2. 数学定义

扰乱变换形式为酉变换：
$$
dS(l) = \exp(-\lambda(l)\, dl)
$$
其中 $\lambda(l)$ 是控制扰动结构的生成元。

扰动后的哈密顿量为：
$$
H'(l) = S^\dagger(l)\, H(l)\, S(l)
$$

---

### 3. 实现方式

- $\lambda$ 通常在局域子空间中生成；
- 保持对称性、不引入非物理耦合；
- 可以设计为稀疏扰动或随机扰动。

---

### 4. 优势

- 物理上不改变谱；
- 数值上提升 CUT 收敛性；
- 避免 $\eta_{ij} \to 0$ 的路径僵局；
- 可在 GPU 上高效并行实现。

## 🎓 示例：扰动变换如何解决固定点问题

### 🧱 原始哈密顿量（$2 \times 2$ 近简并子系统）

考虑如下简化模型：
$$
H = \begin{pmatrix}
E + \epsilon & v \\
v & E - \epsilon
\end{pmatrix}
$$

- $E$：中心能量；
- $\epsilon \ll 1$：近简并；
- $v$：两个能级之间的耦合。

---

### 🔁 CUT 的对角化流程

选择 Wegner 生成元：
$$
\eta = [H_0, V] \Rightarrow \eta_{12} = (H_{11} - H_{22}) \cdot V_{12} = 2\epsilon \cdot v
$$

由于 $\epsilon$ 极小，导致：
- $\eta_{12} \approx 0$；
- $\frac{dH_{12}}{dl} \approx 0$，CUT 停滞；
- 对角化难以继续，出现**固定点现象**。

---

### 🔧 引入扰乱变换（Scrambling Transform）

我们设计一个酉变换 $S$，扰乱哈密顿量：

$$
H' = S^\dagger H S = \begin{pmatrix}
E + \epsilon' & v' \\
v' & E - \epsilon'
\end{pmatrix}
$$

设 $\epsilon' \gg \epsilon$，但酉变换保证谱不变。

此时有：
$$
\eta'_{12} = 2\epsilon' \cdot v'
$$

若 $\epsilon' = 0.1$，而原先 $\epsilon = 10^{-5}$，则对角化效率提升约 $10^4$ 倍。

---

### ✅ 总结

- CUT 卡在能级近简并导致的 $\eta_{ij} \to 0$；
- 扰乱变换人为放大能级差，**提升 CUT 收敛性**；
- 保留谱结构，不影响物理性质；
- 可拓展到多体系统中的局域扰动设计。

---

### 💡 延伸说明

- 在真实系统中，扰乱变换并不只作用于 $2 \times 2$ 子块；
- 而是识别多个“近简并局部子空间”，施加**稀疏扰动**；
- 可结合 CUT 的张量表示，并行处理多个受限子空间。

# p11

# p12&p13

### 1. 传统方法的问题

计算物理量需执行：
$$
\langle \psi | U^\dagger(l)\, O(l)\, U(l) | \psi \rangle
$$
但 $U(l)$ 是 $2^L \times 2^L$ 矩阵，不能存储、计算量大。

---

### 2. 对角基的优势

CUT 后本征态 $|E_n\rangle$ 是张量积态；
可直接在该基中进行统计平均：
$$
\langle O \rangle \approx \frac{1}{\mathcal{N}_s} \sum_{n=1}^{\mathcal{N}_s} \langle E_n | O | E_n \rangle
$$

---

### 3. 采样策略

- 采样 $\mathcal{N}_s$ 个态（如 64、128）；
- 每个态表示为二进制位串，GPU 并行评估；
- 避免了还原 $U$ 的过程。

---

### 4. 适用条件

- 高温极限；
- 局域可观测量；
- 长时间演化后的稳态测量。

# p14

# p15

- 系统中只有一个哈密顿量 $H(l)$，它随着流时间 $l$ 演化。
- 格点数 $L$ 决定了希尔伯特空间大小为 $2^L$。
- 哈密顿量不是直接存储为 $2^L \times 2^L$ 矩阵，而是展开为 Pauli 字符串之和：
  $$
  H(l) = \sum_\alpha c_\alpha(l) P_\alpha
  $$
- 每个 $P_\alpha$ 是长度为 $L$ 的张量积字符串（如 $I \otimes X \otimes Z \cdots$）。
- 虽然 Pauli 基有 $4^L$ 个，但实际用到的 $c_\alpha$ 很稀疏，适合稀疏张量存储。

### 示例：$L = 2$ 时的 Pauli 展开

考虑 Heisenberg 相互作用：
$$
H = J (\sigma^x_1 \sigma^x_2 + \sigma^y_1 \sigma^y_2 + \sigma^z_1 \sigma^z_2)
$$

用 Pauli 张量积形式写为：
$$
H = J \cdot (X \otimes X + Y \otimes Y + Z \otimes Z)
$$

因此在 Pauli 展开下，有：

| Pauli 字符串 $P_\alpha$ | 系数 $c_\alpha$ |
|-------------------------|----------------|
| $X \otimes X$           | $J$            |
| $Y \otimes Y$           | $J$            |
| $Z \otimes Z$           | $J$            |
| 其余（如 $I \otimes X$）| $0$            |

说明：

- 总共 $4^2 = 16$ 个 Pauli 字符串；
- 仅有 3 个非零系数；
- Pauli 展开自然稀疏，适合 CUT 存储与更新。

# p17

## 生成元 $\eta(l)$ 的详细解释（答辩准备）

### 1. 定义与作用

生成元 $\eta(l)$ 控制 CUT 中哈密顿量 $H(l)$ 的演化方向：
$$
\frac{dH(l)}{dl} = [\eta(l), H(l)]
$$
Wegner 的选择是：
$$
\eta(l) = [H_{\text{diag}}, H_{\text{off}}]
$$
意图是通过对易子逐步消除非对角项，实现 $H(l)$ 的对角化。

---

### 2. 对角项与非对角项的识别

- Pauli 字符串为 $L$ 位张量积，如 $I \otimes Z \otimes Z$；
- 若字符串中只含 $Z$ 和 $I$，在计算基上是**对角的**；
- 含有 $X$ 或 $Y$ 则为**非对角项**，因其作用会引起比特翻转（跳跃）；

因此 $H(l)$ 被拆为：
- $H_{\text{diag}}$：对角字符串项；
- $H_{\text{off}}$：非对角字符串项；

---

### 3. GPU 并行实现策略

- 遍历所有 $(P_\mu, P_\nu)$ 组合，其中 $P_\mu \in H_{\text{diag}}$，$P_\nu \in H_{\text{off}}$；
- 每一对组合计算对易子：
  $$
  [P_\mu, P_\nu] = 2i f_{\mu\nu} P_\lambda
  $$
- 每对组合的计算是独立的 → 完全并行；
- 生成元 $\eta(l)$ 仍由若干 Pauli 字符串 $P_\lambda$ 构成；
- 其存储格式与 $H(l)$ 一致，可直接复用张量结构；

# p18

## 缺乏结构压缩的详细分析（答辩准备）

### 1. 理论结构上的稀疏性

- CUT 中的哈密顿量 $H(l)$ 通常为局域相互作用的和，具有天然的稀疏性：
  - 一维系统中，多体项受限于局域范围；
  - 二维系统中虽然连通性上升，但仍受限于物理模型（例如最近邻、弱耦合）；
- 实际上：
  - 在 $L \sim 50$ 的一维链上，活跃 Pauli 项的数量仅占全展开空间（$4^L$）的一小部分；
  - 初始扰动和流动过程中，系数演化受限于初始耦合结构，不会激活远离对角线的项。

### 2. 实现上的资源浪费

- 尽管哈密顿量在数学上是稀疏的，但在 GPU 实现中，作者使用了**密集张量**结构：
  - 每一项都对应一个完整的 Pauli 字符串与浮点数系数；
  - 没有采用如 COO、CSR、TT 等结构压缩形式；
  - 导致显存消耗和数据访问开销显著增加。

### 3. 潜在优化方向

- 若能结合：
  - 稀疏表示（如 COO 或稀疏哈希映射）；
  - 稀疏乘法优化；
  - 或低秩张量分解（如 Tensor Train）；
- 将显著降低存储压力，提升 GPU 并行效率。

# p19

## CUT 中压缩策略的深入解析

### 1. 为什么需要压缩？

- 四阶张量项 $H^{(4)}_{ijkq}$ 在 $L$ 点系统中规模为 $\mathcal{O}(L^4)$；
- CUT 流动过程需持续更新这些项；
- 不压缩的张量表示在 GPU 或 CPU 上都难以支撑较大系统；
- 因此需要结构压缩或变换策略。

---

### 2. Tensor Train（TT）分解

- 将高阶张量重构为多个三阶张量的链式乘积：
  $$
  H(l) \approx \sum_{i_1, \dots, i_L} G^{(1)}_{i_1} G^{(2)}_{i_2} \cdots G^{(L)}_{i_L}
  $$
- 存储成本降低为 $\mathcal{O}(L D^2)$，支持 GPU contraction；
- 存在截断误差，需要调节 bond dimension 以平衡效率与精度；
- 实验中误差通常控制在 5% 以内。

---

### 3. Low-Rank 分解

- 将高阶张量 reshape 为矩阵，做秩分解（如 SVD）；
- 初始压缩率高，但 CUT 更新流动中秩常常增长；
- 难以稳定控制秩，误差迅速积累；
- 可用于短时间或近似模拟。

---

### 4. 符号（Symbolic）Pauli 展开（无误差）

- 将 $H(l)$ 精确写为 Pauli 项线性组合：
  $$
  H = \sum_\alpha c_\alpha P_\alpha
  $$
- 保留所有信息，便于 CUT 运算；
- 缺点是当 $L$ 较大时存储复杂度达 $\mathcal{O}(4^L)$；
- 仅适用于小系统验证或理论参考。

---

### 5. 矩阵分块（Block Structure Exploitation）

- 若系统具有守恒量或对称性，可将哈密顿量划分为块对角或稀疏带状结构；
- 例如：
  - 粒子数守恒 → 不同粒子数 sector 可分块；
  - 自旋守恒 → $S_z$ 保持下的子空间分块；
- 每一块独立计算，可显著减少冗余自由度；
- 在 ED 与张量网络中广泛应用，但当前 CUT 实现未显式利用此策略；
- ✅ 精确，无误差；
- ⚠️ 实现需结合对称性识别与张量再构造。

---

### 6. 小结对比

| 方法                | 压缩率         | 是否近似 | 是否精确 | 是否适用于 CUT | 特点说明                        |
|---------------------|----------------|-----------|------------|------------------|---------------------------------|
| Tensor Train (TT)   | 中高           | ✅        | ❌         | ✅               | 并行友好，误差可控              |
| Low-Rank            | 高（初期）     | ✅        | ❌         | ⚠️               | 易爆秩，短程有效                |
| 符号 Pauli 表达     | 无压缩         | ❌        | ✅         | ✅               | 存储极大，仅适用于小系统        |
| 矩阵分块            | 高（结构限定） | ❌        | ✅         | ⚠️               | 需先识别守恒量或对称性结构      |
