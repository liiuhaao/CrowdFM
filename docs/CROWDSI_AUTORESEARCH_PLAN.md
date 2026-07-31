# CrowdSI-FM Autoresearch Master Plan

> 本文件是 CrowdSI-FM 的**单一总控文档**。代码代理应以本文件为最高优先级执行研究、实现、实验、诊断与迭代；`IDEA.md`、`CROWDSI_SPEC.md`、`EXPERIMENT_PLAN.md` 和 `CALIBRATION_RESULTS.md` 提供补充细节，但不得与本文件的阶段门控、统计定义和停止规则冲突。

---

## 0. 一句话定义

**CrowdSI-FM 在合成众包世界上预训练一个可变规模的 crowd foundation model，使其推断整个部署数据集的潜在 crowd mechanism，并仅在 task-disjoint held-out annotations 提供充分预测证据时，在测试时更新该低维机制后验，从而安全地适应新的 worker population、response noise、assignment policy 与 dependence structure。**

---

## 1. 核心研究问题

### 1.1 原始问题

CrowdFM 学习一个固定的跨数据集映射：

```math
G \mapsto q_\theta(Y_k\mid G),
```

其中 `G` 是稀疏 worker-task-option annotation graph。它通过 synthetic domain randomization 获得 zero-shot aggregation 能力，但面对以下 deployment shift 时没有显式可更新对象：

- worker ability distribution 变化；
- task difficulty / discrimination distribution 变化；
- class-conditioned response bias；
- worker coalition / correlated responses；
- non-random worker-task assignment；
- Sybil workers、targeted corruption；
- temporal drift；
- 多种机制的未见组合。

### 1.2 CrowdSI 研究问题

> 能否让一个 crowd foundation model 不仅聚合标签，还 amortize 一个数据集级 crowd mechanism posterior，在没有 deployment gold labels 的情况下从 annotations 做 test-time system identification，并只在独立 predictive evidence 支持时启用 adaptation？

### 1.3 非目标

本工作**不声称**：

- 无 gold label 时可以认证 task truth 一定正确；
- non-rejection 证明 deployment mechanism 与真实世界完全一致；
- 任意 observationally equivalent failure 都可检测；
- e-value 能修复任意 simulator misspecification；
- assignment head 当前已经拥有完整 finite-sample gate。

正确定位是：

```text
zero-shot crowd aggregation
→ amortized crowd-system initialization
→ unlabeled test-time mechanism identification
→ evidence-gated adaptation
```

---

## 2. 形式化生成模型

对一个 deployment dataset `D`，定义：

- workers: `i = 1,...,M`；
- tasks: `k = 1,...,N`；
- classes/options: `a,c = 1,...,K`；
- assignment indicator: `O_ik ∈ {0,1}`；
- observed response: `A_ik ∈ [K]` when `O_ik=1`；
- latent truth: `Y_k ∈ [K]`；
- dataset mechanism: `Z_D`；
- worker latent: `U_i`；
- task latent: `V_k`。

基础机制分解：

```math
Z_{\mathcal D}\sim p(Z),
```

```math
U_i\sim p_\theta(U\mid Z_{\mathcal D}),
\qquad
V_k\sim p_\theta(V\mid Z_{\mathcal D}),
```

```math
Y_k\sim p_\theta(Y_k\mid V_k,Z_{\mathcal D}),
```

```math
O_{ik}\sim p_\theta(O_{ik}\mid U_i,V_k,Z_{\mathcal D}),
```

```math
A_{ik}\mid O_{ik}=1,Y_k
\sim
p_\theta(A_{ik}\mid Y_k,U_i,V_k,Z_{\mathcal D}).
```

该分解与 separately exchangeable worker-task arrays 一致：worker ID 和 task ID 的排列不应改变 dataset mechanism posterior，worker/task outputs 应分别 permutation equivariant。

### 2.1 Shared-truth likelihood

同一 task 上所有 held-out workers 共享一个 `Y_k`。因此 audit/adaptation likelihood 必须为：

```math
p(A_{H_k}\mid G_C,Z)
=
\sum_{c=1}^{K}
q_\theta(Y_k=c\mid G_C,Z)
\prod_{e\in H_k}
P_\theta(A_e\mid Y_k=c,G_C,Z).
```

禁止替换为 edgewise marginal likelihood 的简单乘积：

```math
\prod_{e\in H_k}\sum_c q_k(c)P_e(A_e\mid c),
```

因为该错误形式会破坏同一 task 中由共享真值导致的合法依赖。

---

## 3. 模型结构

主模型：`src/cfm/model/CrowdSIFM.py`

### 3.1 CrowdFM backbone

原始 CrowdFM 提供：

- worker embeddings `z_w`；
- task embeddings `z_t`；
- option embeddings `z_o`；
- original task logits `hat_task_option_base`。

必须保留原始输出，用于与 CrowdSI zero-shot/adapted 独立比较。

### 3.2 Dataset-level Gaussian mechanism encoder

机制编码器输出：

```math
q_\phi(Z_{\mathcal D}\mid G)
=
\mathcal N(\mu_G,\operatorname{diag}(\sigma_G^2)).
```

当前 pooling 输入：

1. mean worker embedding；
2. mean task embedding；
3. mean option embedding；
4. mean encoded observed-edge embedding；
5. worker/task normalized degree statistics：mean、std、min、max。

Degree statistics 是 load-bearing 设计：assignment density 与 selection structure 可能被 normalized graph attention 淡化，不能只依赖 backbone embeddings。

### 3.3 Compositional mechanism basis

设 `R` 个 learned mechanism primitives：

```math
\alpha(Z)=\operatorname{softmax}(WZ+b),
```

```math
C(Z)=\sum_{r=1}^{R}\alpha_r(Z)B_r.
```

目的不是监督恢复 simulator family ID，而是支持：

- 相同 primitive 的不同强度；
- 多机制组合；
- 训练只见单机制、测试见组合；
- 跨 `M,N,K,density` 的机制表示稳定性。

Primitive index 本身不可识别，跨 seed 比较时应使用 permutation-invariant metrics 或先做匹配。

### 3.4 Mechanism-conditioned worker/task adapters

使用 `C(Z)` 调制 worker/task representations：

```math
\tilde U_i=f_w([z_i^w,C(Z)]),
\qquad
\tilde V_k=f_t([z_k^t,C(Z)]).
```

### 3.5 Truth head

输出：

```math
q_\theta(Y_k=c\mid G,Z).
```

这是 CrowdSI zero-shot/adapted 聚合的直接预测。

### 3.6 Edge-conditioned emission head

对 query edge `(i,k)` 输出：

```math
P_{ik}^{Z}(a\mid c)
=
\Pr_\theta(A_{ik}=a\mid Y_k=c,G,i,k,Z).
```

Tensor contract：

```text
hat_annotation_given_truth: [Q,K,K]
```

其中第二维是 candidate truth，第三维是 reported option。

### 3.7 Assignment propensity head

输出：

```math
\pi_{ik}^{Z}=\Pr_\theta(O_{ik}=1\mid G,i,k,Z).
```

当前 assignment head 用于 system-identification training signal。第一版 e-value 是 response-only；assignment e-value 必须使用独立、无泄漏的 mask split 后再加入。

---

## 4. 训练目标

主实现：`src/cfm/si/training.py`

总损失：

```math
\mathcal L
=
\lambda_Y\mathcal L_{truth}
+
\lambda_A\mathcal L_{annotation}
+
\lambda_O\mathcal L_{assignment}
+
\lambda_Z\mathcal L_{KL}
+
\lambda_C\mathcal L_{consistency}.
```

### 4.1 Truth loss

Synthetic world 有 task truth 时：

```math
\mathcal L_{truth}
=-\sum_k\log q_\theta(Y_k\mid G_C,Z).
```

### 4.2 Masked annotation loss

有 known truth 时：

```math
\mathcal L_{annotation}
=-\sum_{(i,k)\in H}
\log P_{ik}^{Z}(A_{ik}\mid Y_k).
```

无 truth 的真实数据做 self-supervised training 时：

```math
\mathcal L_{annotation}
=-\sum_k
\log\sum_c q_k(c\mid G_C,Z)
\prod_{(i,k)\in H_k}P_{ik}^{Z}(A_{ik}\mid c).
```

### 4.3 Assignment loss

完整目标应近似 full worker-task mask likelihood：

```math
-\sum_{i,k}
\left[
O_{ik}\log\pi_{ik}
+(1-O_{ik})\log(1-\pi_{ik})
\right].
```

实现中：

- positives 来自被随机隐藏的 observed edges；
- negatives 来自 unobserved worker-task pairs；
- positive 和 negative 均使用 inverse-sampling weighting；
- 禁止普通 sampled BCE 后直接解释为 calibrated propensity。

### 4.4 Mechanism prior KL

```math
\mathcal L_{KL}
=
D_{KL}(q_\phi(Z\mid G_C)\|\mathcal N(0,I)).
```

### 4.5 Cross-view consistency

同一 crowd world 抽两个独立 masked views：

```math
\mathcal L_{consistency}
=
D_{SKL}
\left(
q_\phi(Z\mid G^{(1)}),
q_\phi(Z\mid G^{(2)})
\right).
```

目的：迫使 `Z_D` 表示 dataset mechanism，而不是某个局部 edge pattern。

---

## 5. Test-time system identification

主实现：`src/cfm/si/adaptation.py`

### 5.1 Base posterior

从 context graph 得到：

```math
q_0(Z)=\mathcal N(\mu_0,\operatorname{diag}(\sigma_0^2)).
```

### 5.2 Latent-only generalized-Bayes update

冻结所有 network weights，只优化：

```text
adapted_mean
adapted_log_variance
```

目标：

```math
\max_q
\frac{1}{\tau}
\mathbb E_{Z\sim q}
[\log p_\theta(A_H\mid G_C,Z)]
-
\lambda
D_{KL}(q(Z)\|q_0(Z)).
```

- `tau = likelihood_temperature`；
- `lambda = adaptation_kl_weight`；
- 这是低维 mechanism adaptation，不是网络 fine-tuning；
- 模型 `train/eval` 状态和所有 `requires_grad` flags 必须在 adaptation 后恢复。

---

## 6. Task-disjoint cross-fitted e-value gate

### 6.1 为什么必须按 task 切分

同一 task 的 workers 共享 latent truth `Y_k`。若按 edge 随机分 fold，fold A/B 会通过同一个 `Y_k` 相关，破坏独立 evidence 解释。

要求：

```text
context ∩ fold_A = ∅
context ∩ fold_B = ∅
fold_A ∩ fold_B = ∅
tasks(fold_A) ∩ tasks(fold_B) = ∅
```

### 6.2 Current plug-in null

当前 null 使用 amortized mechanism mean：

```math
H_0: Z=\mu_0.
```

在 fold A 上得到 adapted posterior `q_A`，在 fold B 上评估：

```math
E_{A\to B}
=
\frac{
\int p(A_B\mid G_C,z)q_A(z)\,dz
}{
p(A_B\mid G_C,\mu_0)
}.
```

对称地得到 `E_{B→A}`，组合：

```math
E=\frac12(E_{A\to B}+E_{B\to A}).
```

Gate：

```math
E\ge 1/\alpha
\quad\Rightarrow\quad
\text{adaptation supported}.
```

### 6.3 当前理论范围

在以下条件下，每个 fold ratio 是 e-value：

- null denominator 是固定 normalized predictive law；
- numerator 是只使用 opposite fold 学得的 normalized predictive mixture；
- task units 条件独立；
- likelihood 与 split 无泄漏。

当前保证控制的是：

> 在 plug-in null `Z=mu_0` 正确时，错误替换该固定机制的概率。

当前不保证：

- 完整 amortized posterior null；
- arbitrary model misspecification；
- truth correctness；
- response 与 assignment 的联合 gate。

### 6.4 必须实现的理论升级

优先升级 denominator 为 full null posterior predictive：

```math
p_0(A_H\mid G_C)
=
\int p(A_H\mid G_C,z)q_0(z)\,dz.
```

需要验证：

- ratio 仍为合法 e-value；
- numerator/denominator Monte Carlo approximation 不引入系统性 anti-conservatism；
- 使用 common random numbers 或 lower-confidence e-value 是否必要。

---

## 7. 三种输出必须分开报告

每个 dataset/world 同时输出：

1. **Original CrowdFM**：`hat_task_option_base`；
2. **CrowdSI zero-shot**：使用 amortized `mu_0`；
3. **CrowdSI selected/adapted**：使用 evidence-selected mechanism。

核心分解：

```math
\Delta_{architecture}
=
Acc(CrowdSI\text{-zero-shot})-Acc(CrowdFM),
```

```math
\Delta_{adaptation}
=
Acc(CrowdSI\text{-adapted})-Acc(CrowdSI\text{-zero-shot}).
```

不允许只比较 adapted 与 CrowdFM，否则无法判断收益来自新 architecture 还是 test-time adaptation。

---

## 8. 论文核心 claims 与通过标准

### C1. Amortized mechanism representation

**Claim**：模型能从可变规模 annotation graph 推断稳定、可迁移的 dataset-level crowd mechanism representation。

通过标准：

- 同一 world 两个 masked views 的 posterior distance 显著低于不同 worlds；
- 相同机制、不同 `M,N,K,density` 的 latent 可对齐；
- mechanism strength 可从 latent 线性或低复杂度解码；
- primitive mixture 对 unseen compositions 有可解释组合行为。

### C2. Zero-shot aggregation 不退化

**Claim**：加入 mechanism architecture 后，在 in-prior worlds 上不显著损害 CrowdFM。

通过标准：

```text
CrowdSI zero-shot accuracy ≥ CrowdFM accuracy - 0.5 percentage point
```

并在至少部分机制上有稳定提升。

### C3. Unseen-mechanism adaptation

**Claim**：低维 test-time system identification 在训练时完全未见的 mechanism families/compositions 上改善 aggregation。

通过标准：

- 多个 held-out families 上平均提升；
- 至少 3 个随机种子；
- 95% confidence interval 不跨 0，或 paired statistical test 显著；
- improvement 不只出现在极端 severity；
- 在同等 annotation budget 下优于 full-network TTA。

### C4. Safe evidence gating

**Claim**：e-value gate 在 null 下控制 false adaptation，在 shift 下有随样本量与 severity 增长的 power。

通过标准：

- `alpha ∈ {0.01,0.05,0.10}`；
- 至少 500–1000 in-prior worlds；
- false-adaptation rate 不系统超过 nominal；
- power 随 task count、shift severity 单调上升；
- evidence-gated adaptation 的 worst-case degradation 小于 always-adapt。

### C5. Realistic transfer

**Claim**：结果不依赖模型反演自己定义的 simulator。

通过标准：

- real-mask semi-synthetic experiments；
- 真实 crowdsourcing benchmarks；
- deployment gold truth 仅用于最终 evaluation，不用于 adaptation/gating；
- 至少一个 real dataset 上 adapted CrowdSI 显著优于 CrowdFM 或显著改善 risk/coverage。

---

## 9. 必须比较的 baselines

### Aggregation baselines

- Majority vote；
- Dawid-Skene；
- MACE；
- GLAD；
- EBCC / dependence-aware model（实现可用时）；
- DGN / DGN-Adapt（若代码可用且许可允许）；
- original CrowdFM。

### Architecture baselines

- CrowdFM + ordinary global token；
- PredictiveCFM without dataset mechanism；
- CrowdSI without primitives；
- CrowdSI discrete family classifier；
- CrowdSI continuous Gaussian latent；
- oracle mechanism parameters；
- oracle family ID。

### Adaptation baselines

- no adaptation；
- always latent-adapt；
- response-head-only adaptation；
- truth-head-only adaptation；
- last-layer fine-tuning；
- full-network test-time fine-tuning；
- entropy minimization；
- held-out NLL improvement threshold；
- gradient-norm / prediction-instability gate；
- oracle gate。

### Gate baselines

- CrowdFM entropy；
- task margin；
- split-view prediction divergence；
- latent Mahalanobis/MMD OOD score；
- fixed threshold on held-out NLL；
- likelihood-ratio without cross-fitting；
- edge-random split instead of task split；
- e-value with plug-in denominator；
- e-value with full posterior denominator。

---

## 10. Synthetic mechanism families

Current implemented families：

```text
irt
class_bias
coalition
assignment_bias
mixed
```

必须继续新增：

1. temporal drift；
2. worker group / legitimate subjectivity；
3. low-rank Ising/Potts dependence；
4. copycat / answer imitation；
5. Sybil insertion；
6. targeted class attack；
7. adversarial worker-task assignment；
8. heterogeneous task types；
9. heavy-tailed worker ability；
10. class imbalance / semantic option asymmetry。

### 10.1 Hold-out 规则

禁止只 hold out parameter ranges。必须至少包含：

- hold out complete family；
- train single families, test mixtures；
- train pairwise mixtures, test triple mixtures；
- hold out interaction form，例如 train additive, test nonlinear interaction；
- hold out assignment mechanism while response mechanism remains in-prior；
- hold out dependence mechanism while marginal confusion remains matched。

---

## 11. 详细实验阶段与门控

## Phase 0 — Correctness and smoke validation

命令：

```bash
git pull --ff-only origin agent/cbr-audit-core
pytest -q
python train_crowdsi.py config=config/crowdsi_smoke.yaml
```

必须检查：

- all tests pass；
- all losses finite；
- mechanism encoder、truth head、emission head、assignment head 均有非零梯度；
- checkpoint 中 `crowdsi_trained=true`；
- resume/load 严格成功；
- one shared truth likelihood 数值测试通过；
- task-disjoint split invariants 通过；
- adaptation 后 network weights bitwise unchanged；
- model state 与 requires-grad flags 恢复。

**STOP 条件**：任何 NaN、shape mismatch、data leakage、checkpoint incompatibility，先修复，不进入 Phase 1。

---

## Phase 1 — In-prior predictive validation

目的：先验证模型会生成 annotations 和 mask，再讨论 adaptation。

指标：

- task accuracy / NLL；
- masked annotation joint NLL；
- edge Brier score；
- response top-1 accuracy；
- assignment AUROC/AUPRC；
- assignment calibration；
- posterior consistency；
- primitive entropy 与 collapse；
- metrics stratified by `K`, density, worker support, task degree。

通过标准：

- response model 优于 unconditional option-frequency baseline；
- response model 优于 worker-frequency baseline；
- assignment head 优于 density-only baseline；
- CrowdSI zero-shot 不明显弱于 CrowdFM；
- mechanism posterior 不完全 collapse 到 prior；
- primitives 不全部使用同一个 component。

失败应对：

- response NLL 不好：先检查 shared-truth likelihood、query leakage、truth loss 权重；
- assignment calibration 差：检查 importance weights 和 negative sampling；
- posterior collapse：降低 KL、使用 KL warm-up/free bits、增强 view consistency；
- primitive collapse：加 entropy/diversity regularizer，但必须做消融；
- zero-shot accuracy 下降：初始化 truth head 接近原 CrowdFM head，或增加 distillation loss。

---

## Phase 2 — Exact/near-null e-value calibration

构造两类 null：

### Null A: exact decoder null

直接从训练后模型自己的 predictive law 采样 world/held-out responses。该实验用于验证代码和 e-value 理论。

预期：

```math
\Pr(E\ge1/\alpha)\le\alpha.
```

### Null B: unseen in-prior simulator worlds

从 declared training prior 采样但不参与训练。

用于检测 learned model error 对 gate calibration 的影响。

必须报告：

- false adaptation rate；
- e-value distribution；
- plug-in null vs posterior null；
- fold direction separately；
- task count sweep；
- Monte Carlo sample sweep；
- confidence intervals。

失败应对：

- Exact null 失败：实现/数学错误，立即停止；
- Exact null 正确但 simulator null 超标：模型 misspecification，不允许声称 finite-sample deployment validity；
- posterior denominator 改善：升级为主方法；
- approximate Monte Carlo 导致超标：增加 samples、使用 conservative lower bounds 或 sample splitting。

---

## Phase 3 — Held-out mechanism adaptation

实验矩阵至少包含：

| Train mechanisms | Test mechanism | Purpose |
|---|---|---|
| IRT | class bias | unseen marginal response law |
| IRT + class bias | coalition | unseen dependence |
| IRT + class bias | assignment bias | unseen selection mechanism |
| all single families | mixed | compositional generalization |
| pairwise mixtures | triple mixture | higher-order composition |
| static mechanisms | temporal drift | temporal OOD |

每个设置 sweep：

- number of workers；
- number of tasks；
- `K`；
- labels per task；
- shift severity；
- coalition size；
- assignment strength。

主表必须包含：

```text
CrowdFM
CrowdSI zero-shot
CrowdSI always-adapt
CrowdSI e-gated
full-network TTA
oracle latent
oracle gate
```

必须报告：

- aggregation accuracy；
- adaptation rate；
- adaptation gain conditional on gate；
- false adaptation cost；
- worst-case degradation；
- response NLL before/after；
- mechanism posterior movement；
- runtime。

---

## Phase 4 — Mechanism representation analysis

### 4.1 Cross-view consistency

同一 world 的两个 masks：

```math
D(q(Z\mid G^{(1)}),q(Z\mid G^{(2)})).
```

与不同 world 对比。

### 4.2 Scale invariance

固定生成机制，改变：

- `M`；
- `N`；
- `K`；
- density。

检查 latent 与 primitive weights 是否稳定。

### 4.3 Compositionality

训练单机制，测试组合，验证：

- mixture weights 是否近似组合；
- adapted latent 是否沿合理方向移动；
- composition performance 是否优于 discrete-family baseline。

### 4.4 Intervention

固定 graph embeddings 或模拟参数，单独改变：

- class-bias strength；
- coalition fraction；
- assignment strength；
- ability variance。

检查 latent direction 与 decoder behavior。

### 4.5 Identifiability boundary

构造两个产生相同 observable distribution 的 mechanism parameterizations，证明 latent 不应被要求区分。报告 prediction-equivalence，而不是伪造 parameter recovery。

---

## Phase 5 — Real-mask semi-synthetic

使用真实数据的 worker-task mask，保留：

- sparsity；
- degree heterogeneity；
- worker overlap；
- task support distribution。

在真实 mask 上注入：

- in-prior IRT；
- class bias；
- coalition；
- assignment-conditioned responses；
- mixtures。

作用：排除 synthetic dense graph 造成的虚假成功。

候选数据集：LabelMe、RTE、TREC、Dog、Bird、Music、CIFAR-10H/10N 等，以仓库实际可用数据为准。

---

## Phase 6 — Real benchmarks

原则：

- adaptation/gating 不使用 gold truth；
- gold truth 只用于最终评估；
- 每个 dataset 至少多个 cross-fit seeds；
- 报告 CrowdFM / CrowdSI-zero-shot / CrowdSI-adapted；
- 报告 response prediction、assignment prediction、e-value、accuracy gain；
- 报告没有 adaptation 的 dataset，避免选择性展示。

主结果：

- accuracy；
- macro accuracy / balanced accuracy（类不平衡时）；
- NLL；
- risk-coverage；
- AURC；
- runtime；
- adaptation frequency；
- seed stability。

---

## Phase 7 — Required ablations

必须至少包含：

1. remove mechanism latent；
2. remove compositional basis；
3. discrete family embedding；
4. remove assignment head；
5. remove degree statistics；
6. remove response head；
7. remove truth loss；
8. remove consistency loss；
9. remove KL；
10. frozen vs joint backbone training；
11. latent mean-only adaptation；
12. mean+variance adaptation；
13. always adapt vs gated；
14. task split vs edge split；
15. plug-in denominator vs full posterior denominator；
16. posterior mixture samples；
17. latent dimension；
18. number of primitives；
19. adaptation steps / LR / KL weight / temperature；
20. real assignment likelihood vs unweighted sampled BCE。

---

## 12. AAAI 主会最低完成标准

只有同时满足以下条件，才将工作定位为 AAAI main-track ready：

### Novelty

- 明确定义 `test-time crowd system identification` 新问题；
- 方法不是普通 global token 或普通 TTA；
- dataset mechanism posterior、joint mask/response model 与 evidence gate 构成不可替代闭环。

### Theory

至少完成：

1. task-disjoint cross-fitted e-value level theorem；
2. power consistency：在 KL-separated alternative 下 `log E / |H|` 收敛到正值；
3. 明确 plug-in null 与 posterior null 的区别；
4. observational equivalence/impossibility statement。

### Experiments

- exact null calibration；
- held-out complete mechanisms；
- compositional generalization；
- real-mask semi-synthetic；
- real benchmarks；
- strong TTA and aggregation baselines；
- three-output decomposition；
- at least 3 training seeds；
- statistical uncertainty。

### Practical result

至少满足：

- multiple held-out mechanisms 上 adapted > zero-shot；
- gate 显著减少 always-adapt 的负迁移；
- real-mask 或 real dataset 上有可复现收益；
- runtime 合理。

若只有同分布 synthetic improvement，则降级为 workshop；若 held-out synthetic work 但没有 real-mask/real data，则 AAAI borderline。

---

## 13. 失败模式与决策树

### F1. CrowdSI zero-shot 比 CrowdFM 差

处理顺序：

1. 检查 truth head 初始化；
2. 增加 CrowdFM logit distillation；
3. residual mechanism adapter 而非完全替换 head；
4. freeze backbone 再训 heads；
5. 调整 loss 权重。

停止规则：若多 seed、充分训练后仍持续下降 >2pp，重新设计 architecture，不继续做 adaptation 实验。

### F2. Mechanism posterior collapse

处理：

- KL warm-up；
- free bits；
- 减小 KL weight；
- 增强 mechanism-changing augmentations；
- contrastive/world-ID auxiliary loss，仅作为训练辅助；
- 检查 decoder 是否绕过 `Z`。

### F3. Primitives collapse

处理：

- router entropy regularization；
- primitive orthogonality/diversity loss；
- load-balancing；
- Gumbel-softmax/discrete alternative baseline。

不得因为 latent 可视化不好就直接加入强监督 family labels作为主方法。

### F4. Adaptation improves likelihood but harms truth accuracy

这是核心风险。处理：

- joint objective 中增加 truth-preserving regularization；
- adaptation gate 同时检查 response evidence 与 truth-posterior stability；
- constrain KL / trust region；
- adapt only mechanism subspace influencing emissions, then evaluate whether truth head should share it；
- introduce two latents：response mechanism 与 aggregation-relevant mechanism；
- 报告 likelihood-accuracy misalignment，不得隐藏。

### F5. E-value null anti-conservative

- exact null 失败：代码/理论 bug；
- plug-in null 失败但 posterior null 正确：更换 denominator；
- Monte Carlo approximation 失败：conservative estimator；
- learned model null 失败：不能宣称 nominal level，只能称 evidence score。

### F6. Gating 几乎从不触发

- 检查 alternative capacity；
- 增加 evidence task count；
- power analysis；
- 减少 adaptation KL；
- 检查 denominator 是否过宽；
- 不得简单降低 threshold 破坏 level。

### F7. Gating 经常触发但 adaptation 无收益

说明机制 shift 可预测但不与 aggregation error 对齐。处理：

- 研究 response mismatch 与 truth error 的关联；
- 训练 aggregation-relevant latent；
- 加入 selective routing，而非强制 adaptation；
- 将该结果定位为 audit 而非 adaptation only when empirically justified。

### F8. Real data 无提升

- 先看 zero-shot response NLL 与 mask likelihood；
- 检查 real mechanism 是否超出 simulator；
- synthetic + real unlabeled self-supervised pretraining；
- real-mask training；
- 增加机制 families；
- 若真实数据仍无收益，论文不能以 deployment adaptation 为主 claim。

---

## 14. Autoregressive Code Agent 执行协议

### 14.1 每轮循环

代理必须按以下循环：

1. **Read state**：读取本文件、相关代码、最新 logs、tests；
2. **Form one hypothesis**：每次只提出一个主要失败假设；
3. **Design smallest discriminative experiment**：能区分至少两个解释；
4. **Implement minimally**：避免同时修改多个 load-bearing 模块；
5. **Run tests/smoke**；
6. **Run bounded experiment**；
7. **Write result**：保存 config、seed、commit、metrics、结论；
8. **Decide**：accept / reject hypothesis；
9. **Update this document or a linked log**；
10. **Commit**。

### 14.2 禁止行为

- 不得在 tests 失败时继续大规模实验；
- 不得只看单 seed；
- 不得用 deployment truth 训练/gate；
- 不得把同机制参数 hold-out 称为 mechanism OOD；
- 不得将 p/e-value calibration failure 用调 threshold 掩盖；
- 不得同时改变 simulator、architecture、loss 和 gate 后宣称因果结论；
- 不得删除负结果；
- 不得 merge PR，除非用户明确要求；
- 不得关闭当前 draft PR。

### 14.3 实验命名

建议格式：

```text
<phase>_<mechanism-split>_<model>_<gate>_<seed>_<date>
```

例如：

```text
p3_train-single_test-mixed_crowdsi_fullpost_evalue_s42
```

### 14.4 必须记录的 metadata

每个 run：

```json
{
  "git_commit": "...",
  "config": "...",
  "seed": 42,
  "train_mechanisms": ["..."],
  "test_mechanisms": ["..."],
  "M": "range",
  "N": "range",
  "K": "range",
  "density": "range",
  "checkpoint": "...",
  "runtime_seconds": 0,
  "gpu": "...",
  "crowdfm_accuracy": 0,
  "crowdsi_zero_shot_accuracy": 0,
  "crowdsi_selected_accuracy": 0,
  "e_value": 0,
  "adaptation_supported": false,
  "response_nll_before": 0,
  "response_nll_after": 0,
  "assignment_nll": 0,
  "mechanism_shift_norm": 0
}
```

### 14.5 自动停止条件

代理必须停止并报告，而不是无限调参，当：

- 连续 3 个有区分力的实验否定同一核心假设；
- exact-null gate 无法校准；
- adapted likelihood 与 task accuracy 持续负相关；
- real-mask experiments 明显否定 synthetic conclusion；
- 计算/数据错误导致结果不可解释；
- 需要更改研究问题而不仅是实现细节。

---

## 15. 推荐执行顺序

严格按以下优先级：

```text
P0 tests + smoke
P1 in-prior truth/response/assignment validation
P2 exact-null e-value
P2b full-posterior denominator
P3 one held-out-family pilot
P3b strong TTA baselines
P4 mechanism representation/composition
P5 real-mask semi-synthetic
P6 real benchmarks
P7 full ablations
P8 scaling/runtime
paper figures + theorem finalization
```

不要在 P2 前运行大规模 OOD power；不要在 P3 pilot 失败时批量跑完整矩阵。

---

## 16. 当前代码状态

Primary files：

```text
src/cfm/model/CrowdSIFM.py
src/cfm/data/crowdsi_simulator.py
src/cfm/si/likelihood.py
src/cfm/si/split.py
src/cfm/si/adaptation.py
src/cfm/si/training.py
src/cfm/si/pipeline.py
train_crowdsi.py
evaluate_crowdsi.py
config/crowdsi_train.yaml
config/crowdsi_eval.yaml
config/crowdsi_smoke.yaml
tests/test_crowdsi.py
```

Ablations/history：

```text
PredictiveCFM: fixed-mechanism response-audit baseline
legacy confusion CbR: negative-result reproduction only
```

已知未完成项：

- full posterior null denominator；
- assignment e-value；
- temporal / subjective / Ising-Potts mechanisms；
- explicit mechanism identifiability regularization；
- `Q*K*K` query chunking；
- robust checkpoint resume and experiment registry；
- real-mask injection pipeline；
- complete baseline wrappers。

---

## 17. 最终论文叙事

推荐标题：

> **CrowdSI: Amortized System Identification for Adaptive Crowd Foundation Models**

核心贡献叙事：

1. 提出 crowd foundation model 的 test-time system identification 问题；
2. 建立 global-worker-task-edge 的 exchangeable generative mechanism model；
3. 使用 compositional mechanism posterior 在新 crowd graph 上做 latent-only adaptation；
4. 使用 task-disjoint cross-fitted e-value 控制无证据更新；
5. 证明 held-out mechanism 与 composition 上的 adaptation/generalization；
6. 在 real masks 和 real crowds 上验证部署价值。

Reviewer 最可能的质疑及回答：

### “只是 CrowdFM + global latent + TTA”

回答必须依靠：

- joint assignment/response/truth mechanism；
- compositional generalization；
- task-disjoint evidence gate；
- latent-only 与 full-network TTA 对比；
- exact-null calibration。

### “模型只是在反演自己的 simulator”

回答必须依靠：

- held-out families；
- interaction-form holdout；
- real-mask semi-synthetic；
- real datasets；
- simulator-independent baselines。

### “likelihood improvement 不等于 truth improvement”

回答必须依靠：

- three-output decomposition；
- likelihood vs accuracy correlation；
- negative-transfer analysis；
- evidence-gated vs always-adapt；
- trust-region / aggregation-relevant latent ablation。

### “e-value 保证太窄”

回答必须诚实区分：

- exact plug-in null theorem；
- full posterior null extension；
- learned-model calibration；
- deployment misspecification boundary。

---

## 18. 第一批立即执行任务

1. 拉取分支并运行所有 tests；
2. 运行 `config/crowdsi_smoke.yaml`；
3. 添加 checkpoint resume test；
4. 添加 exact-model-null e-value calibration test；
5. 实现 full posterior denominator；
6. 实现一个最小 held-out-family pilot：train `irt,class_bias`, test `coalition`；
7. 同时跑 CrowdFM、CrowdSI zero-shot、always-adapt、e-gated、full-network TTA；
8. 输出 first decision report：
   - architecture 是否成立；
   - latent 是否被使用；
   - exact-null 是否校准；
   - adaptation 是否有正向信号；
   - 是否值得进入完整实验。

---

## 19. 当前 go/no-go 判断

CrowdSI-FM 具有 AAAI 主会级别的**问题与方法上限**，但只有同时完成以下三条才算真正 work：

```text
unseen-mechanism adaptation
+ safe evidence gating
+ realistic/real-world transfer
```

任何单独一条都不足以支撑主会论文：

- 只有 mechanism latent：普通 representation learning；
- 只有 adaptation：普通 TTA；
- 只有 e-value：窄统计 wrapper；
- 只有 synthetic gains：可能反演 simulator；
- 只有 real-data gains 但无机制分析：难证明贡献来源。

本项目的核心目标是证明三者形成一个不可替代的闭环。