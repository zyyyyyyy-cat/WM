# E1 实验：M-SpecGene 预训练权重迁移至 UAD 的完整修改指导

实验名称：E1 — Spectral Foundation UAD

实验目标： 在不改变原 UAD 网络结构、损失函数、采样策略和训练流程的情况下，将 ImageNet 预训练的 ViT-B/16 替换为 M-SpecGene 多光谱预训练的 ViT-B/16，检验这种视觉知识来源是否有利于 WHU-MARS 的身份表征学习。

本轮我已重新核对：

- M-SpecGene 原论文第 4 节的预训练与下游迁移机制；
- M-SpecGene 官方公开的 ViT-B 权重、转换脚本及参数命名；
- UAD 原论文第 4 节的共享主干、ProCA 和 GPD；
- UAD 官方 `vit_pytorch.py`、`make_model.py`、数据读取及训练代码。

## 一、正式实施前的方案审查

方案可实施，建议保持 E1 单变量设计

| 关键审查项      | 结论                        | 最终决定                               |
| ---------- | ------------------------- | ---------------------------------- |
| 架构是否兼容     | 两者均为标准结构的 ViT-B/16，主要维度一致 | 保留原 UAD ViT，不替换整个网络                |
| 是否有公开预训练模型 | 作者已公开原始 ViT-B 权重及转换权重     | 使用官方权重，不自行重新预训练                    |
| 是否需要引入新模块  | 不需要                       | 不增加 CMSS、Decoder、Attention 分支或辅助损失 |
| 是否存在迁移风险   | 有，尤其是参数名称、位置编码和输入分布       | 必须完成权重映射及加载验证                      |
| 是否需要修改训练策略 | E1 暂时不需要                  | 完整保留原 UAD 训练配置                     |
| 是否必然改善 TIR | 无法保证                      | 通过完整训练和同协议检索比较验证                   |

### 1.1 本次实验究竟借鉴了什么？

核心参考论文：

Zhou et al., M-SpecGene: Generalized Foundation Model for RGBT Multispectral Vision, ICCV 2025

- 第 4 节、图 3(a)： 使用共享权重 ViT、双模态自监督重建学习跨光谱视觉表征。
- 第 4.3 节： GMM-CMSS 渐进式 Masking，改善多光谱预训练的信息选择。
- 第 4.4 节、图 3(b)： 将预训练后的 ViT Encoder 迁移到下游任务。
- 第 5 节、表 7： 验证其预训练表征相对其他预训练方式在所测下游任务中的效果。

  M-SpecGene(1).pdf



我们实际借鉴的是第 4.4 节的 Encoder 迁移思想，并利用第 4.1—4.3 节训练所得的预训练权重。

不是重新把 GMM-CMSS 加入 UAD。

### 1.2 审查中最重要的一项限制

M-SpecGene 原论文的下游 RGB-T 任务通常同时处理 RGB 和 Thermal 特征，再进行特征融合；而我们的 AS-ReID 要求：

\\[ x\longrightarrow F\_\theta(x)\in\mathbb R^{768}. \\]

也就是单张图像输入、单个共享 Encoder、单个 CLS 特征检索。

因此，这次迁移利用的是 M-SpecGene 已学习的视觉先验，而不是完整复制原论文的双模态推理方式。

如果迁移后取得收益，可以说多光谱预训练初始化有助于该任务；但不能直接将原论文的双模态融合效果视为我们单图 ReID 中已得到验证的能力。

这一限制并不妨碍 E1，反而使实验边界更加清楚。

## 二、E1 最终网络与训练设计

M-SpecGene ViT-B/16 预训练权重

仅提取共享 Encoder

原 UAD 共享 ViT-B/16

256×128 输入，12 Blocks，768 维 CLS

原 UAD 身份训练

CE + WRT + ProCA + GPD

归一化 pre-BN CLS：768 维

RGB / NIR / TIR × Ground / Aerial 混合 Gallery

E1 的训练目标仍然是：

\\[ \boxed{ \mathcal L\_{\mathrm{E1}} = \mathcal L\_{\mathrm{CE}} + \mathcal L\_{\mathrm{WRT}} + 0.01\mathcal L\_{\mathrm{ProCA}} + \mathcal L\_{\mathrm{GPD}} } \\]

没有任何新的损失项。

按照 UAD 原论文第 4.2—4.5 节，WRT 负责 Batch 内身份排序，ProCA 负责模态中心与身份中心的聚合，GPD 利用全局身份原型进行判别学习。E1 不应改变这些机制。

WHU(2).pdf



这样实验变量只有：

\\[ \boxed{ \theta\_{\mathrm{ViT}}^{(0)}: \quad \theta\_{\mathrm{ImageNet}} \longrightarrow \theta\_{\mathrm{M\text{-}SpecGene}} } \\]

而不是更换整个 UAD 学习框架。

## 三、官方模型获取：推荐使用哪一个权重？

官方项目：[M-SpecGene GitHub](https://github.com/CalayZhou/M-SpecGene?utm_source=chatgpt.com)

作者提供了不同用途的模型文件。为了避免选错权重，我建议如下：

| 模型文件                                 | 内容                          | E1 是否使用           |
| ------------------------------------ | --------------------------- | ----------------- |
| `M-SpecGene_VIT-B.pth`               | 原始多光谱自监督预训练模型               | 可以，需提取并转换 Encoder |
| `M-SpecGene_VIT-B_det_transform.pth` | 从预训练模型提取、转换的检测用 ViT Encoder | 推荐作为迁移起点          |
| `M-SpecGene_VIT-B_seg_transform.pth` | 用于分割框架的预训练 Encoder 转换版本     | 可参考，不作为首选         |
| `FLIR_iter_90000.pth` 等              | 已针对具体数据集微调的检测模型             | 不使用               |

推荐权重：M-SpecGene_VIT-B_det_transform.pth

这是预训练 Encoder 的格式转换版本，而不是在 FLIR 等数据集上训练完成的检测模型。

[官方转换权重下载](https://drive.google.com/file/d/111OG0Ejv8pd8nLdLs74f1rq7NgvSTd8e/view?utm_source=chatgpt.com)

[官方原始预训练权重下载](https://drive.google.com/file/d/1COVlrnQPoqFjK-aJpSvNb3diSDblw0nb/view?utm_source=chatgpt.com)

[官方权重转换脚本](https://github.com/CalayZhou/M-SpecGene/blob/main/tool/M-SpecGeneTransform_det.py?utm_source=chatgpt.com)

推荐检测用 Encoder 转换文件的原因，是作者已经将部分 MMPretrain 命名转换为标准 ViT 命名，更接近 UAD 使用的 TransReID 模型。

但它仍不是一份已经完成 UAD 适配和验证的权重。 使用前必须检查实际文件中的 `model` 字典、每个参数的名称及形状。

如果转换文件与原始权重的结构不一致，优先回到原始 `M-SpecGene_VIT-B.pth`，按照官方 `tool/M-SpecGeneTransform_det.py` 和 `tool/model_keys.json` 的规则重新生成 Encoder 权重。

## 四、需要修改的代码文件

建议采用最小侵入式修改。

| 文件或模块                                             | 操作                              | 必要性  |
| ------------------------------------------------- | ------------------------------- | ---- |
| 新建 `tools/convert_mspecgene_to_uad.py`            | 将官方预训练权重转换为 UAD 主干的参数格式，验证映射与形状 | 必须   |
| `model/make_model.py`                             | 增加 `mspecgene` 预训练来源的加载分支       | 必须   |
| `model/backbones/vit_pytorch.py`                  | 复用位置编码插值；仅在现有函数不能正确处理时修改        | 按需   |
| `configs/UAD.yml` 或新建 `configs/UAD_MSpecGene.yml` | 设置新权重路径和预训练来源                   | 必须   |
| `processor/processor.py`                          | 原训练逻辑不变，仅建议输出初始化来源及训练元信息        | 非必要  |
| `datasets/bases.py`                               | 不修改                             | 保持原样 |
| `datasets/make_dataloader.py`                     | 不修改                             | 保持原样 |
| `loss/make_loss.py`                               | 不修改                             | 保持原样 |
| `loss/wrt_loss.py`                                | 不修改                             | 保持原样 |
| `datasets/sampler_ddp.py`                         | 不修改实验逻辑                         | 保持原样 |

其中，`tools/convert_mspecgene_to_uad.py` 是建议新建的项目脚本，并非 M-SpecGene 作者已经提供的同名文件。

### 为什么不直接在 `vit_pytorch.py` 中写全部转换逻辑？

因为权重转换和模型训练是两件事。

更清楚的组织方式是：

一次性转换

`M-SpecGene 官方权重` → `E1 专用 UAD Backbone 权重`

每次训练

`E1 专用权重` → `构建原 UAD 模型` → `正常训练`

这样每次正式训练不需要重新执行参数命名转换，也能防止不同训练进程使用不同的转换结果。

## 五、最重要的部分：权重转换与加载

### 步骤 1：提取共享 ViT Encoder

论文依据：M-SpecGene 第 4 节、第 4.4 节；官方 `tool/M-SpecGeneTransform_det.py`。

保留以下参数：

- Patch Embedding；
- CLS Token；
- Position Embedding；
- 12 个 Transformer Blocks；
- 各 Block 的 QKV Attention、FFN、LayerNorm；
- 最后一层 LayerNorm。

不需要迁移：

- GMM-CMSS 采样器；
- MAE 重建 Decoder；
- 预训练重建 Head；
- 仅在预训练阶段使用的其他辅助组件。

因为 E1 只需要使用作者训练好的 Encoder，而不是重新执行多光谱掩码重建。

### 步骤 2：完成参数名称映射

这是最容易出现隐性错误的位置。

M-SpecGene 使用 MMPretrain 命名，UAD 使用另一套 ViT 命名。

| M-SpecGene 原始参数                          | UAD 参数                    |
| ---------------------------------------- | ------------------------- |
| `backbone.patch_embed.projection.weight` | `patch_embed.proj.weight` |
| `backbone.patch_embed.projection.bias`   | `patch_embed.proj.bias`   |
| `backbone.cls_token`                     | `cls_token`               |
| `backbone.pos_embed`                     | `pos_embed`               |
| `backbone.layers.0.ln1.*`                | `blocks.0.norm1.*`        |
| `backbone.layers.0.attn.qkv.*`           | `blocks.0.attn.qkv.*`     |
| `backbone.layers.0.attn.proj.*`          | `blocks.0.attn.proj.*`    |
| `backbone.layers.0.ln2.*`                | `blocks.0.norm2.*`        |
| `backbone.layers.0.ffn.layers.0.0.*`     | `blocks.0.mlp.fc1.*`      |
| `backbone.layers.0.ffn.layers.1.*`       | `blocks.0.mlp.fc2.*`      |
| `backbone.ln1.*`                         | `norm.*`                  |

上述 Block 0 的规则需要应用到 Block 0—11。

作者提供的 `model_keys.json` 可作为转换参数清单参考。

必须注意：这些是参数名称映射规则，具体还要以实际下载的 Checkpoint 中的键和张量形状为准。

特别要检查 Attention 的 QKV 权重及 Bias 是否完整，不能只加载 Attention 的输出投影。

### 步骤 3：适配位置编码

这是第二个关键点。

M-SpecGene 预训练配置采用 224×224 输入、16×16 Patch：

\\[ 14\times14=196\text{ 个 Patch}. \\]

加上 CLS Token：

\\[ N\_{\mathrm{pretrain}}=197. \\]

UAD 使用 256×128 输入、16×16 Patch：

\\[ 16\times8=128\text{ 个 Patch}. \\]

加上 CLS Token：

\\[ N\_{\mathrm{UAD}}=129. \\]

因此需要：

\\[ [1,197,768] \longrightarrow [1,129,768]. \\]

推荐直接复用 UAD `vit_pytorch.py` 中已经具有的 `resize_pos_embed()` 思路：

1. 单独保留 CLS 位置编码；
2. 将其余位置编码还原为原 14×14 网格；
3. 双线性插值到 16×8 网格；
4. 重新展平并拼接 CLS 位置编码。

不要简单裁剪 197 个 Token 为 129 个，也不要丢弃全部位置编码重新随机初始化。

若实际 Checkpoint 的位置编码数量不是 197，必须根据实际元数据确认旧网格形状，不能强行按 14×14 重排。

还需要注意：M-SpecGene 的 MAE 预训练位置编码是固定的正弦余弦位置表示，而 UAD 的位置编码属于可训练参数。迁移时复制数值即可，第一版按原 UAD 的方式继续参与微调，不额外冻结位置编码。

### 步骤 4：严格检查参数加载

这是我认为 E1 实现中最需要认真执行的工程要求。

UAD 原始 `load_param()` 使用异常捕获处理部分不匹配参数，打印错误后可能继续执行。

因此可能出现：

> 日志显示程序已经开始训练，但实际上有若干 Transformer Block 没有加载成功。

建议转换脚本在正式训练前完成以下检查：

| 检查项                | 必须达到的要求                                 |
| ------------------ | --------------------------------------- |
| Patch Embedding    | Weight、Bias 完整且形状一致                     |
| Transformer        | 12 个 Block 的 Attention、FFN 和 Norm 均成功映射 |
| QKV                | Weight、Bias 完整                          |
| 最终 LayerNorm       | 成功加载                                    |
| CLS Token          | 成功加载                                    |
| Position Embedding | 正确插值并加载                                 |
| 其他非主干参数            | 允许 UAD 的分类器及 BNNeck 重新初始化               |

对于上述主干参数，不允许仅打印警告后继续训练。关键参数缺失或形状错误，应直接停止。

转换完成后保存一份加载报告，至少记录：

- 成功加载的参数个数与名称；
- 未加载参数及原因；
- 每一组关键参数的形状；
- 整体 Backbone 参数加载比例；
- 来源 Checkpoint 的文件名及哈希值。

这不是额外的研究诊断，而是确保 E1 这个实验确实在使用 M-SpecGene 视觉知识。

### 步骤 5：重新初始化身份相关组件

M-SpecGene 没有为 WHU-MARS 的 500 个训练身份提供分类器。

因此：

| 组件            | 初始化方式            |
| ------------- | ---------------- |
| 共享 ViT 主干     | 来自 M-SpecGene    |
| 分类器           | 原 UAD 随机初始化      |
| BNNeck        | 原 UAD 初始化        |
| ProCA         | 原计算方式            |
| GPD 原型 Memory | 原 UAD 的初始状态和更新流程 |

不应从 M-SpecGene 的检测或分割任务中加载已经训练好的 Task Head。

CLS Token 本身可以从 M-SpecGene 迁移，但身份分类器必须重新学习。

## 六、输入数据与预处理：保持单变量，同时注意光谱适配

参考依据：M-SpecGene 预训练数据配置、UAD `datasets/bases.py` 和 `datasets/make_dataloader.py`。

这一部分很容易让 E1 从“更换初始化实验”变成“初始化＋预处理联合实验”。

### 6.1 输入通道

M-SpecGene 的 ViT-B/16 使用三通道 Patch Embedding；UAD 数据读取函数使用：

`Image.open(img_path).convert('RGB')`

因此，E1 不需要修改 Patch Embedding 的输入通道数。

需要确认真实数据经过读取后均为：

\\[ [3,256,128]. \\]

尤其注意：

- RGB：保留原有 RGB 三通道；
- NIR：继续使用原 UAD 的三通道读取结果；
- TIR：继续使用原 UAD 的三通道读取结果。

如果原始 NIR/TIR 是灰度图，PIL 可能将灰度值复制到三个通道；如果原始文件是伪彩色图，不能假设三个通道一定相同。

这次不专门将 TIR 改为灰度复制、增加伪彩色转换或改变热红外亮度映射。

这些都属于新的输入适配实验，不是 E1 的必要组成部分。

### 6.2 输入归一化

这里存在一个明确的来源差异。

M-SpecGene 原预训练配置采用近似 ImageNet 标准化：

\\[ \mu=[0.485,0.456,0.406], \\]

\\[ \sigma=[0.229,0.224,0.225]. \\]

而 UAD 官方 `configs/UAD.yml` 中为：

\\[ \mu=[0.5,0.5,0.5], \qquad \sigma=[0.5,0.5,0.5]. \\]

两者不同。

我的明确建议：E1 第一轮保持 UAD 的归一化设置不变。

原因是本轮只需要回答：

> 在当前 UAD 训练协议下，使用 M-SpecGene 初始化是否比现有 ImageNet 初始化更好？

如果同时修改归一化，就不能单独归因于初始化。

这并不意味着归一化差异不重要。它确实可能影响预训练知识向 WHU 的迁移效果，所以需要在实验记录中注明。

若 E1 效果不理想，可以后续将输入标准化单独作为适配因素分析，但不要提前把多个因素混在第一轮中。

### 6.3 数据增强

全部沿用原 UAD：

| 数据处理                   | E1 |
| ---------------------- | -- |
| Resize 256×128         | 保持 |
| Random Horizontal Flip | 保持 |
| Padding + Random Crop  | 保持 |
| Random Erasing         | 保持 |
| 输入归一化                  | 保持 |
| RGB/NIR/TIR 训练比例       | 保持 |

不增加 M-SpecGene 的 90% Masking。

90% Masking 是 M-SpecGene 的预训练策略，不是下游 ReID 微调时必须执行的操作。

在 WHU 身份训练阶段再次遮挡 90% Patch，反而可能删除原本已经稀缺的人体身份线索，而且会构成完全不同的实验。

## 七、配置文件：只改变预训练来源

建议新建：

`configs/UAD_MSpecGene.yml`

以免覆盖已验证的原 UAD 配置。

主要配置如下。

| 配置项                       | E1 设置                   | 说明                |
| ------------------------- | ----------------------- | ----------------- |
| `MODEL.PRETRAIN_CHOICE`   | `mspecgene`             | 新增预训练来源           |
| `MODEL.PRETRAIN_PATH`     | 转换后的 E1 权重路径            | 必须指向实际 Encoder 权重 |
| `MODEL.TRANSFORMER_TYPE`  | `vit_base_in`           | 不变                |
| `MODEL.STRIDE_SIZE`       | `[16,16]`               | 不变                |
| `INPUT.SIZE_TRAIN`        | `[256,128]`             | 不变                |
| `INPUT.PIXEL_MEAN / STD`  | 原 UAD 的 `[0.5,0.5,0.5]` | 不变                |
| `DATALOADER.SAMPLER`      | `PKM`                   | 不变                |
| `DATALOADER.NUM_INSTANCE` | `4`                     | 不变                |
| `SOLVER.MAX_EPOCHS`       | `120`                   | 不变                |
| `MODEL.PCA_LOSS_WEIGHT`   | `0.01`                  | 不变                |
| `SOLVER.LOSS_TYPE`        | `base+pca+gpd`          | 不变                |
| `TEST.NECK_FEAT`          | `before`                | 不变                |
| `TEST.FEAT_NORM`          | `yes`                   | 不变                |
| `TEST.RE_RANKING`         | `False`                 | 不变                |
| `OUTPUT_DIR`              | E1 单独输出目录               | 防止覆盖 B0           |

这里配置值来自官方示例，但最终应以你已复现成功的 B0-DDP 实际配置为准。如果本地 B0 的参数与官方 YAML 不完全相同，以本地 B0 为比较基准，不要借 E1 顺便重置优化器、学习率或 Batch Size。

### 必须同步修改 `make_model.py`

官方 `model/make_model.py` 中目前仅对：

`PRETRAIN_CHOICE == 'imagenet'`

执行预训练加载。

因此，如果只是把配置改为 `mspecgene`，而没有新增对应加载逻辑，模型可能根本不会加载预训练权重。

新增的 `mspecgene` 分支需要完成：

1. 读取已转换的 Encoder 权重；
2. 加载到 `self.base`；
3. 检查主干参数是否全部符合要求；
4. 记录成功加载信息；
5. 正常初始化原 UAD 分类器和 BNNeck。

原 `forward()` 不需要修改。

也不需要在训练过程每个 Epoch 重新加载 M-SpecGene 权重。

## 八、训练流程：完全沿用 UAD

训练阶段：

\\[ \boxed{ \text{M-SpecGene 初始化} \rightarrow \text{UAD 120 Epoch 联合训练} \rightarrow \text{单图 CLS 推理} } \\]

### 8.1 不冻结主干

我建议 E1 采用全参数微调。

理由是：

M-SpecGene 的预训练目标主要是多光谱视觉结构重建，而 WHU 的目标是身份分类与跨模态检索。

需要让整个 ViT 学会将这些视觉先验转化为身份判别特征。

如果冻结大部分主干，只训练分类器或末端层，可能无法完成充分的任务适配。

而且额外设计冻结阶段会改变 B0 的学习流程。

所以 E1 不使用冻结前几层、LoRA 或特殊分层学习率。

### 8.2 保留原优化器

M-SpecGene 在预训练中使用 AdamW，不代表迁移到 UAD 后也必须使用 AdamW。

当前 E1 使用的是预训练权重，而不是继续 M-SpecGene 的自监督预训练任务。

因此，训练优化器、学习率调度、Warm-up、梯度裁剪和混合精度均沿用 B0。

### 8.3 保留 UAD 的 ProCA 和 GPD

这一点不能因为换了 Backbone 就删除。

因为原 UAD 不只是一个 ViT 分类器，它的完整方法还包括：

- WRT：Batch 内困难正负样本排序；
- ProCA：同身份模态中心聚合；
- GPD：全局身份原型判别。

如果 E1 删除 ProCA 或 GPD，就无法判断初始化是否对完整 UAD有效。

同时，官方代码中的 `PCA()` 函数实际上实现的是论文所述 ProCA，而不是传统统计学的主成分分析。

GPD 的 `GPD_MOMENTUM=0.2` 是代码中新中心的更新权重，对应旧原型权重 0.8；不应把它误读为需要改为 0.8 才与论文一致。

## 九、DDP 的额外注意事项

DDP 沿用当前已成功运行的 UAD-DDP，不重新设计分布式训练。

只需特别注意四点。

### 9.1 权重必须在 DDP 包装前加载

正确顺序：

`创建 UAD 模型 → 加载 M-SpecGene Encoder → 构建 DDP 训练 → 正常优化`

避免先创建 DDP，再由某个 Rank 单独修改 Backbone 参数。

### 9.2 避免不同 Rank 使用不一致的初始化

建议所有 Rank 使用同一份已经转换好的权重文件，DDP 包装前完成加载。

转换过程只进行一次，不要让每张 GPU 在训练时独立生成权重文件。

Rank 0 统一输出完整加载报告即可。

### 9.3 核对实际迭代次数

此前 PKM-DDP 的实验已经发现过 Sampler 长度和实际采样迭代数之间的不一致问题。

E1 最重要的是：

不要在更换初始化的同时修改 PKM 采样器行为。

记录 B0/E1 每 Epoch 的实际优化步数，而不只看 `len(train_loader)`。

如果决定修复 Sampler 的长度问题，就需要在 B0 和 E1 中同时使用修复后的协议，不能只修改 E1，否则训练样本暴露量和学习率进度可能不再一致。

### 9.4 保持原型更新顺序

原 UAD 在当前 Batch 中计算 ProCA、GPD，完成反向传播和优化后再更新 PrototypeBank。

E1 不修改这一时序，也不新增跨 Rank 的原型同步逻辑。

## 十、正式开始 120 Epoch 前的必要检查

这些不是额外研究实验，而是一次性工程验收，确保运行的 E1 确实符合设计。

| 检查阶段   | 检查内容                         | 通过标准               |
| ------ | ---------------------------- | ------------------ |
| 权重检查   | 来源是否为原始预训练 Encoder           | 未误用检测微调模型          |
| 参数检查   | 12 Blocks、Patch、CLS、Norm、QKV | 关键参数无缺失、无形状错误      |
| 位置编码   | 197 → 129                    | 正确插值，输出形状一致        |
| 输入检查   | RGB/NIR/TIR                  | 均为 `[B,3,256,128]` |
| 前向检查   | 输出 CLS                       | `[B,768]`，数值有限     |
| 反向检查   | CE/WRT/ProCA/GPD             | 能正常求梯度与更新主干        |
| DDP 检查 | 多卡训练                         | 各 Rank 正常完成同步      |
| 推理检查   | 原 UAD 测试路径                   | 仍输出归一化 pre-BN CLS  |

这里还建议进行一个很轻量的权重检查：

在训练开始前，对比转换后 Backbone 的关键张量与原始 M-SpecGene 对应张量，确认不是只匹配了名称而没有正确复制数值。

如果条件允许，再用单张固定图像比较原始 Encoder 与转换后 Encoder 的推理输出。只有在输入、位置编码、输出接口完全一致时，才应期待数值接近；这属于可选的一次性转换验证，不是新的训练实验。

值得注意的是，M-SpecGene 原 `MAEViT.forward()` 中，非 Mask 前向的条件应按源码实际行为调用；如果使用原 MAEViT 做输出对照，`mask=None` 才是安全的非 Mask 路径，不能仅根据注释假设传 `mask=False` 也一定正确。


## 十二、建议清单

阶段 A：准备权重

- 下载官方 M-SpecGene ViT-B Encoder 转换权重。
- 核实 Checkpoint 结构及来源，确认不是检测任务微调权重。
- 建立 UAD 主干参数映射，正确转换位置编码。
- 验证所有关键 Backbone 参数的加载结果并保存报告。

阶段 B：最小代码修改

- 新增 `mspecgene` 初始化选项。
- 修改 `make_model.py`，支持可靠加载。
- 创建独立的 E1 配置与输出目录。
- 不修改 UAD 模型前向、损失、采样器和数据增强。

阶段 C：工程验收

- 检查输入通道、归一化、Patch 网格和 CLS 输出。
- 验证一次前向、反向及 DDP 训练步骤。
- 确认模型主干已实际加载 M-SpecGene 参数，而非大部分随机初始化。

阶段 D：正式训练与评估

- 使用与 B0 完全一致的 WHU 训练协议，完成 120 Epoch。
- 保存模型、配置、参数加载报告及训练日志。
- 输出总体 AS-ReID、九组模态方向和四组平台方向的结果。
- 对比 B0，判断预训练初始化是否有实际身份判别收益。

## 最终审查结论

E1 不需要设计新模块，也不应该额外增加任何损失。

本轮方案的核心只有：

\\[ \boxed{ \text{M-SpecGene Encoder 权重} \rightarrow \text{原 UAD ViT-B/16} \rightarrow \text{完整原 UAD 身份训练} } \\]

最需要把握的是三件事：

第一，迁移必须真实有效。 参数命名、QKV、FFN、LayerNorm、CLS 和位置编码都必须正确加载，不能让实验意外退化为部分随机初始化。

第二，训练必须严格可比。 保持本地已复现成功的 UAD-DDP 训练与测试协议，不能因为换了预训练来源，就同时改变输入归一化、Sampler、损失或优化器。

第三，评价必须关注真正的身份学习能力。 不只看总体 mAP，也要看 TIR 内部身份判别、RGB/NIR↔TIR 跨模态匹配及跨平台检索是否改善。

因此，E1 目前已经具备明确的实施路径。建议按上述范围完成最小修改后，进行一次加载与训练步骤验收，再直接进入正式训练，不需要继续引入教师、Adapter、Masking 或新的身份损失。