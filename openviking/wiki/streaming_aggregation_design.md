# Wiki 流式、逐层向上聚合方案（第一版）

本文描述新的 Wiki 聚合管线设计。第一版的目标是：文档先生成 card，然后按固定批次从下到上聚合；每一层使用同一套候选节点操作协议，不再由配置限制层数，也不再额外调用模型判断“是否继续向上聚合”。

本文是设计大纲，当前不包含代码改动。

## 1. 核心原则

1. **流式处理**：当前层的 card 按每批 15 个顺序输入模型。下一批只能看到上一批操作执行后的候选状态。
2. **候选状态由代码维护**：模型只返回严格的操作 JSON；候选列表的创建、更新、删除和最终物化由固定程序完成。
3. **操作按数组顺序执行**：同一响应中的后续操作可以引用前面操作创建或修改的候选。程序只有在全部操作校验成功后才提交本批结果。
4. **候选可修正**：候选可以被分配 card、重命名、修改 scope、合并或拆分；候选不是不可变的最终节点。
5. **允许多重归属**：一个 card 可以同时属于多个候选节点，因此最终 Wiki 结构可以是 DAG，而不是严格树。
6. **状态由成员数推导**：候选包含 1 个不同的子 card 时为“待定（pending）”；包含至少 2 个不同的子 card 时为“暂定（provisional）”。模型不能直接设置状态。
7. **只向上传递新聚合节点**：一层结束后，待定候选被删除；其唯一 card 不透传。下一层只接收本层新生成的 Node Card。
8. **自然终止**：某层清理后没有任何暂定候选时，候选列表为空，不生成 Node Card，向上聚合结束。
9. **层间必须有收缩**：为避免多重归属造成层级不收缩，程序要求一层最终的 provisional 候选数严格少于该层输入 card 数；不满足时视为模型输出无效并重试。

## 2. 层级和术语

### 2.1 Layer 0

Layer 0 是原始资源对应的 Document Card 集合。原始 card 仍写入 Wiki card 存储，并作为第一层聚合输入，但不要求每个原始 card 都生成一个独立 WikiNode。

### 2.2 Layer 1 及更高层

- Layer 1 输入：原始 Document Card。
- Layer 2 输入：Layer 1 暂定候选生成的 Node Card。
- Layer N 输入：Layer N-1 暂定候选生成的 Node Card。

每一层都执行完全相同的“分批输入 → 操作执行 → 候选清理 → 节点物化”流程。`depth` 只记录节点所在层级，不再作为最大层数限制。

### 2.3 候选状态

```text
pending（待定）      card_ids 去重后数量为 1
provisional（暂定）  card_ids 去重后数量 >= 2
```

待定候选只是当前层内的临时占位：后续批次可以继续向其中分配 card，使其变为暂定。所有批次结束后，仍为待定的候选直接删除，不生成 WikiNode，也不把其唯一 card 送入下一层。

## 3. 总体流程

```text
WikiService.build_wiki
  ↓
加载资源 → ResourceDocument
  ↓
逐文档生成原始 DocumentCard（Layer 0）
  ↓
写入原始 card
  ↓
current_layer_cards = 原始 DocumentCard
depth = 1
  ↓
┌─ while current_layer_cards 非空 ───────────────────────────┐
│  按 15 个 card 切分为批次（批次之间串行）                  │
│    existing_candidates + current_batch_cards               │
│      → LLM 只返回 {"operations": [...]}                   │
│      → 程序按顺序校验并执行操作                             │
│  所有批次结束                                               │
│    → 删除 pending 候选                                      │
│    → 保留 provisional 候选                                  │
│  若 provisional 为空：停止向上聚合                          │
│  否则：                                                      │
│    → provisional 候选生成 WikiNode、正文和 Node Card        │
│    → current_layer_cards = 新生成的 Node Card               │
│    → depth += 1                                             │
└────────────────────────────────────────────────────────────┘
  ↓
写入运行记录，返回 PipelineArtifacts / 服务结果
```

## 4. 各阶段数据结构

以下字段为第一版建议的逻辑结构。现有代码中的 `DocumentCard.doc_id` 可以继续作为 card ID 使用；文档中称为 `card_id` 的地方，在实现时可映射到该字段，避免随机 source ID 由模型重新生成。

### 4.1 Wiki 构建入口

`WikiService.build_wiki` 的输入：

```json
{
  "resource_uris": ["viking://resources/..."],
  "wiki_root_uri": "viking://wiki/",
  "card_input_mode": "summary",
  "max_card_input_chars": 20000
}
```

其中 `ctx` 为代码侧请求上下文，不进入模型 JSON。`resource_uris` 是待处理资源；其余字段控制资源加载和输出位置。

### 4.2 ResourceDocument

`ResourceDocument` 由 `WikiContentLoader` 从已入库资源目录组装，不是 LLM 生成的。`summary` 模式下，程序读取各叶子条目的语义摘要；`.abstract.md` 和 `.overview.md` 等隐藏语义文件不会作为普通条目再次读入，原始 chunk 正文也不会用于生成这一模式下的 Document Card。若没有可用语义摘要，加载会失败并提示改用 `raw_chunk` 模式。

资源加载阶段的输出：

```json
{
  "doc_id": "doc_001",
  "resource_uri": "viking://resources/paper_001",
  "title": "论文标题",
  "content_or_structure": "...",
  "source_sections": [
    {
      "section_uri": "viking://resources/paper_001#section-1",
      "content": "..."
    }
  ],
  "metadata": {}
}
```

主要用途：生成 Document Card，以及后续生成节点正文时提供可引用的原始内容。`doc_id` 是代码侧稳定标识。

### 4.3 DocumentCard

每篇资源或每个已生成 Wiki 节点都有一个 card：

```json
{
  "doc_id": "doc_001",
  "resource_uri": "viking://resources/paper_001",
  "title": "论文标题",
  "summary": "...",
  "main_points": ["...", "..."],
  "important_terms": ["..."],
  "candidate_topics": ["Memory-Efficient Training", "Optimizer Design"],
  "markdown": "..."
}
```

字段说明：

- `doc_id`：代码生成或继承的稳定 card ID；模型不得改写。
- `title`、`summary`、`main_points`、`important_terms`：card 的语义内容。
- `candidate_topics`：来自 card 的宽泛主题提示，只是聚合参考，不是候选节点的正式边界。
- `markdown`：写入 card 文件的展示内容。

### 4.4 LLM 聚合请求

每一批只发送以下顶层字段：

```json
{
  "existing_candidates": [],
  "current_batch_cards": []
}
```

#### `existing_candidates`

候选列表由程序物化后传入模型。程序内部仍完整维护候选到 card ID 的映射；模型输入则只给历史成员的精简摘要视图：

```json
[
  {
    "candidate_id": "candidate_0001",
    "title": "Optimizer State Quantization",
    "scope": "覆盖训练阶段优化器状态的低比特量化及其内存、稳定性影响；不覆盖权重量化和推理优化。",
    "cards": [
      {"card_id": "doc_001", "summary": "该 card 的摘要"},
      {"card_id": "doc_004", "summary": "该 card 的摘要"}
    ],
    "status": "provisional"
  }
]
```

- `candidate_id`：代码生成的持久候选 ID，模型只能引用，不能自行改变。
- `title`：当前面向读者的候选名称，可通过 `rename_candidate` 修改。
- `scope`：当前候选的知识边界，包括覆盖和排除内容，可通过 `update_scope` 修改。
- `cards`：当前候选的历史成员精简视图。每个成员只保留 `card_id` 和 `summary`，不重复携带完整 card、正文、URI 或 markdown。
- `status`：代码根据内部 card ID 集合的数量推导的只读字段；模型不能在操作中设置。

代码内部仍维护去重后的 `card_ids` 集合，用于操作校验、状态计算、合并、拆分和最终 SourceRef 构建；`cards` 只是模型可见的压缩表示。

`candidate_topics` 不作为候选的权威状态字段；它保留在 `current_batch_cards` 中，作为模型判断主题归属的启发式提示。

#### `current_batch_cards`

数组中的元素是当前批次的聚合专用 card 视图。当前批次可以携带尽可能多的语义信息，但不携带原始正文和存储元数据：

```json
[
  {
    "card_id": "doc_005",
    "title": "...",
    "summary": "...",
    "main_points": ["..."],
    "important_terms": ["..."],
    "candidate_topics": ["..."]
  }
]
```

一批最多 15 个元素。批次之间不并行；下一批的 `existing_candidates` 必须是上一批成功提交后的状态。历史 card 不会再次出现在 `current_batch_cards` 中，只会以候选成员的 `card_id + summary` 形式出现在 `existing_candidates` 中。

### 4.5 LLM 聚合响应

模型只能返回：

```json
{
  "operations": [
    {
      "op": "assign_cards",
      "candidate_id": "candidate_0001",
      "card_ids": ["doc_005"]
    }
  ]
}
```

顶层不能有 `nodes`、`unassigned_source_ids`、自然语言解释或其他字段。每个 operation 只能使用其对应 schema 声明的字段，禁止额外字段。

支持的操作：

#### `create_candidate`

```json
{
  "op": "create_candidate",
  "candidate_ref": "new_optimizer_topic",
  "title": "Optimizer State Quantization",
  "scope": "...",
  "card_ids": ["doc_005", "doc_006"]
}
```

代码在执行时生成真正的 `candidate_id`。`candidate_ref` 只是本次响应内的临时引用，供后续操作引用，不写入最终 Wiki 数据。

#### `assign_cards`

```json
{
  "op": "assign_cards",
  "candidate_id": "candidate_0001",
  "card_ids": ["doc_007"]
}
```

将 card 加入候选，采用集合并集语义，不会从其他候选移除 card；因此允许多重归属。

#### `rename_candidate`

```json
{
  "op": "rename_candidate",
  "candidate_id": "candidate_0001",
  "title": "Low-Bit Optimizer State Quantization"
}
```

#### `update_scope`

```json
{
  "op": "update_scope",
  "candidate_id": "candidate_0001",
  "scope": "..."
}
```

`scope` 是候选当前的正式知识边界，不等同于 `candidate_topics`。它用于判断后续 card 是否适合加入，以及指导节点正文和 Node Card 生成。

#### `merge_candidates`

```json
{
  "op": "merge_candidates",
  "target_candidate_id": "candidate_0001",
  "source_candidate_ids": ["candidate_0002", "candidate_0003"]
}
```

将 source 候选的全部 card 成员并入 target；target 的 title/scope 保留当前值，必要时可在后续 operation 中重命名或更新 scope；source 候选随后删除。合并不会丢失任何 card 成员。

#### `split_candidate`

```json
{
  "op": "split_candidate",
  "candidate_id": "candidate_0001",
  "groups": [
    {
      "candidate_ref": "optimizer_memory",
      "title": "Optimizer Memory Quantization",
      "scope": "...",
      "card_ids": ["doc_001", "doc_004"]
    },
    {
      "candidate_ref": "optimizer_stability",
      "title": "Optimizer Stability Under Quantization",
      "scope": "...",
      "card_ids": ["doc_004", "doc_008"]
    }
  ]
}
```

代码为每个 group 生成新候选 ID，并删除原候选。校验要求：原候选的每个 card 至少出现在一个 group 中；group 之间允许重叠，以保留多重归属。`candidate_ref` 仅用于同一响应内的后续引用。

## 5. 操作执行和校验

每批采用事务式执行：

```text
复制上一批候选状态
  ↓
按 operations 数组顺序逐项执行
  ↓
校验全部通过后提交
  ↓
任一项失败则整批回滚，并用同一输入重试
```

至少需要校验：

1. JSON 顶层和 operation 均符合严格 schema，禁止未知字段。
2. 引用的 `candidate_id` 已存在，或是前面 operation 创建的 `candidate_ref`。
3. 所有 `card_ids` 都存在于当前层已知 card 集合。
4. `merge_candidates` 不能把 target 同时作为 source；候选不能合并到自身。
5. `split_candidate` 不得遗漏原候选的任何 card。
6. 当前批次结束后，本批所有 card 至少属于一个候选；没有被归属的 card 必须由程序创建单 card pending 候选，或要求模型重试。推荐由固定程序兜底创建，以保证不丢 card。
7. 同一候选的 `card_ids` 去重后计算状态；重复 ID 不增加成员数。

模型应被明确告知：操作将按数组顺序执行，后续操作可以引用前面操作创建的候选；模型不得返回最终候选列表，只能返回操作。

## 6. 一层结束时的清理和物化

所有批次完成后，程序重新计算每个候选的状态：

```text
card_ids 数量 = 1  → pending，删除
card_ids 数量 >= 2 → provisional，保留
```

删除 pending 候选时，其唯一 card 不进入下一层。若清理后 provisional 候选为空：

```text
当前层不生成 WikiNode
不生成 Node Card
current_layer_cards 为空
整个向上聚合停止
```

若存在 provisional 候选，则每个候选物化为一个正式 `WikiNode`，并生成节点正文和 Node Card。

### 6.1 WikiNode

```json
{
  "node_id": "optimizer_state_quantization",
  "title": "Optimizer State Quantization",
  "depth": 1,
  "scope": "...",
  "parent_node_ids": [],
  "child_node_ids": []
}
```

- `node_id` 由代码根据 title 生成并去重，不由模型决定。
- `depth` 是当前层编号，从 1 开始递增。
- Layer 1 的候选 card 通常对应原始文档，`SourceRef.ref_type = "document"`。
- Layer 2+ 的候选 card 对应上一层 Node Card，`SourceRef.ref_type = "wiki_node"`，并将其 IDs 写入 `child_node_ids`。
- 新方案不再使用候选级别的 `rejected` 状态；pending 候选在清理时直接删除。正式 WikiNode 不再保留冗余的 `status="active"` 字段。

### 6.2 SourceRef 和来源分配

候选的 `card_ids` 由代码转换为节点来源引用：

```json
{
  "ref_id": "...",
  "ref_type": "document",
  "doc_id": "doc_001",
  "resource_uri": "viking://resources/paper_001",
  "card_uri": "viking://wiki/cards/doc_001.card.json",
  "title": "论文标题",
  "support_scope": "节点 scope",
  "matched_topics": []
}
```

同一个 card 如果属于多个候选，则在多个节点的 `source_refs_by_node` 中分别出现。

### 6.3 节点正文和 Node Card

节点正文生成输入（保留一个节点返回多个 Markdown 文档的能力）：

```text
WikiNode + 该节点的 SourceRef 对应 ResourceDocument / 子节点正文
```

一次 `node_documents` 调用默认返回一个文档；如果内容确实较长，可以返回多个 `NodeDocument`。输出：

```json
{
  "documents": [
    {
      "title": "Overview",
      "content": "..."
    }
  ]
}
```

`document_id` 不是 LLM 返回字段，而是程序在校验后按顺序生成（例如 `0001`、`0002`）。

随后生成一个 Node Card。Node Card 仍使用 `DocumentCard` 结构，其 `doc_id` 为当前 `node_id`，`resource_uri` 指向 Wiki 节点目录；它是下一层的输入 card。

## 7. 层间状态

每层只保留以下两类状态：

```text
current_layer_cards  当前层待处理输入（原始 card 或上一层 Node Card）
candidate_state      当前层跨批次累积的候选列表
```

层结束后：

```text
candidate_state 中的 provisional 候选
  → WikiNode + 节点正文 + Node Card
  → 下一层 current_layer_cards
```

pending 候选及其唯一 card 不进入下一层。候选状态不会跨层复用；下一层从上一层新生成的 Node Card 重新开始候选聚合。

## 8. 配置调整

第一版与聚合有关的配置建议为：

```python
class WikiGenerationLimits:
    aggregation_batch_size: int = 15
    max_concurrent_cards: int = 10
    max_concurrent_nodes: int = 10
```

- `aggregation_batch_size`：每批输入的 card 数量上限，第一版固定为 15；批次之间串行。
- `max_concurrent_cards`：原始文档 card 生成请求的最大并发数。
- `max_concurrent_nodes`：节点正文和 Node Card 生成请求的最大并发数。

新方案不再需要以下聚合控制：

- `max_depth`；
- `min_refs_per_node`；
- `min_child_nodes_per_parent`；
- `LayerDecisionRunner` / `next_layer_decision`；
- `continue_upward`；
- 候选的 `active/rejected` 数量过滤逻辑。

## 9. 最终 PipelineArtifacts

运行结束后继续提供统一产物：

```json
{
  "cards": ["所有原始 card 和已生成 Node Card"],
  "nodes": ["所有已物化 WikiNode"],
  "source_refs_by_node": {
    "node_id": ["SourceRef", "..."]
  },
  "node_contexts": ["节点、正文、Node Card 和来源的组合上下文"]
}
```

与旧方案相比，`nodes` 不再包含因来源数量不足而保留的 rejected 节点；未能和其他 card 聚合的 pending 候选不会出现在最终节点索引中。

## 10. 必须覆盖的测试场景

1. 第一批创建多个 pending 候选，并正确保留 15 个以内的 card。
2. 后续批次向已有候选分配 card，使 pending 变为 provisional。
3. 一个 card 同时属于多个候选。
4. rename、scope 更新和按顺序执行的多操作响应。
5. merge 后所有 card 成员取并集。
6. split 后原候选的每个 card 都至少保留在一个新候选中，支持 group 重叠。
7. 严格拒绝未知字段、未知 candidate ID、未知 card ID 和非法 split。
8. 一批操作失败时整批回滚并重试。
9. 所有批次结束后 pending 被删除，provisional 正确物化。
10. 一层全为 pending 时候选列表为空，立即停止且不生成更高层节点。
11. Layer 1 使用原始 Document Card，Layer 2+ 使用上一层 Node Card，但调用同一个聚合流程。
12. 93 篇或更多文档按 15 个一批顺序处理，不依赖固定最大层数。
