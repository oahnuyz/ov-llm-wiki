"""Wiki 生成管线配置。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class WikiGenerationLimits:
    # 聚合阶段每次发送给 LLM 的 card 数量上限。
    aggregation_batch_size: int = 15
    # 同时发起多少个文档卡片生成请求。
    max_concurrent_cards: int = 10
    # 同时发起多少个节点内容生成请求。
    max_concurrent_nodes: int = 10


@dataclass
class WikiConfig:
    # 写入产物中的管线版本标识。
    pipeline_version: str = "wiki_v2_doc_card"
    # 来源资源所在的根 URI，用来校验和记录引用来源。
    resource_root_uri: str = "viking://resources/"
    # Wiki 产物写入的根 URI。
    wiki_root_uri: str = "viking://wiki/"
    # 控制聚合批次大小和 LLM 请求并发量。
    limits: WikiGenerationLimits = field(default_factory=WikiGenerationLimits)
    # 传给底层 VLM/LLM 的模型配置。
    vlm_config: dict[str, Any] | None = None
