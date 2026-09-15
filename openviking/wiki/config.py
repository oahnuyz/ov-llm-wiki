"""Wiki 生成管线配置。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class WikiGenerationLimits:
    # 每批加入的新 card 数量；此前未分配的 card 额外结转。
    aggregation_batch_size: int = 25
    # 每批最大决策轮数；达到上限后保留状态并结束本批。
    aggregation_agent_max_turns: int = 40
    # 仅用于节点聚合 agent 每轮工具调用的最大输出 token 数。
    aggregation_agent_max_tokens: int = 12288
    # 层结束后按滑动窗口切分目录节点时的最大子 card 数量。
    max_cards_per_node: int = 1000
    # 同时发起多少个文档卡片生成请求。
    max_concurrent_cards: int = 10
    # 同时发起多少个节点内容生成请求。
    max_concurrent_nodes: int = 10
    # 单次 Wiki LLM 请求的超时秒数；超时后按统一策略重试。
    llm_request_timeout_seconds: float = 300.0


@dataclass
class WikiConfig:
    # 写入产物中的管线版本标识。
    pipeline_version: str = "wiki_v2_doc_card"
    # 来源资源所在的根 URI，用来校验和记录引用来源。
    resource_root_uri: str = "viking://resources/"
    # Wiki 产物写入的根 URI。
    wiki_root_uri: str = "viking://wiki/"
    # 控制聚合 agent 和 LLM 请求并发量。
    limits: WikiGenerationLimits = field(default_factory=WikiGenerationLimits)
    # 传给底层 VLM/LLM 的模型配置。
    vlm_config: dict[str, Any] | None = None
