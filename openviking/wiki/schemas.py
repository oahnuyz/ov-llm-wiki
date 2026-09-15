"""Pydantic schemas used by the Wiki pipeline."""

from __future__ import annotations

import re
from typing import Annotated, Any, Literal, Union

from pydantic import AfterValidator, BaseModel, ConfigDict, Field

NODE_ID_RE = re.compile(r"^[a-z0-9_]+$")


def _require_text(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("value must be a non-empty string")
    return value.strip()


def _require_node_id(value: str) -> str:
    value = _require_text(value)
    if not NODE_ID_RE.match(value):
        raise ValueError("node_id must contain only lowercase letters, digits, and underscores")
    return value


def _require_resource_uri(value: str) -> str:
    value = _require_text(value)
    if not value.startswith("viking://resources/"):
        raise ValueError("resource_uri must start with viking://resources/")
    return value


def _require_optional_resource_uri(value: str) -> str:
    if not value:
        return value
    return _require_resource_uri(value)


def _require_source_uri(value: str) -> str:
    value = _require_text(value)
    if not (value.startswith("viking://resources/") or value.startswith("viking://wiki/")):
        raise ValueError("source uri must start with viking://resources/ or viking://wiki/")
    return value


def _require_wiki_uri(value: str) -> str:
    value = _require_text(value)
    if not value.startswith("viking://wiki/"):
        raise ValueError("resource_uri must start with viking://wiki/")
    return value


NonEmptyStr = Annotated[str, AfterValidator(_require_text)]
NodeId = Annotated[str, AfterValidator(_require_node_id)]
ResourceUri = Annotated[str, AfterValidator(_require_resource_uri)]
OptionalResourceUri = Annotated[str, AfterValidator(_require_optional_resource_uri)]
SourceUri = Annotated[str, AfterValidator(_require_source_uri)]
WikiUri = Annotated[str, AfterValidator(_require_wiki_uri)]
NonEmptyStrList = Annotated[list[str], Field(min_length=1)]
AtLeastTwoStrList = Annotated[list[str], Field(min_length=2)]


class StrictModel(BaseModel):
    """所有 Wiki 数据模型的基类，禁止接收未声明字段。"""
    model_config = ConfigDict(extra="forbid")


class ResourceDocumentDraft(StrictModel):
    """解析器列出的文档边界条目。"""
    doc_id: NonEmptyStr
    title: NonEmptyStr
    relative_uri: str = ""


class WikiResourceInput(StrictModel):
    """入库后传给 Wiki pipeline 的文档入口记录，用来定位资源并加载内容生成 Document Card。"""
    doc_id: NonEmptyStr
    resource_uri: ResourceUri
    title: NonEmptyStr
    document_dir_uri: OptionalResourceUri = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class SourceSection(StrictModel):
    """节点正文生成使用的来源片段。"""
    section_uri: NonEmptyStr
    content: NonEmptyStr


class ResourceDocument(StrictModel):
    """已加载好内容的资源文档，是生成 Document Card 时传给 LLM 的输入。"""
    doc_id: NonEmptyStr
    resource_uri: SourceUri
    title: NonEmptyStr
    content_or_structure: str = ""
    source_sections: list[SourceSection] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentCardContent(StrictModel):
    """LLM 为单篇文档提炼的语义卡片内容，不包含系统已知的文档标识字段。"""
    title: NonEmptyStr
    summary: NonEmptyStr
    candidate_topics: NonEmptyStrList


class DocumentCard(DocumentCardContent):
    """原始来源文档的结构化卡片。"""
    doc_id: NonEmptyStr
    resource_uri: ResourceUri
    markdown: str = ""


class NodeCardContent(StrictModel):
    """LLM 为 Wiki 节点正文生成的更具体语义描述。"""
    summary: NonEmptyStr


class NodeCard(NodeCardContent):
    """Wiki 节点的结构化卡片，以节点 scope 代替候选主题。"""
    doc_id: NodeId
    resource_uri: WikiUri
    title: NonEmptyStr
    scope: NonEmptyStr
    markdown: str = ""


class AggregationMemberView(StrictModel):
    """已分配成员的简略视图，不含 summary。"""
    card_id: NonEmptyStr
    title: NonEmptyStr
    candidate_topics: NonEmptyStrList | None = None
    scope: NonEmptyStr | None = None


class AggregationCardView(AggregationMemberView):
    """待分配 card 的完整语义视图。"""
    summary: NonEmptyStr


class AggregationNodeView(StrictModel):
    """聚合 agent 当前可编辑的目录节点及其简略成员 card。"""
    node_id: NodeId
    title: NonEmptyStr
    scope: NonEmptyStr
    cards: list[AggregationMemberView] = Field(min_length=2)


class CreateNodeToolArgs(StrictModel):
    node_id: NodeId
    title: NonEmptyStr
    scope: NonEmptyStr
    card_ids: AtLeastTwoStrList


class AddCardsToolArgs(StrictModel):
    node_id: NodeId
    card_ids: NonEmptyStrList


class RemoveCardsToolArgs(StrictModel):
    node_id: NodeId
    card_ids: NonEmptyStrList


class MergeNodesToolArgs(StrictModel):
    target_node_id: NodeId
    source_node_ids: NonEmptyStrList


class SplitNodeGroup(StrictModel):
    node_id: NodeId
    title: NonEmptyStr
    scope: NonEmptyStr
    card_ids: NonEmptyStrList


class SplitNodeToolArgs(StrictModel):
    node_id: NodeId
    groups: list[SplitNodeGroup] = Field(min_length=2)


class RenameNodeToolArgs(StrictModel):
    node_id: NodeId
    title: NonEmptyStr


class UpdateNodeScopeToolArgs(StrictModel):
    node_id: NodeId
    scope: NonEmptyStr


class ReadSummaryToolArgs(StrictModel):
    card_ids: NonEmptyStrList


class FinishToolArgs(StrictModel):
    pass


class WikiNode(StrictModel):
    """Wiki 目录图中的内部节点，保存稳定标识、主题边界和层级关系。"""
    node_id: NodeId
    title: NonEmptyStr
    depth: int = Field(ge=1)
    scope: NonEmptyStr
    parent_node_ids: list[str] = Field(default_factory=list)
    child_node_ids: list[str] = Field(default_factory=list)
    aggregation_order: int = Field(default=0, ge=0)
    primary_parent_id: str = ""


class SourceRef(StrictModel):
    """节点写正文时可使用的来源，可能是原始文档，也可能是子 Wiki 节点。"""
    ref_id: NonEmptyStr
    ref_type: Literal["document", "wiki_node"] = "document"
    doc_id: NonEmptyStr
    resource_uri: SourceUri
    card_uri: NonEmptyStr
    title: NonEmptyStr
    support_scope: NonEmptyStr
    matched_topics: list[str] = Field(default_factory=list)


class SourceAssignmentResult(StrictModel):
    """来源分配阶段的完整结果，按节点组织可引用来源并记录未分配来源。"""
    source_refs_by_node: dict[str, list[SourceRef]]
    unassigned_source_ids: list[str] = Field(default_factory=list)


class SourceAssignmentItem(StrictModel):
    """模型返回的一条来源分配，把一个 Wiki 节点绑定到一组下层来源 ID。"""
    node_id: NodeId
    source_ids: NonEmptyStrList
    support_scope: NonEmptyStr


class SourceAssignmentResponse(StrictModel):
    """来源分配步骤的结构化响应，保留模型返回的节点到来源 ID 的绑定。"""
    assignments: list[SourceAssignmentItem]
    unassigned_source_ids: list[str] = Field(default_factory=list)


class NodeDocumentContent(StrictModel):
    """LLM 生成的节点正文内容，不包含代码侧确定的文档 ID。"""
    title: NonEmptyStr = "High-Level Knowledge"
    content: NonEmptyStr


class NodeDocument(NodeDocumentContent):
    """节点目录下生成的 Markdown 正文，最终会直接写成 0001.md 等文件。"""
    document_id: NonEmptyStr


class NodeDocumentsResponse(StrictModel):
    """节点正文生成步骤的结构化响应。"""
    documents: list[NodeDocumentContent]


class GeneratedNodeContext(StrictModel):
    """单个节点生成完成后的内存上下文，汇总节点 card、正文文档和来源。"""
    node: WikiNode
    card: NodeCard
    documents: list[NodeDocument]
    source_refs: list[SourceRef]


class PipelineArtifacts(StrictModel):
    """Wiki pipeline 一次运行的内存产物集合，用于串联各阶段输出。"""
    cards: list[Union[DocumentCard, NodeCard]] = Field(default_factory=list)
    nodes: list[WikiNode] = Field(default_factory=list)
    source_refs_by_node: dict[str, list[SourceRef]] = Field(default_factory=dict)
    node_contexts: list[GeneratedNodeContext] = Field(default_factory=list)
