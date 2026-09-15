# Batched Tool-Calling Aggregation

## Purpose

Construct the Wiki from the bottom up. Each depth processes cards in sequential
batches, sharing the directory nodes built so far. Each batch adds 25 new cards
by default, plus any unassigned cards carried from preceding batches. Original
Document Cards enter the first depth; only newly generated Node Cards enter
the next depth. Unassigned cards are recorded but do not move to the next depth.

## Layer Lifecycle

```text
current-layer cards
  -> introduce up to 25 new cards, retaining previous unassigned cards
  -> fresh batch context and empty read-summary cache
  -> agent turn: compact directories + full unassigned cards + read summaries
  -> execute ordered tools and update affected card memberships
  -> repeat until finish, fewer than 5 pending with later batches, or 40 turns
  -> next batch, if any new cards remain
  -> post-layer sliding-window split for oversized nodes
  -> materialize source refs, Markdown, and Node Cards
  -> record unassigned cards, including when no directories were created
  -> stop if no directories exist; otherwise aggregate the new Node Cards
```

The batching loop follows the original card order. Leftovers do not consume the
25-new-card allowance, so a batch can contain more than 25 unassigned cards.
There are no extra batches containing only leftovers after the last new cards.

## Agent Context

Each turn renders a fresh state snapshot without previous conversational turns:

```json
{
  "has_more_batches": true,
  "existing_nodes": [
    {
      "node_id": "sparse_retrieval",
      "title": "Sparse Retrieval Models",
      "scope": "Methods and evaluation of sparse retrieval models.",
      "cards": [
        {
          "card_id": "paper_a",
          "title": "Paper A",
          "candidate_topics": ["Sparse retrieval"]
        }
      ]
    }
  ],
  "unassigned_cards": [
    {
      "card_id": "paper_c",
      "title": "Paper C",
      "summary": "Complete card summary.",
      "candidate_topics": ["Retrieval evaluation"]
    }
  ],
  "read_summaries": [
    {"card_id": "paper_a", "summary": "Previously read summary."}
  ]
}
```

The example abbreviates directory membership; actual directories require at
least two distinct cards. Member cards show ID, title, and candidate topics
for documents or scope for Node Cards. Their summaries are hidden. Pending
cards show the same fields plus summary. No batch number enters the prompt.

`read_summary` accepts one or more introduced card IDs that currently belong
to at least one directory. It cannot read pending cards, future cards, or the
new directories themselves: their Node Cards do not exist until materialization.
All IDs are validated before any summary is cached. Repeated reads are deduplicated.
The cache persists for the current batch and is cleared at the next batch.

When a card becomes unassigned, its full view returns to the pending list and
its read-cache entry is deleted immediately, even if another tool reassigns it
later in the same response. Assignment removes the full pending view; an existing
read-cache entry otherwise stays until the batch ends.

Only the latest turn's tool errors or zero-tool-call feedback are appended to
the next prompt. There is no consecutive-no-progress counter or failure rule.
The prompt explicitly says more new cards will arrive when appropriate; the
last batch instead says all cards have arrived and requests `finish` even when
no cards remain pending.

## Ordered Tools and Incremental Membership

Calls execute in response order. Invalid calls leave state unchanged and
produce feedback for the next turn.

| Tool | Required fields | Effect |
| --- | --- | --- |
| `create_node` | `node_id`, `title`, `scope`, `card_ids` | Create a coherent directory from at least two introduced cards. |
| `add_cards` | `node_id`, `card_ids` | Add cards, allowing membership in multiple directories. |
| `remove_cards` | `node_id`, `card_ids` | Remove members while retaining at least two cards. |
| `merge_nodes` | `target_node_id`, `source_node_ids` | Union members into the target and remove source directories. |
| `split_node` | `node_id`, `groups` | Replace a directory with at least two groups; preserve every original member. Groups may overlap. |
| `rename_node` | `node_id`, `title` | Rename a directory. |
| `update_node_scope` | `node_id`, `scope` | Update a directory's knowledge boundary. |
| `read_summary` | `card_ids` | Read hidden summaries of directory member cards. |
| `finish` | none | End this batch; calls later in the same response are ignored. |

Each introduced card has a membership count. Editing tools return count deltas
only for cards affected by the edit. Create/add increment newly established
memberships; remove decrements removed memberships. Merge/split compute net
deltas across the affected directories, preserving DAG overlap. Rename/scope
edits return no deltas. State updates do not scan unrelated cards or directories.

A count reaching zero immediately restores the card to the current pending
list. A positive count removes it from that list. This maintains the invariant:

```text
pending = introduced cards - cards belonging to at least one directory
```

Rendering the complete directory list still visits its members each turn;
incremental membership tracking avoids a separate full-layer assignment scan.
Batching reduces summary volume but does not place a fixed bound on context:
leftovers, directory membership, and explicitly read summaries can still grow.

## Batch Completion

After executing the ordered calls for one model response, end the batch when:

1. A valid `finish` was executed; later calls in that response are ignored.
2. Otherwise, the per-batch decision-round limit (default 40) was reached.
3. Otherwise, fewer than 5 cards remain pending and more new cards exist.

At the round limit, retain valid edits, record the end reason and a warning,
and continue with the next batch. At the last batch, retain remaining cards
as the final unassigned result and finish the layer. Fewer than 5 pending cards
never automatically ends the last batch, even when the count is zero.
A response without tool calls does not count as `finish`; it consumes a round
and produces feedback. Valid calls in truncated responses still execute.

## Materialization and Stopping

After all batches, oversized directories are split with sliding windows using
`max_cards_per_node`. For maximum size M, the window advances by M - 1, sharing
one boundary card with its neighbor. Split IDs contain depth and part number.
This bounds source content per node body, not aggregation context size.

The program then constructs SourceRefs, generates node Markdown, and creates
Node Cards. Model output supplies only Node Card summary; title and scope come
from the directory. Initial Document Cards instead generate title, summary,
and candidate topics from the document input, without copying a garbled filename.

Only the new Node Cards enter the next depth. Every layer's unassigned IDs are
collected in the final `source_assignments.json`, including the empty layer
that ends upward aggregation. If no directories exist at the end of a layer,
the complete aggregation stops.

## Observability and Configuration

Every turn is persisted through the existing callback to
`viking://wiki/run/aggregation_operations.jsonl`. Records include depth,
batch index, turn within batch, whether more batches exist, executed calls
and read results, errors, before/after state summaries, and batch end reason
(`finish`, `few_unassigned`, or `max_turns`). The response payload includes
text, all returned calls, finish reason, and usage. Batch indices are for logs only.

```python
WikiGenerationLimits(
    aggregation_batch_size=25,
    aggregation_agent_max_turns=40,
    max_cards_per_node=1000,
    max_concurrent_cards=10,
    max_concurrent_nodes=10,
    llm_request_timeout_seconds=300.0,
)
```

Concurrency settings apply to independent card and node-content generation.
Aggregation batches and their tools execute sequentially.
