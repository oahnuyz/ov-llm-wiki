# Full-Layer Tool-Calling Aggregation

## Purpose

This design constructs a Wiki from the bottom up. At each depth, a single
tool-calling agent sees every card in the current layer and incrementally
builds only the coherent directory nodes warranted by those cards. It may
leave cards unassigned. Those cards are deliberately not carried into the next
depth: after a complete agent loop, forcing them upward would create weak or
overly broad aggregation.

The first layer receives original Document Cards. Each later layer receives
only the Node Cards created at the preceding depth. Nodes therefore become
progressively broader as the hierarchy rises.

## Layer Lifecycle

```text
current-layer cards
  -> agent turn: existing directory nodes + unassigned cards
  -> ordered function calls, executed immediately
  -> updated directory nodes + recomputed unassigned cards
  -> repeat until finish_layer
  -> post-layer sliding-window split for oversized nodes
  -> materialize nodes, source refs, Markdown, and Node Cards
  -> only new Node Cards enter the next depth

If no directory nodes are materialized at a depth, aggregation stops.
```

The agent is limited by `aggregation_agent_max_turns` (default `50`). Hitting
that limit is an error rather than silently accepting an unfinished layer.

## Agent Input

Each turn renders one prompt with this state:

```json
{
  "existing_nodes": [
    {
      "node_id": "sparse_retrieval",
      "title": "Sparse Retrieval Models",
      "scope": "Methods and evaluation of sparse retrieval models.",
      "cards": [
        {
          "card_id": "paper_a",
          "title": "...",
          "summary": "...",
          "candidate_topics": ["..."]
        }
      ]
    }
  ],
  "unassigned_cards": [
    {
      "card_id": "child_node_c",
      "title": "Child Node C",
      "summary": "...",
      "scope": "The precise knowledge boundary of Child Node C."
    }
  ]
}
```

`existing_nodes` contains full member card views so the agent can correct an
earlier grouping. `unassigned_cards` is derived after every tool turn from all
cards not currently used by any directory node. It is not persisted as an
independent state object. When the previous turn has errors, a separate
`tool_errors` feedback block is appended at the very end of the prompt, after
the rendered template. It contains the latest tool validation, zero-call, or
no-progress feedback and is replaced rather than accumulated on later turns.

The input holds semantic card data only. Original document cards carry
`candidate_topics`; generated Wiki node cards carry their authoritative
`scope` instead. It never includes source Markdown, raw chunks, or resource
storage metadata.

## Ordered Tool Operations

The provider returns standard function calls. The program validates and
executes them in their returned order, so later calls in one response observe
the effect of earlier calls.

| Tool | Required fields | Effect |
| --- | --- | --- |
| `create_node` | `node_id`, `title`, `scope`, `card_ids` | Create a directory node from at least two known cards. |
| `add_cards` | `node_id`, `card_ids` | Add known cards to a node. Existing membership elsewhere remains, allowing DAG overlap. |
| `remove_cards` | `node_id`, `card_ids` | Remove member cards while retaining at least two cards. |
| `merge_nodes` | `target_node_id`, `source_node_ids` | Union source members into the target and remove the source nodes. |
| `split_node` | `node_id`, `groups` | Replace one node with two or more groups; every original card must remain in at least one group. Groups may overlap. |
| `rename_node` | `node_id`, `title` | Rename a node. |
| `update_node_scope` | `node_id`, `scope` | Change a node's knowledge boundary. |
| `finish_layer` | none | Finish the current depth. It should be called alone after earlier edits are complete. |

`node_id` is a stable lowercase identifier of letters, digits, and underscores.
The agent may only use supplied card IDs and existing node IDs. Invalid calls
are rejected without changing state and are returned in `tool_errors` on the
next turn.

## Semantic Rules

- A directory node must contain at least two distinct cards and represent a
  specific coherent topic, not a common keyword or broad catch-all category.
- Nodes are editable for the entire layer. The agent can revise scope,
  membership, or structure after inspecting all relevant cards.
- A card may appear in multiple substantively different nodes. This preserves
  the Wiki DAG while prohibiting near-duplicate nodes.
- A card that has no credible relation to another card remains unassigned. It
  does not become a singleton node and does not pass into the next layer.
- The agent must not force a grouping merely to cover every card. It calls
  `finish_layer` once no useful aggregation or correction remains.

There are no `pending`, `provisional`, `active`, or rejection states. A node
either exists as a valid directory node or does not exist.

## Post-Layer Splitting

`max_cards_per_node` controls only post-layer materialization, not agent
reasoning. If a finalized node has more members than the configured maximum,
the program divides its ordered members with a sliding window. For maximum
size `M`, each child window contains up to `M` cards and the cursor advances by
`M - 1`, so neighboring windows share one boundary card. The generated node
IDs include both depth and part number, such as `_d1_1` and `_d2_1`, so a
repeated topic name at a higher layer cannot collide with a lower-layer split.

For `M = 4` and cards `[a, b, c, d, e, f, g]`, the materialized nodes contain:

```text
[a, b, c, d]
         [d, e, f, g]
```

The overlap preserves local context while bounding the source content used
for each node body.

## Materialization And Stopping

Only a validated and executed `finish_layer` call completes a layer. A response
without structured tool calls never completes it, even with `finish_reason=length`.
Valid calls in truncated responses still execute; errors are fed back next turn.
Three consecutive turns without an actual node state change or a valid finish
cause an explicit failure. Changes to titles, scopes, or membership reset this
counter; successful no-op calls do not. The existing 50-turn cap also remains.

For every finalized directory node, the fixed program creates `SourceRef`
entries from its validated card IDs, generates its Markdown documents, and
generates a Node Card. The next depth receives precisely those new Node Cards.
Unassigned current-layer cards are recorded in `source_assignments.json` but
are excluded from the next depth.

If the agent finishes a layer without creating any valid directory node, there
are no Node Cards to aggregate and the complete Wiki build stops. There is no
fixed depth limit and no separate model judgement about whether to continue.

## Observability

Every agent turn is appended to:

```text
viking://wiki/run/aggregation_operations.jsonl
```

Each record includes depth, turn, executed function calls, validation errors,
the node/unassigned-card state before and after execution, and whether the
layer was finished. It also includes `made_progress`, `consecutive_no_progress`,
and a `response` payload with returned text, all calls (including rejected ones),
finish reason, and usage. The existing per-turn callback persists this record
before a stall error is raised, so later build failures do not lose the response.
This is the adapter's response payload, not raw HTTP bytes. The regular Wiki
logs also retain the rendered prompt, tool schema, and request metadata on
successful completion.

## Configuration

```python
WikiGenerationLimits(
    aggregation_agent_max_turns=50,
    max_cards_per_node=4,
    max_concurrent_cards=10,
    max_concurrent_nodes=10,
    llm_request_timeout_seconds=300.0,
)
```

`max_concurrent_cards` and `max_concurrent_nodes` govern independent LLM calls
for card and node-content generation. They do not batch or limit the cards
shown to the aggregation agent.
