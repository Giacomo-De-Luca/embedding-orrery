# Dataset acceptance tests (defined before implementation)

1. Complete dictionary, including inactive/isolated features; a reordered node
   table resolves the canonical model::sae::feature identity independently of row.
2. Reject duplicate IDs/indices/features, dangling or mismatched endpoints,
   self-links, reverse/duplicate undirected pairs, nulls and incorrect Arrow types.
3. Reject NaN/infinite scores, weights outside (0,1], unsafe/fractional counts,
   impossible joint counts, and partial/nonfinite coordinates. Accept typed empty edges.
4. Separate-token A/B fixture: one document joint event and no token joint event;
   explicitly processed empty events remain in the denominator. Dedupe event activity.
5. Exact bounded-memory counts match a small brute-force oracle, including >255
   events. Deterministic NPMI ranking uses support then feature index for ties;
   mutual top-k bounds degree and preserves every node.
6. Source audit rejects wrong SAE metadata, incomplete/unproven processing coverage,
   unknown feature/event IDs, invalid activations, incompatible linked collections,
   and unexpected source sizes (including a top-k seed).
7. Repeat exports have identical hashes/revisions; Arrow round-trips preserve types;
   checksum tampering is rejected and immutable outputs are never overwritten.
8. Full live build: audit all source documents/rows, verify bundles and distributions,
   independently recount a deterministic edge sample from source event bitsets, and
   read Python Arrow files in JavaScript/TypeScript if the Arrow runtime is available.

Implementation order: tests, reusable Python contract/counting/source/writer classes,
config-driven build, source/build documentation, full export verification, then the
repository-required code-quality-reviewer. React rendering/catalog work is outside
this dataset-only request.
