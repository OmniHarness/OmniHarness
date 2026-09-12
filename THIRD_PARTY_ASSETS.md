# Third-party Assets

The repository uses the following resources.

- `resources/source_image_pool_comfybench/` contains source images and
  metadata from ComfyBench. Metadata records upstream paths, task identifiers,
  file sizes, and SHA-256 digests.
- `resources/source_image_pool_gpt/` contains independently generated source
  images for the source-pool isolation setting. Their metadata identifies them
  as generated images and records file sizes and SHA-256 digests.
- `resources/symbolic_policy/` contains author-provided reference workflows and failure
  records. Historical trace paths in these records describe their original
  experiments; they are not required local execution paths. Runtime retrieval
  and export parameterize workflow inputs before reuse.
- `resources/comfyui_node_reference/` contains node descriptions and a compact
  catalog. The included package-level `licenses.md` records are preserved.
  This corpus is reference material, not executable custom-node code. The
  live ComfyUI `/object_info` response supplies execution-time node schemas.
- `assets/` and `docs/assets/` contain paper figures and project-page media.
  These directories are not runtime source-image or policy libraries.

The root Apache-2.0 license applies to OmniHarness code. It does not relicense
third-party images, model weights, node packages, or benchmark material.
Upstream licenses and included notices continue to apply. No model weights
or executable ComfyUI custom-node packages are bundled; names in workflow
JSON files identify dependencies.

## Evaluation Provenance

The six `scripts/evaluate_*.py` files target the benchmark protocols listed in
[BENCHMARKS.md](BENCHMARKS.md) and record upstream implementation URLs in their
summaries. KRIS-Bench rubric text is retained in
`scripts/kris_official_prompts.py` with its upstream attribution. Benchmark
annotations and generated result media remain outside version control.
