# ComfyUI Node Reference

This directory provides reference documentation for ComfyUI node semantics.
It does not contain executable custom-node packages.

- `node_catalog.json` is a compact machine-readable catalog of 3,205 node
  classes, descriptions, inputs, and outputs.
- `documentation_corpus/` contains 3,412 Markdown and CSV documents organized
  by core or custom-node package.
- The corpus preserves 102 per-package `licenses.md` records supplied by the
  upstream documentation snapshot.

The documentation accompanies the
[ComfyBench](https://github.com/xxyQwQ/ComfyBench) resource collection.

The copied catalog has SHA-256
`dc24f759fd0af1bd40112e325795da34be3d06d4fb5af2f2bd93b86fd7076af8`.
The corpus contains 3,412 files totaling 15,382,768 bytes.

This material is reference data only. OmniHarness validates executable graphs
against the live ComfyUI `/object_info` endpoint, which remains authoritative
for the node classes and schemas actually installed on a worker.

The root OmniHarness Apache-2.0 license does not relicense this corpus.
Preserve the included package notices and applicable upstream terms. See
[benchmark evaluation](../../BENCHMARKS.md) and
[resource provenance](../../THIRD_PARTY_ASSETS.md).
