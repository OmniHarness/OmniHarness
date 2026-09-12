# Benchmark Evaluation

The repository includes task adapters and evaluators for the six benchmarks
used in the paper. Dataset annotations belong under `bench data/`; generated
media belong under `results/`. Both directories are excluded from Git.

| Benchmark | Data directory | Task adapter | Evaluator |
|---|---|---|---|
| [ComfyBench](https://github.com/xxyQwQ/ComfyBench) | `bench data/ComfyBench/` | `scripts/run_comfybench.py` | `scripts/evaluate_comfybench.py` |
| [GenEval](https://github.com/djghosh13/geneval) | `bench data/GenEval/` | `scripts/run_geneval.py` | `scripts/evaluate_geneval.py` |
| [GenEval2](https://github.com/facebookresearch/GenEval2) | `bench data/GenEval2/` | `scripts/run_geneval2.py` | `scripts/evaluate_geneval2.py` |
| [KRIS-Bench](https://github.com/mercurystraw/Kris_Bench) | `bench data/KRIS_Bench/` | `scripts/run_kris_bench.py` | `scripts/evaluate_kris_bench.py` |
| [Reason-Edit](https://github.com/TencentARC/SmartEdit) | `bench data/ReasonEdit/` | `scripts/run_reasonedit.py` | `scripts/evaluate_reasonedit.py` |
| [WISE](https://github.com/PKU-YuanGroup/WISE/tree/main/WISE_legacy) | `bench data/WISE/` | `scripts/run_wise.py --variant original` | `scripts/evaluate_wise.py --variant original` |

The benchmark runner passes task instructions and source media to OmniHarness.
Reference workflows, evaluation labels, and masks remain outside the agent's
working directory. Reason-Edit masks are used only by its evaluator.

## Evaluation Environments

`requirements.txt` retains PyTorch 1.12.1, torchvision 0.13.1, and the CUDA 11.3
MMCV stack used by GenEval. Transformers 4.30.2 supplies the legacy CLIP APIs;
its [declared PyTorch dependency](https://github.com/huggingface/transformers/blob/v4.30.2/setup.py)
includes PyTorch 1.12.1.

GenEval2's Qwen3-VL judge requires Transformers 4.57.0, whose
[declared PyTorch dependency](https://github.com/huggingface/transformers/blob/v4.57.0/setup.py)
is PyTorch 2.2 or later. Use a separate CPython 3.10 environment and install
only its dependencies. Do not install the legacy `requirements.txt` into it.
The following Linux and Windows wheel pair is listed in the
[PyTorch installation archive](https://pytorch.org/get-started/previous-versions/)
and requires a driver compatible with CUDA 12.4:

```bash
python -m venv ../omniharness-geneval2-env
source ../omniharness-geneval2-env/bin/activate
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install transformers==4.57.0 accelerate==0.33.0 numpy==1.26.4 Pillow==11.3.0 requests==2.32.5 PyYAML==6.0.2
python scripts/evaluate_geneval2.py --method soft_tifa_am
```

On Windows PowerShell, activate with
`..\omniharness-geneval2-env\Scripts\Activate.ps1`.
For other devices, select a matching PyTorch and torchvision pair from the
installation archive. Generation and the other evaluators keep their original environment.

## Policy Library State

- **Self-directed inquiry** starts from empty libraries at a new `--symbolic-policy` path.
  The default configuration uses 50 iterations, ten candidates per iteration,
  and consolidation every five iterations. The full capability space covers six T2I and six I2I
  categories. Inquiry for GenEval, GenEval2, and WISE uses only a general
  T2I capability space with `inquiry --modality T2I`, without source images
  or benchmark evaluation tasks. For I2I inquiry,
  source images may be used, while downstream instructions, target outputs,
  reference workflows, benchmark annotations, and evaluation labels are withheld.
- **Downstream online execution** instantiates, adapts, and composes policies to each task and updates
  the library from execution feedback. Use an independent working library for
  each comparison. The supplied `resources/symbolic_policy` is a reference
  library of 43 workflow records and historical failure records. It includes
  downstream task traces and does not constitute an isolated inquiry snapshot.
  Controlled evaluations require inquiry from a new empty library and a
  snapshot exported before downstream execution.
- **Frozen execution** uses `--memory-mode frozen`. Task-specific adaptation,
  composition, verification, and recovery remain active; persistent library
  updates are disabled. Export the inquiry state before online downstream updates
  to obtain the frozen policy snapshot \(\mathcal{K}_{\mathrm{inquiry}}\).
- **Retry budget** defaults to four retries after the initial attempt.
  `--max-retries 4` therefore permits at most five attempts.

Each benchmark invocation executes its selected tasks in a new session. Task
order affects online evolution. Use `--start`, `--limit`, or sharding arguments
to choose a subset; independently evolving shards do not share the same
trajectory as a sequential run.

## Evaluation Outputs

Use `--list-only` on a task adapter to inspect available tasks and
`--validate-only` on an evaluator to check result coverage without model calls.
Each evaluator writes per-sample records and a summary to its `--output-dir`.
An existing evaluation record requires a new output directory or explicit
`--overwrite` to recompute scores for the current inputs and configuration.

The ComfyBench media evaluator reports **output coverage** and **Resolve**.
An image or video file by itself is not evidence of executable workflow
construction, so this evaluator does not infer **Pass** from file presence.
Pass requires workflow execution evidence. The other evaluators implement
their respective detection, visual-question-answering, image-preservation,
or judge-rubric metrics, as specified in their source files.
GenEval2 defaults to Soft-TIFA_AM. Its Overall score is the arithmetic mean
of the Object, Attribute, Count, Position, and Verb scores, reported as a
percentage in `overall_score_percent`. All five skills are required.

Numbers shown on the project page are paper-reported results. Media coverage
and local software checks do not reproduce those scores. The repository's
provided media and policy metadata are retained as research artifacts.

Benchmark data, evaluator models, and node packages retain their upstream
terms. See [Third-party assets](THIRD_PARTY_ASSETS.md) for resource provenance.
