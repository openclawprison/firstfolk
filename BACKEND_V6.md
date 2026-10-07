# Adaptive cognitive agent backend v6

A runnable backend for persistent, distinct, human-inspired agents. Python 3.11+; local JSON API and SQLite persistence.

## Start

From this directory in PowerShell:

```powershell
.\start.ps1
```

The API runs at http://127.0.0.1:8765. The script uses the workspace venv and the existing local `gemma3:4b` model if available. It does not download a model. Core simulation and verified tools also run with `python server.py` without external packages.

Current upstream integrations and evidence: [research/RESEARCH_V6.md](research/RESEARCH_V6.md). All 74 tests passed with optional providers installed; see [verification_v6.json](research/verification_v6.json). V5 traits and behavior remain documented in [DYNAMICS_V5.md](research/DYNAMICS_V5.md).

## Implemented

- Three-million-locus diploid genomes, crossover, mutation and lineage.
- Thirty-six configurable/developing dispositions, twelve affect proxies, intentions, relationship history, creative artifacts and thirteen actions.
- Sixteen inherited traits plus a developing 16-module functional graph with 24 synthetic routes and eight inherited control parameters.
- Optional actual pyactr retrieval, actual MiniGrid perception-action tasks and persistent partial-observation Q-learning.
- Supplied individual-choice predictors and held-out evaluations with explicit scope, baseline and uncertainty.
- Episodic memory, semantic evidence with conflicting claims, procedural policies and capacity-limited working memory.
- Goals, learned outcome models, bounded lookahead, appraisal proxies, homeostasis, habits and social expectations.
- Rest/sleep consolidation, skill practice, teaching and internal agent messages.
- Verified arithmetic and A* planning; persistent Q-learning in native grids and actual Gymnasium FrozenLake.
- Local language-model reasoning with allowlisted tool execution and memory citations; an optional image-observation interface.
- Published Schaefer cortical parcel metadata, functional map, ablation controls, research citations and benchmark reports.

Start with [research/ADAPTIVE_V3.md](research/ADAPTIVE_V3.md) for transfer, lifetime classifiers, profiles, teaching and perspective tracking. Read [research/IMPLEMENTATION.md](research/IMPLEMENTATION.md) for the complete API, research basis, brain-map interpretation, installation and limitations. View [research/brain_map.png](research/brain_map.png).

## Fresh installation and tests

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements-cognition.txt
.venv/Scripts/python.exe -m unittest discover -s tests -v
.venv/Scripts/python.exe benchmark.py
```

Gymnasium and plotting dependencies are optional. Earlier v3 results are in [research/verification_v3.json](research/verification_v3.json). Current v6 results and 74-test validation are linked above. Earlier local model arithmetic tool use was verified. The native Ollama vision test exceeded its 180-second timeout, so real image interpretation remains unverified on this setup. Run `benchmark_adaptation.py` for the v3 held-out evaluations.

## Scope

This is a research prototype, not a complete human brain or a biological genome-to-person model. Atlas labels and coordinates are real published metadata; runtime routing and activity are synthetic. Simulated skill counters are distinct from verified task results. Language-model weights do not learn during an agent's lifetime. Benchmark successes concern small specified environments, not general human capability.

The server binds only to localhost, with no authentication. Keep it local. Data persists under `data/`; it is excluded from the source archive. Configuring a hosted model sends selected agent memories and tasks to that endpoint; the tested model setup uses localhost.
