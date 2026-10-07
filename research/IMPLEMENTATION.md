# Research cognitive backend v2

## What this upgrade implements

The backend now combines an inherited synthetic genome, a developing functional graph, complementary memory stores, bounded action selection, actual task tools, actual reinforcement learning, and an optional language/vision-model component. The graph is an inspectable computational model. It is not a reconstructed human brain.

| Mechanism | Implementation | Research relationship |
|---|---|---|
| Episodic memory | Persistent events, salience/relevance/recency retrieval | Simplified fast-memory analogue of complementary learning systems |
| Semantic evidence | Structured subject/predicate/value records with source IDs, alternative claims and relative support | Slower accumulation inspired by CLS; relative support is not calibrated probability |
| Procedural memory | Persistent action habits and learned Q tables | Reinforcement learning; distinct from narrative memory |
| Working memory | Priority-ranked workspace, inherited capacity of 2–7 items | Capacity limitation and cognitive control; chosen range is not a fitted human model |
| Executive control | Explicit goals, progress tracking, learned-effect lookahead and action bias | PFC theory and cognitive architectures |
| Learned world model | Online mean/variance of action outcomes and mean state changes; bounded lookahead over those estimates | Independent model-based planning baseline; observational averages are not causal inference |
| Reward prediction | Expected reward, prediction error, calibration error accumulation | Schultz, Dayan and Montague; no simulated dopamine biology |
| Consolidation | Sleep/rest-related replay into knowledge and slow action-value updates | Replay and CLS research; no oscillations or synapses modeled |
| Appraisal | Reward and surprise update valence, arousal, fear and frustration proxies | Hand-designed computational states with causal effects |
| Homeostasis | Simulated energy, hunger, hydration and sleep pressure; resource-consuming actions | Engineering analogue; no physiology model |
| Social inference | Directed trust and Beta-style cooperation expectations grounded in interaction history | Narrow social prediction, not a complete theory-of-mind model |
| Development | Maturation, skill practice, teaching and bounded reward-modulated routing plasticity | Independent heuristic implementation |
| Language reasoning | Existing model consumes memories, goals, state and grounded tool results | CoALA-inspired separation of language component and agent memory |
| Verified tools | Bounded arithmetic interpreter and A* grid planning | Actual computations with inspectable results |
| Environment learning | Native grid Q-learning and actual Gymnasium FrozenLake integration | Q-learning; evaluation is environment-specific |

## Brain mapping

The atlas layer contains 100 published Schaefer cortical parcel labels, seven network assignments, RGB colors and MNI RAS centroid coordinates. It was downloaded from the authors' CBIG GitHub repository. The exact commit and SHA-256 checksums are in [third_party/provenance.json](third_party/provenance.json); the upstream MIT license is retained in [third_party/CBIG_LICENSE.md](third_party/CBIG_LICENSE.md).

The separate runtime layer contains 16 functional modules and 24 directed routes. Their anatomical names are broad analogies, not one-to-one localization claims. Brain functions are distributed; the actual brain has far greater anatomical and dynamic complexity. Atlas centroids are not used to fabricate connectivity. No diffusion MRI, fMRI activity, synaptic connectome, voxel segmentation or personal scan is included.

The genome determines initial route strengths plus eight operational parameters: workspace capacity, learning rate, discount, habit weight, replay fraction, sleep threshold, inhibition and social update rate. Sleep threshold and inhibition are used in sleep scheduling and cognitive control. Existing 16-trait expression is preserved for compatibility. Individual experience adjusts routing strengths while inherited DNA remains fixed.

`GET /brain/map` returns the atlas and generic functional map. `GET /agents/{id}/brain` adds that agent's actual synthetic routing weights and activity values. [brain_map.png](brain_map.png) shows three published centroid projections above the synthetic module graph.

Software ablations can disable selected module IDs. Working-memory ablation empties the workspace; hippocampal ablation suppresses recall and replay; executive ablation removes goal/forecast bias; striatal ablation disables policy training and habit bias. These are computational experiments, not clinical injury simulations. Other region ablations zero graph activity; they do not necessarily disable every related capability or external tool.

## New API routes

All JSON bodies below are examples. IDs come from `POST /agents`.

| Method | Route | Example body |
|---|---|---|
| GET | `/brain/map` | Published atlas + functional graph |
| GET | `/research/sources` | Research source registry |
| GET | `/agents/{id}/brain` | Individual synthetic module state |
| POST | `/agents/{id}/observe` | `{"text":"The lamp is red","claims":[{"subject":"lamp","predicate":"color","value":"red"}],"confidence":0.8}` |
| POST | `/agents/{id}/goals` | `{"metric":"resources","target":10,"priority":0.8}` |
| POST | `/agents/{id}/consolidate` | `{}` |
| POST | `/agents/{id}/lesions` | `{"regions":["executive"]}`; restore with `{"regions":[]}` |
| POST | `/agents/{id}/solve` | `{"kind":"arithmetic","expression":"41*17+3"}` |
| POST | `/agents/{id}/solve` | `{"kind":"route","grid":["S..",".#.","..G"]}` |
| POST | `/agents/{id}/train` | `{"kind":"grid","grid":["S..",".#.","..G"],"episodes":200,"seed":11}` |
| POST | `/agents/{id}/train` | `{"kind":"gym","episodes":1000,"seed":42}` |
| POST | `/agents/{id}/reason` | `{"task":"Use the arithmetic tool to calculate 41*17+3"}` |
| POST | `/agents/{id}/perceive` | `{"image_base64":"...","mime_type":"image/png","question":"What is visible?"}`; requires vision-capable configured model |
| POST | `/messages` | `{"sender_id":"...","recipient_id":"...","text":"Let us cooperate"}` |

`reason` performs model tool planning, executes at most three allowlisted computations, and requests a response grounded in their results. No shell, browser, arbitrary code or external communication tool is exposed. Plain `think` remains available. Model generations are recorded with model provenance and no invented feedback reward. A model response can still be wrong; structured output and citations do not make it infallible.

`perceive` accepts PNG/JPEG images up to 1 MB, sends them only to the configured model endpoint, and records textual observations with `vision_model` provenance. Model interpretations are explicitly unverified. Images are not automatically converted to semantic facts, and the image bytes are not stored in the agent's SQLite biography.

## Run and reproduce

```powershell
.\start.ps1
```

The start script uses the workspace venv when available and connects to the existing local `gemma3:4b` if Ollama reports it. It does not download a model. Custom model environment variables take precedence. Agent state and selected memories go to whichever endpoint you configure; the tested setup uses only localhost.

For a fresh checkout with optional research dependencies:

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements-research.txt
.venv/Scripts/python.exe -m unittest discover -s tests -v
.venv/Scripts/python.exe benchmark.py
.venv/Scripts/python.exe render_brain_map.py
```

Core simulation, tools, memory and grid learning need only Python's standard library. Gymnasium and Matplotlib are optional. The full benchmark writes [benchmark_results.json](benchmark_results.json). Dependency versions tested here are Gymnasium 1.3.0 and Matplotlib 3.11.2.

The five seeded FrozenLake trials in the saved benchmark improved from 0–3.3% success before training to 100% afterward over 30 evaluation episodes per trial, on the deterministic default map. Grid routing and arithmetic also produced verified results. This evaluates a small environment-specific learned policy, not human-level navigation or generalization to new maps. The grid baseline uses random tie-breaking and can already succeed before training; the report preserves those successes instead of hiding them.

Final verification: 26 tests passed. The actual local `gemma3:4b` model successfully selected the arithmetic tool, answered `700`, and cited the tool's stored memory; the trace is in [local_model_verification.json](local_model_verification.json). Initial requests with larger context timed out; the final pipeline limits context and response size. The live vision smoke test returned HTTP 400 on this machine. Vision input validation and provenance are unit-tested, but successful real image interpretation has not been verified. CPU-only model inference can be slow; language-model requests have a 180-second timeout and return a controlled error on provider failure.

## Research and upstream projects

- [CoALA](https://arxiv.org/abs/2309.02427): conceptual framework for memory, learning and decision cycles.
- [Complementary learning systems](https://doi.org/10.1037/0033-295X.102.3.419): fast episodic acquisition and slow integration.
- [Miller and Cohen](https://doi.org/10.1146/annurev.neuro.24.1.167): prefrontal cognitive-control theory.
- [Schultz, Dayan and Montague](https://doi.org/10.1126/science.275.5306.1593): prediction and reward signals.
- [Human hippocampal replay](https://www.nature.com/articles/s41467-018-06213-1): replay and later memory performance.
- [Schaefer et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC6095216/) and [CBIG GitHub](https://github.com/ThomasYeoLab/CBIG): actual atlas metadata used here.
- [Gymnasium GitHub](https://github.com/Farama-Foundation/Gymnasium) and [FrozenLake documentation](https://gymnasium.farama.org/environments/toy_text/frozen_lake/): external environment actually integrated and evaluated.

Soar, Nengo and Nilearn were reviewed as possible integrations. They are not installed or incorporated into this version; citing them does not imply their capabilities exist in this backend. No upstream engine source code was copied. The imported atlas files retain their upstream license; Gymnasium is installed as its own package.

## Limits and next research work

This version does not implement all human capabilities. It has no biological genome-to-person model, human-calibrated developmental model, neuron-level brain simulation, continuous sensory body, real-world motor skills, intrinsic consciousness assessment, or lifetime training of language-model weights. Affect variables are computational proxies. Social inference predicts cooperation, not complex beliefs or intentions. Q tables specialize to individual environments. Toy-world skill counters remain separate from verified task results.

Millions of loci still compress into a small number of operational traits and graph parameters. Increasing loci alone cannot establish richer individuality. Further progress needs learned perceptual representations, richer persistent worlds, transfer tests, semantic embedding retrieval, continual neural learning, and controlled human behavioral comparisons. The included tests and benchmarks establish software behavior in specified tasks, not biological or human equivalence.
