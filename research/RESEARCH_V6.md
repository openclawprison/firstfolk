# Research integrations v6: evidence instead of a human percentage

The requested 90%-human target has not been achieved or measured. The earlier 10/100 theoretical estimate was subjective and has no validated denominator; it should not be used as a progress metric. We do not replace it with a higher invented score.

This version integrates real upstream implementations and adds a task-specific individual-choice evaluation path. All 74 tests passed, `pip check` passed, and the running localhost API passed recall, participant prediction, training and environment checks. Source hashes and results are saved in `verification_v6.json`.

## Existing work researched and used

| Existing project | Result in this backend |
|---|---|
| [pyactr](https://github.com/jakdot/pyactr) | Installed 0.3.2 and executed its real declarative-memory retrieval engine; persistent rehearsal timestamps and optional normal retrieval policy |
| [MiniGrid](https://github.com/Farama-Foundation/Minigrid) | Installed 3.1.0 and executed actual partial-observation environments with keys, doors, orientation and reward |
| [Soar](https://github.com/SoarGroup/Soar) | Reviewed architecture/build documentation; not installed or integrated. A separate native architecture requires its own agent knowledge and validation |
| [Letta](https://github.com/letta-ai/letta) | Reviewed current stateful-agent project and move to letta-code; not installed or integrated |

The optional dependency versions are pinned in `requirements-cognition.txt`. Source hashes and upstream licenses are retained in `research/third_party/upstream_packages/`. The source archive does not bundle dependency binaries. Our adapters and MiniGrid policies are original code; the retrieval engine and task environments execute the installed upstream packages. Reviewing Soar/Letta is not counted as integrating them.

## ACT-R memory

The pyactr integration constructs actual declarative chunks from owned memory records. Its base-level activation, threshold and latency calculations select records; repeated retrieval adds bounded rehearsal traces. The benchmark shows increased modeled latency with delay and decreased latency after rehearsal. These are expected qualitative effects of this model; no human response-time dataset was fitted.

Enable with `POST /agents/{id}/cognition`:

```json
{"operation":"memory_mode","mode":"actr"}
```

The normal memory API then selects up to 64 candidates using the existing lexical relevance filter before ACT-R retrieval. Simulation recall also uses ACT-R when enabled. Memory mode remains opt-in; existing agents default to heuristic retrieval. Cognitive ablations remain enforced. This is not a complete ACT-R production, visual or motor model.

For explicit retrieval, use `actr_recall` with optional `action`, `limit`, `delay`, `threshold` and `rehearse`. Delay advances a simulated memory clock, not wall time or a fitted biological clock. Activation and latency are model outputs, not measured neural activity.

## Real perception-action tasks

The object-aware controller receives only MiniGrid's partial symbolic image, compass direction and mission. It builds spatial memory through relative odometry and uses an engineered planner to pick up keys, open compatible doors and reach visible goals. The controller never receives the environment object, true agent coordinates or hidden grid.

It solved 20 seeds of DoorKey 6×6 and 20 seeds of DoorKey 8×8 with a 200-step budget; the matched random baseline solved none. The controller's movement and interaction rules are engineered. This result is not learned general embodiment or human perception. Symbolic object IDs are already supplied by the environment; no visual recognition from raw pixels is demonstrated.

Separately, a tabular Q learner uses hashes of partial observation images and direction. Three 600-episode random-start 6×6 runs improved from 60% success before training to 100% on 20 further evaluation seeds with weights frozen. The room is small and its layout is shared across training/evaluation. Starts and observations can repeat across seeds. This is learned navigation in a fixed task family, not novel-layout generalization or a human-level result. Step penalties are training reward modifications; evaluations use native rewards.

Operations:

- `embodied_evaluate`: engineered planner versus random baseline in supported Empty/DoorKey tasks.
- `embodied_train`: actual tabular learning in supported random-start empty rooms.
- `embodied_policy_evaluate`: frozen evaluation with overlap checks on training/evaluation seeds.

## A defensible individual-choice percentage

`participant_train` learns from supplied `context`/`action` records with an explicit dataset source. `participant_predict` predicts an action for a supplied context. `participant_evaluate` measures held-out exact action agreement, a majority-action baseline, abstention coverage and a Wilson interval. Normalized training overlap and duplicate held-out contexts are rejected. Untrained human actions remain in the evaluation and count as errors rather than being silently excluded.

The reasoning tool planner can call a learned participant predictor. Its output is marked as a prediction, never a verified fact about a real person. Participant source labels are supplied metadata and remain unverified. This does not fit language-model weights or independently reproduce a person's identity.

Our demonstration uses six training records and four held-out synthetic records. It achieved 100% agreement versus a 50% majority baseline, but its nominal 95% Wilson interval is approximately 51.0%–100.0%. Four synthetic examples cannot establish ≥90% fidelity. The interval assumes independent trials; correlated choices from one participant weaken that assumption. No real participant data was used.

Example training body:

```json
{"operation":"participant_train","participant":"example_label","source":"supplied_dataset","examples":[{"context":"quiet reading alone","action":"rest"},{"context":"party with friends","action":"socialize"}]}
```

Use distinct held-out records in `participant_evaluate`. The returned `human_equivalence_percent` is always null; task agreement is not whole-person similarity.

## What the theory supports—and what remains missing

[The Common/Standard Model of the Mind](https://doi.org/10.1609/aimag.v38i4.2744) provides an architectural research framework, not a formula converting installed modules into a human percentage. [ACT-R](https://act-r.psy.cmu.edu/) supports task-specific cognitive models. [MiniGrid's research paper](https://arxiv.org/abs/2306.13831) supports reproducible goal-directed tests. Neither supports claiming that this backend is 90% human.

The 52 named trait controls and synthetic brain graph remain unchanged from v5. Most personality/affect coefficients remain hand-designed and unvalidated. Perception here is symbolic and limited; real vision remains unverified after the earlier local timeout. There is no continuous physical embodiment, broad lifelong knowledge learning, biological genotype-to-person mapping, whole-brain connectome simulation, human developmental trajectory, or consciousness evidence.

Reaching a justified task-specific 90% target requires a defined task population, real held-out examples, baseline comparisons, repeatability and enough independent trials. Reaching a 90%-human claim requires a valid whole-human measure that this project does not have. More packages or trait names cannot supply that evidence.

## Reproduce

From the backend directory:

```powershell
.venv/Scripts/python.exe -m pip install -r requirements-cognition.txt
.venv/Scripts/python.exe -m unittest discover -s tests -v
.venv/Scripts/python.exe benchmark_integrations.py
.\start.ps1
```

The integration tests run real packages when available; a minimal install skips those optional tests. The recorded 74-test run includes the optional providers. Core simulation remains usable without them. Historical v3–v5 reports retain their original scope and results.

## Live language-model check

The final local participant-tool check returned status `model_did_not_select_expected_tool`. Its exact result, timing and scope are recorded in `model_v6_verification.json`. This single synthetic probe is separate from the deterministic integration tests and is not human behavioral validation.
