# Adaptive agent backend v3

This version adds measurable reuse of learned skills, incremental trainable weights, behavior profiles, cultural teaching, and explicit perspective tracking. It does not clone a whole human. No real participant data has been supplied or used.

## What changed

### Reusable movement knowledge

The agent learns action-to-displacement statistics from observed transitions. It reuses those statistics on unseen layouts, including larger mazes. During navigation it sees only its own position, the declared goal coordinate and five local cells; the solver is not handed the full maze.

An engineered frontier planner operates on the map the agent builds through local sensing. This planner is not learned, and the result should not be described as end-to-end neural navigation. Frozen evaluation disables updates to the learned transition model. New-layout hashes are checked against recorded training hashes. Optional online adaptation invalidates controls when observations contradict the old movement model.

### Lifetime model updates

Each agent can train small linear softmax classifiers using supplied text/label examples. Signed hashed words and bigrams form 128 features; supervised SGD changes persistent classifier weights. A bounded demonstration replay bank supports subsequent one-example feedback without throwing away all older classes.

This is actual lifetime parameter learning in a small classifier. It is not fine-tuning of the language model, semantic understanding, or a learned biological brain. Scores are uncalibrated and expose an abstention flag. A specially named `action_preferences` classifier can bias autonomous simulation decisions; the decision trace records its input context and output scores.

### Profiles grounded in supplied records

`profile` stores biography, values, preferences and writing examples with a source and evidence-memory ID. The language model receives a bounded version of this profile and is instructed to preserve supplied preferences without claiming to be the real person or inventing missing facts.

To evaluate a particular person's behavioral likeness, supply example situations and their actual responses, reserve genuinely unseen responses, and measure prediction agreement. The existing classifier evaluator rejects exact training-text overlap. Exact overlap checks do not detect semantic duplicates or guarantee a rigorous study. A profile or high score on a tiny fixture does not establish a human clone.

### Teaching and perspectives

`teach-skill` trains a student from a teacher's retained demonstrations. It copies neither the teacher's biography nor DNA. This is a simple implementation of non-genetic knowledge transmission.

Perspective tracking distinguishes the declared actual location of an object from what an actor last observed. It can therefore answer a simple false-belief scenario and return unknown when the actor has no evidence. This is symbolic bookkeeping over explicit event visibility, not a learned theory-of-mind network or evidence of consciousness.

## New API

All operations use the existing local API at `http://127.0.0.1:8765`.

`POST /agents/{id}/profile`:

```json
{
  "label": "Fictional Mira",
  "biography": ["I study engineering."],
  "values": ["fairness"],
  "preferences": ["quiet reading"],
  "style_examples": ["I would rather read somewhere peaceful."],
  "source": "synthetic_fixture"
}
```

`POST /agents/{id}/cognition` accepts one operation plus its arguments:

| Operation | Arguments |
|---|---|
| `transfer_train` | `grids`: array of layouts, optional `steps_per_grid`, `seed` |
| `navigate` | `grid`, optional `max_steps`, `seed`, `adapt`, `control_order` |
| `transfer_evaluate` | optional `seeds`, odd `size` (5–21), `max_steps` |
| `skill_train` | `name`, `examples` containing text/label, optional `epochs`, `seed` |
| `skill_predict` | `name`, `text` |
| `skill_evaluate` | `name`, held-out `examples` |
| `perspective_event` | `object_name`, `location`, `observers` |
| `perspective_query` | `actor`, `object_name` |

Example learned text skill:

```json
{
  "operation": "skill_train",
  "name": "preferences",
  "examples": [
    {"text": "quiet reading at home", "label": "quiet"},
    {"text": "social party with friends", "label": "social"}
  ],
  "epochs": 60
}
```

Continued `skill_train` calls may contain a single feedback example using an existing class. Predictions never establish what a real person will do with certainty.

`POST /teach-skill` accepts `teacher_id`, `student_id`, `name`, optional `epochs`, `seed`.

The model reasoning tool registry now also accepts `skill` predictions and `belief` queries. Their outputs are marked as computed predictions rather than verified facts. Arithmetic and route computations retain their distinct provenance.

## Evaluation and reproduction

```powershell
.venv/Scripts/python.exe -m unittest discover -s tests -v
.venv/Scripts/python.exe benchmark_adaptation.py
```

The saved [adaptation_benchmark.json](adaptation_benchmark.json) records every held-out layout and baseline result. The run trained on two 7×7 mazes, then solved 20 unseen 11×11 mazes and 20 unseen 15×15 mazes using local observations and a frozen movement model. The random baseline solved zero of the smaller layouts and one of the larger layouts. The agent also succeeded after a control permutation with online adaptation enabled.

The text classifier and taught student both classified all four held-out synthetic preference examples correctly. Perspective bookkeeping answered 20 generated visibility scenarios correctly. These small fixtures validate mechanisms, not general human behavior. No human fidelity score or human-level capability percentage is reported.

## Research basis

- [CoALA](https://arxiv.org/abs/2309.02427) motivates separating memory, learning and decision mechanisms from the language component.
- [Successor Features for Transfer in Reinforcement Learning](https://papers.nips.cc/paper_files/paper/2017/hash/350db081a661525235354dd3e19b8c05-Abstract.html) motivates separating reusable dynamics from task-specific goals. This backend does not implement successor features or claim their theoretical guarantees; its transition-counting model is simpler.
- [LLM Agents Grounded in Self-Reports Enable General-Purpose Simulation of Individuals](https://arxiv.org/abs/2411.10109) motivates grounding individual simulations in supplied records and evaluating against held-out responses. No dataset from that study is imported, and its findings are not results of this backend.
- [Wimmer and Perner](https://doi.org/10.1016/0010-0277(83)90004-5) motivates distinguishing an actor's belief from the world. Our explicit bookkeeping does not reproduce human developmental performance.
- [Ollama vision documentation](https://docs.ollama.com/capabilities/vision) supplies the native local image API format. The provider adapter now supports `/api/chat` as well as compatible chat-completions endpoints. Native vision verification is recorded separately when completed.

## What remains missing

There is still no biological genome-to-person mapping, whole-brain simulation, human body, continuous real-world sensory learning, voice imitation, general language-model lifetime training, or validated replication of a real individual. Atlas metadata and the functional brain diagram retain their v2 limitations. The next empirical work requires richer tasks, stronger baselines, semantic representations, controlled forgetting/adaptation tests and genuinely held-out human examples.

## Verified setup status

All 38 automated tests passed. The running v3 API passed profile, training, held-out navigation, classifier, perspective and teaching checks. The native Ollama image request timed out after 180 seconds on this CPU setup; image interpretation remains unverified. See `verification_v3.json` and `vision_v3_verification.json`.
