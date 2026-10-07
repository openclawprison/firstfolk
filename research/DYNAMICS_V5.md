# Human-inspired dynamics v5

This version completes the previously interrupted expansion. It passes 60 automated tests, including controlled interventions for every one of the 24 new facets. The running localhost API passed migration, configuration, appraisal, inventory and 10-tick persistence checks. Exact paths, counts and source hashes are in `verification_v5.json`.

## Counts

| Category | v4 | v5 |
|---|---:|---:|
| Fixed inherited trait controls | 16 | 16 |
| Configurable/developing dispositions | 12 | 36 |
| Total named personality controls | 28 | 52 |
| Affect variables | 4 | 12 |
| Appraisal dimensions | 4 | 7 |
| Simulation actions | 9 | 13 |
| Psychological needs | 5 | 5 |
| Body drives | 3 | 3 |
| Functional modules / synthetic routes | 16 / 24 | 16 / 24 |
| Fresh-agent stored numeric state values | 115 | 183 |

Numeric state counts include duplicate baseline values, counters and caches. They grow with experience and learned models. The 52 controls are not 52 independent or validated human traits. Defaults for the 36 dispositions derive from the original inherited controls; they become separately configurable but are not additional biological genes. Four dispositions currently drift slowly with experience; the others remain configurable tendencies.

## Added facets and their direct score effects

| Facet | Target action | Context signal |
|---|---|---|
| learning_progress_drive | practice | progress |
| creativity | create | autonomy |
| diligence | practice | competence |
| organization | plan | uncertainty |
| perfectionism | practice | error |
| patience | plan | stress |
| frustration_tolerance | practice | frustration |
| forgiveness | repair | resentment |
| gratitude | help | gratitude |
| fairness | help | unfairness |
| generosity | help | resources |
| modesty | practice | competence |
| social_confidence | socialize | belonging |
| rejection_sensitivity | socialize | rejection |
| attachment_anxiety | socialize | loneliness |
| attachment_avoidance | socialize | belonging |
| sensory_sensitivity | rest | arousal |
| loss_aversion | explore | loss |
| cognitive_flexibility | explore | repetition |
| ambiguity_aversion | explore | uncertainty |
| self_discipline | practice | fatigue |
| playfulness | play | boredom |
| competitiveness | practice | competence |
| deliberation | plan | uncertainty |

Each facet has an inspectable contribution in the decision trace. Loss aversion, ambiguity aversion, rejection sensitivity and attachment avoidance reduce their target score; the other table entries increase it. These are chosen engineering relationships, not empirical effect sizes. Some constructs overlap intentionally; the inventory does not imply statistical independence.

## New mechanisms

- Seven appraisal inputs: novelty, goal congruence, controllability, social acceptance, agency, fairness and loss. Inputs can be supplied through the API or derived from toy simulation outcomes. They are not automatically inferred from real people.
- Eight additional affect proxies: joy, sadness, anger, shame, pride, gratitude, boredom and loneliness. Different responsibility and fairness inputs produce different reactions. These variables change action scores or partner preferences; they do not establish felt emotion.
- Partner-specific closeness, resentment, gratitude and interaction counts influence partner choice. Forgiveness changes resentment decay. Relationship repair becomes available only for known strained relationships and delivers an internal memory to the recipient after all agents decide.
- Prediction-error reduction produces an action-specific learning-progress signal. This biases future choices but does not train the language model or prove general knowledge acquisition.
- Planning forms a bounded short-term intention; organization and flexibility affect commitment. Existing learned outcome models supply the candidates.
- Mood-congruent recall reweights existing memories while preserving original text and provenance. Fatigue and sensory load reduce workspace capacity. Fatigue also reduces simulated reward-learning rate.
- Recorded milestones form a bounded autobiographical event index. This cites simulation memory IDs, not invented biographical facts.
- New actions: create, play, plan and repair. Creating produces a saved structured design sketch through simple element recombination; it is not human-evaluated creativity. Artifacts retain unique monotonic indexes and a bounded 32-item buffer. Play reduces simulated stress and novelty need.

## Reproduction and API

Run `.venv/Scripts/python.exe -m unittest discover -s tests -v` and `.venv/Scripts/python.exe benchmark_behavior.py` from the backend folder.

`GET /inventory` lists counts and new facet mechanisms. `GET /agents/{id}/inventory` additionally lists every numeric state path. Use `POST /agents/{id}/cognition` with `personality_configure` to change named controls in [0,1].

Example supplied appraisal:

```json
{"operation":"appraise","goal_congruence":-0.8,"agency":0.1,"fairness":0.1,"loss":0.4}
```

Goal congruence permits [-1,1]; the other inputs permit [0,1]. This updates internal reactions and records `cognitive_operation` provenance. It does not assert an external event actually occurred. Invalid inputs roll back atomically.

## Evidence and limits

`behavior_benchmark_v5.json` contains all 24 facet interventions, 1,000 sampled choices for each of two equal-genome agents with contrasting configurations, and a 100-tick eight-agent community run. The intervention fixtures verify that the new controls are connected to behavior. They do not validate the psychological equations or rank the quality of the personalities.

The core remains a small stochastic world with engineered rewards and cooperation outcomes. Creature physiology, neural activity and skill counters are proxies. The atlas retains published 100-parcel metadata; runtime brain routing remains synthetic. No whole-brain/connectome simulation, biological genome-to-person mapping, consciousness evidence, broad embodied perception, general lifelong language-model training or validated replica of a real person is supplied. Vision remains unverified after the prior local timeout.

There is no measured human similarity percentage. The next meaningful evaluation requires held-out human choices or interviews, repeated-person consistency and stronger baselines. Adding controls alone cannot establish human equivalence.

## Research basis

[Self-Determination Theory](https://selfdeterminationtheory.org/topics/application-basic-psychological-needs/) motivates autonomy, competence and relatedness. Our safety/self-worth variables are additional engineering choices. [Human learning-progress research](https://pubmed.ncbi.nlm.nih.gov/34645800/) motivates curiosity signals from improvement. [Personality state research](https://pubmed.ncbi.nlm.nih.gov/11414368/) motivates context-sensitive behavior. [Appraisal-based agents](https://arxiv.org/abs/2309.05076) motivates emotional context in dialogue. None validates our coefficients or the implementation as a human clone. No upstream code from these studies was copied. Existing licensed GitHub atlas data and provenance remain included.
