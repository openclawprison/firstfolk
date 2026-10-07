# Backend v4: review, variable inventory and human similarity

The backend is running at http://127.0.0.1:8765. All 48 tests passed. The review covered runtime modules, persistence, HTTP/model adapters, learning, heredity, atlas rendering, benchmark scripts, startup script and tests. Source hashes and exact numeric state paths are in `verification_v4.json`. This is a code review with regression checks, not formal proof that every possible failure is absent.

## Exact inventory

| Category | Count | Meaning |
|---|---:|---|
| Inherited traits | 16 | Fixed engineering parameters derived from synthetic DNA |
| Developing dispositions | 12 | Configurable tendencies; four gradually adapt through experience |
| Named personality controls | 28 | The above combined; correlated, not 28 independent human dimensions |
| Brain control parameters | 8 | Workspace, learning, discount, habits, replay, sleep, inhibition, social updates |
| Functional modules | 16 | Software analogies, not biological brain regions simulated in detail |
| Synthetic route weights | 24 | Local experience can change these routing weights |
| Psychological needs | 5 | Safety, self-worth, autonomy, competence, belonging proxies |
| Body drives | 3 | Hunger, hydration need, sleep pressure |
| Affect variables | 4 | Valence, arousal, fear, frustration proxies |
| Appraisal dimensions | 4 | Novelty, goal congruence, controllability, social acceptance |
| Available simulation actions | 9 | Rest, explore, practice, socialize, help, sleep, eat, drink, reflect |
| Simulated skill counters | 3 | Reasoning, communication, craft; not objective IQ/task scores |
| Fresh-agent numeric state values | 115 | Includes counters and duplicate baseline values; not a trait count |
| Default genome loci | 3,000,000 | Synthetic diploid loci, 6 MB; not actual human genes |
| Atlas parcels / networks | 100 / 7 | Published anatomical metadata, not runtime neurons |

Inherited traits: curiosity, persistence, sociability, cooperation, caution, trust_prior, plasticity, memory_retention, attention, planning, reward_sensitivity, threat_sensitivity, novelty_preference, recovery, empathy, reciprocity.

Developing dispositions: emotional_regulation, negative_bias, positive_bias, uncertainty_tolerance, impulsivity, delay_discounting, attachment_security, assertiveness, achievement_drive, altruism, self_compassion, routine_preference.

There is no fixed total after learning. Episodic records, beliefs, goals, policies and classifier weights add values. Each classifier has 129 coefficients per class (128 lexical features plus bias), supports 2–20 classes, and the backend supports 30 learned classifiers. These trainable coefficients are not additional personality traits. Derived fatigue is computed from energy, stress and sleep pressure rather than stored as another independent trait.

## Changes with behavioral consequences

- Emotional regulation dampens sustained fear/frustration; positive and negative bias change affective valence.
- Fatigue reduces planning and attention proxies. Attention now changes relevance weighting during memory retrieval.
- Psychological need deficits bias recovery, exploration, practice, and social connection.
- Altruism and assertiveness influence helping and avoidance; attachment security influences social approach.
- Impulsivity changes choice stochasticity under stress; delay discounting reduces the contribution of planned returns.
- Routine preference biases repeated activities; low routine preference penalizes repetition.
- Attachment security, achievement drive, uncertainty tolerance and self-compassion adapt slowly from outcomes with a pull toward their configured baselines. Other dispositions remain configurable baselines.
- Existing agents migrate without changing genomes, inherited traits, profiles or past memories.

These mechanisms use engineered coefficients. The 12 defaults are derived from the existing 16 traits, so the additional names do not add 12 independently inherited dimensions. Users may configure them individually; no inference from real DNA or questionnaire validation is claimed.

## Bugs corrected

Forecasting now carries a complete numerical state, clamps bounded state variables and uses drinking's actual 0.1 resource cost. Previously drink was excluded whenever projected resources were below 0.5. Missing learned tools now produce explicit evidence errors instead of an internal failure. Tests cover these regressions, configuration rollback, migration, fatigue, regulation, decision biases and experience-dependent development.

## API

`GET /inventory` returns category counts. `GET /agents/{id}/inventory` also returns exact current numeric state paths.

`POST /agents/{id}/cognition` with:

```json
{"operation":"personality_configure","dispositions":{"emotional_regulation":0.8,"altruism":0.7},"source":"supplied_configuration"}
```

Values must be finite in [0,1]. Configuration is not a validated psychometric assessment. Run the full suite with `.venv/Scripts/python.exe -m unittest discover -s tests -v`.

## How close to humans?

We have a human-inspired simulation with persistence, distinct choices, constrained memory, learning, motives and social bookkeeping. There is no measured human similarity percentage. No real participant dataset, held-out human behavior test, or replication of a specific person has been completed.

The v4 adaptation rerun solved 40 unseen mazes with frozen learned controls and an engineered planner; synthetic text preference classification and symbolic belief fixtures remain narrow checks. They establish functioning mechanisms, not human-level understanding. Language-model weights remain fixed. Vision remains unverified after a local timeout.

Missing capabilities include biological genome-to-person mapping, a whole-brain connectome simulation, continuous embodied perception/action, broad lifelong learning, validated human emotional processes, and demonstrated consciousness. Atlas metadata and 16 software modules do not fill these gaps. The next defensible measure is task-specific agreement with held-out, consented human choices, compared with simple baselines and repeated-person consistency.

## Research motivation and limits

[Fleeson's experience-sampling study](https://pubmed.ncbi.nlm.nih.gov/11414368/) motivates distinguishing stable tendencies from changing personality states. [Appraisal-based affective agents](https://arxiv.org/abs/2309.05076) motivates supplying appraisal/state information to language responses. These sources do not validate our coefficients or demonstrate that this implementation reproduces their empirical results. No upstream algorithm or code from these papers was copied. Existing GitHub atlas data, pinned provenance and MIT license remain included.
