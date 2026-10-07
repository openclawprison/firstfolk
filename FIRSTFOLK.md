# Firstfolk

A persistent artificial-life world where four individuals begin with inherited variation and innate needs, then learn from local experience. You can watch their actions, bodies, relationships, vocabulary, offspring and deaths in a browser.

**This is a runnable research prototype, not a human clone. Human IQ, consciousness, full human language and 90% human equivalence have not been demonstrated.**

## Run the world

Requires Python 3.11 or newer. The world itself uses only the standard library.

```bash
python world_server.py
```

Open **http://127.0.0.1:8877** and click **Start life**. In PowerShell you can also run `./start-world.ps1`. The observer offers pause, speed, individual inspection, save and JSON export. The simulation starts paused so you can inspect the four founders before their first experience.

The save at `data/firstfolk/world-v7.json` includes genomes, all living/dead records, learned values, vocabulary and the random-generator state. Restarting resumes that world. To start a separate world without destroying the first:

```bash
python world_server.py --seed 19 --port 8878 --save data/firstfolk/seed19.json
```

There is no automatic resurrection or replacement after extinction. Saves are atomic; autosaving occurs while running. Export is a full continuation checkpoint. Local simulation time advances only while this process is running and unpaused. Population limits are 64 living individuals and 512 lifetime individuals; resources, policies, memory and logs are bounded for a laptop.

## What individuals receive

- Sixteen inherited engineering traits, plus the prior backend's 36 dispositions. Existing `personality.bias` and `human_dynamics.bias` are reused through an explicit action mapping. These are correlated controls, not 52 independently validated human measurements.
- Innate hunger, thirst, rest, social and mating urges. The designer supplies these drives and primitive action affordances; the learner discovers outcomes and changes its action values.
- Local terrain/resource cues, nearby individuals and internal sensations. Vision becomes shorter at night. There is no access to global coordinates, hidden terrain, implementation code, administrative events, an internet tool or a pretrained language model.
- Empty experience memories, empty action values and no supplied vocabulary. Founders start as simulated adults; this does not model human prenatal/childhood development. Newborns start with no inherited memories, policies or lexicons.

Agents are not told to build a house, find a cave, invent fire, start a civilization or believe they are human. No artificial-origin biography is supplied. This information boundary is implemented and tested; it is not a guarantee about what a future, more capable learner could infer.

## Mechanics and emergence

The environment supplies grass, water, trees, rocks and caves. It has sky/cloud rendering, rain and a day/night cycle. It is a small 2D symbolic world, not an Earth-scale physics model.

There are thirteen primitive actions: four movement directions, eating, drinking, resting, gathering, placing, rubbing, signaling, pairing and caregiving. There is no `build_house` or `make_fire` action. Placed objects affect cover; friction raises object temperature; dry combustible material can ignite, emit heat and burn away. Trees and caves supply passive cover. Agents receive no construction/fire discovery bonus. The material laws are scalar approximations, not realistic fluid mechanics, combustion or unrestricted engineering invention. Arbitrarily new mechanisms cannot appear unless the world supports them.

Each individual has ten coupled organ proxies: heart, lungs, gut, liver, kidneys, immune system, brain, muscles, skin and reproductive system. Nutrition, hydration, sleep, temperature, oxygen, infection, aging and organ damage affect functioning. These are deliberately simplified equations with engineering coefficients; they have not been fitted to clinical or human biological data. Pathogens are abstract transmissible states, not biological sequences. An individual's organ failure or aging ends its activity permanently.

Reproduction requires nearby, mature individuals of complementary simulated reproductive roles who both choose pairing, adequate resources, a cooldown and gestation. Children receive recombined synthetic diploid genomes with mutation; the world uses 4,096 loci for speed and reuses the backend's heredity implementation. No real human genomic association is implied. Synthetic inheritance and differential survival provide a mechanism for evolution; adaptation over generations must be measured, not assumed.

Care is optional. Dispositions, deprivation, local dependents and learned rewards affect whether an individual helps. A parent may choose another action and a child may be neglected or die. Birth and death are outcomes, not scheduled story beats.

Learning uses tabular Q updates, a learned action-effect model and local directional generalization, modulated by traits, fatigue and cognitive condition. Exploration remains stochastic. This is not human-level reasoning, open-ended invention, or a learned neural brain. Earlier ACT-R/MiniGrid integrations remain available in the separate cognitive API; they are not secretly invoked by this world's inhabitants.

Communication implements a naming-game convention: agents invent syllable tokens for jointly perceptible categories, hear new tokens and can converge on shared ones. These referent categories and syllables are supplied representational primitives. Vocabulary acquisition is not the emergence of human grammar, abstract discourse or English. Relationships and interactions can cross generations; children's vocabulary must be learned through encounters.

## What are they thinking?

Select an individual. The observer displays sensations, innate-drive contributions, learned action values, action-effect predictions, exploration and received reward. Those are the actual computational reasons available in the model. The UI does not generate a fictional inner monologue or claim to expose subjective experience. You can inspect all organs from the observer; that information is excluded from the agent observation.

## Evidence and research

Run tests and the multi-seed experiment:

```bash
python -m unittest discover -s tests -v
python benchmark_world.py --ticks 3000 --seeds 7 19 41
```

The benchmark saves [research/world_benchmark_v7.json](research/world_benchmark_v7.json), including learning-enabled and matched learning-disabled controls. Controls retain the same innate drives; policies diverge after experiences, so these are not identical action trajectories. Reports include birth/death counts, generations, vocabulary conventions and action counts. No IQ or human-equivalence score is emitted. Functional mechanism tests separately force conditions such as weak organs or material heating; those tests do not prove autonomous discovery.

Research basis and distinctions are documented in [research/WORLD_V7.md](research/WORLD_V7.md). The approach draws on [homeostatic reinforcement learning](https://elifesciences.org/articles/04811), [naming games](https://doi.org/10.3390/e23030299), and [multi-agent autocurricula](https://arxiv.org/abs/1909.07528). It is an original lightweight implementation inspired by these ideas, not a reproduction of their reported results or copied upstream simulator code.

The earlier cognitive backend is retained. See [BACKEND_V6.md](BACKEND_V6.md) for the JSON API on port 8765, optional dependency installation, ACT-R/MiniGrid integrations, atlas metadata, historical benchmarks and limitations. Optional dependencies have their own licenses and are not needed for the living world; see [third-party notices](research/third_party/upstream_packages/NOTICES.md).
