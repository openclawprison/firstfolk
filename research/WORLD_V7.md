# Firstfolk research and scope — 7 October 2026

## Research informing the design

1. Keramati and Gutkin (2014), [Homeostatic reinforcement learning for integrating reward collection and physiological stability](https://elifesciences.org/articles/04811), DOI 10.7554/eLife.04811. Physiological internal state can help define motivated learning. Firstfolk uses change in an engineered body/social utility plus a novelty term; it does not reproduce the paper's formal model or its biological validation.
2. Lipowska and Lipowski (2021), [Evolution towards Linguistic Coherence in Naming Game with Migrating Agents](https://doi.org/10.3390/e23030299). Naming games study population conventions. Firstfolk implements a simplified shared-referent token inventory with success consolidation. It does not claim human language or reproduce that paper's spatial experiments.
3. Baker et al. (2019), [Emergent Tool Use From Multi-Agent Autocurricula](https://arxiv.org/abs/1909.07528). Complex behavior can arise from interaction, affordances and training objectives. Firstfolk's small tabular learners and survival drives are substantially weaker than those experiments; open-ended civilization does not follow from adding these mechanisms.
4. [Melting Pot](https://github.com/google-deepmind/meltingpot) and [PettingZoo](https://github.com/Farama-Foundation/PettingZoo) are relevant existing multi-agent research platforms. Reviewed as future integration foundations; neither is installed or invoked by this world. No upstream code from them is claimed as integrated.
5. The prior backend actually integrates [pyactr](https://github.com/jakdot/pyactr) and [MiniGrid](https://github.com/Farama-Foundation/Minigrid) through optional adapters, documented in RESEARCH_V6.md. World agents reuse our genome, personality and human-dynamics code, not the optional language model or ACT-R adapter.

## Supplied machinery versus learned outcomes

Supplied: synthetic trait expression, 36 disposition mappings, thirteen action affordances, local symbolic perception, innate drives, environmental/material laws, organ equations, reproductive compatibility and maturation, pathogen rules, communication referent categories, syllable alphabet, reward definition and learning algorithm.

Learned: experienced action values and action-effect predictions; which local actions help under encountered conditions; syllable-token associations and conventions. Action exploration and trait variation can lead to different trajectories. Newborn memories and policies are empty. No manual shelter or fire objective is used.

Not established: human IQ, human consciousness, absence of every possible simulation-origin inference, human grammar, realistic biological DNA-to-personality mapping, a complete human organ system, neural/connectome equivalence, human developmental learning, adaptation beyond a few generations, or technological progress outside the available material laws.

## Assessment

The experiment must be assessed through observed survival, reproduction, learned-value changes, frozen-policy comparisons, language agreement, trait distributions and repeated independent world seeds. Selection and mutation are mechanisms; a small number of births alone is not evidence of adaptive evolution. A naming-game token is not a natural language. An observer decision trace is not proof of felt thought.

The tests verify observation boundaries, deterministic continuation, innate-drive influence, policy updates, optional care, two-sided pairing, newborn policy isolation, permanent death, coupled organ failure, passive cover, primitive object placement and friction-driven ignition. An ignition test deliberately applies friction; it proves the material mechanic, not that agents independently invented fire.

See world_benchmark_v7.json for the actual autonomous-run outcomes. The learning-disabled condition freezes Q values, outcome-model updates and lexical uptake while retaining innate drives and token invention. It is a control, not a claim about a human population.

## Recorded results

All 91 tests passed with optional dependencies installed; the standard-library run passed 82 and skipped 9 optional-provider tests. Across three learning-enabled seeds at 3,000 ticks each, there were 18 births, no deaths, a maximum generation index of 3, and 142 ignition events. Matched no-learning controls also produced births and ignitions: these outcomes alone do not demonstrate intelligence or intentional invention. Learning-enabled runs recorded 108 successful naming-game agreements across 207 conversations; controls recorded zero agreements. None of these short runs demonstrated mortality; permanent death is verified separately by forced organ-failure tests. See verification_v7.json and world_benchmark_v7.json for complete records.
