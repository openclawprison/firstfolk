"""Reproducible artificial-life outcomes and matched no-learning controls."""
import argparse
import json
from pathlib import Path
import statistics
import time

from living_world import World


def run(seed, ticks, learning):
    w = World(seed=seed, learning=learning)
    for _ in range(ticks//200):
        w.step(200)
    if ticks % 200:
        w.step(ticks % 200)
    agents = w.data['agents']
    actions = {}
    for a in agents:
        for k, v in a['learner']['actions'].items():
            actions[k] = actions.get(k, 0)+v
    return {'seed': seed, 'ticks': ticks, 'learning': learning,
        'stats': w.snapshot()['stats'], 'actions': actions,
        'learning_updates': sum(a['learner']['updates'] for a in agents),
        'mean_founder_age_gain': statistics.mean(a['age']-600 for a in agents if not a['parents']),
        'health': [round(a['health'], 4) for a in agents if a['alive']],
        'vocabularies': {str(a['id']): a['lexicon'] for a in agents},
        'inherited_memory_count': 0,
        'note': 'Generated outcomes, not predetermined milestones or human validation'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ticks', type=int, default=3000)
    parser.add_argument('--seeds', type=int, nargs='+', default=[7, 19, 41])
    parser.add_argument('--output', default='research/world_benchmark_v7.json')
    args = parser.parse_args()
    if not 1 <= args.ticks <= 20000 or not 1 <= len(args.seeds) <= 10:
        raise ValueError('ticks 1..20000; seeds 1..10')
    start = time.monotonic()
    trials = [run(seed, args.ticks, enabled) for seed in args.seeds for enabled in (True, False)]
    result = {'trials': trials, 'elapsed_seconds': round(time.monotonic()-start, 2),
        'scope': 'Matched seeds and innate drives; policies diverge as learning changes actions',
        'human_iq': None, 'human_equivalence_percent': None,
        'language_scope': 'shared referent tokens; no grammar claim',
        'evolution_scope': 'inheritance and selection possible; adaptive evolution needs repeated-generation evidence'}
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({'output': str(path), 'elapsed_seconds': result['elapsed_seconds'],
                     'trials': [{k: t[k] for k in ('seed','learning','stats','mean_founder_age_gain')} for t in trials]}, indent=2))


if __name__ == '__main__':
    main()
