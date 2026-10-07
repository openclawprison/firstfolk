"""Reproducible capability and ablation report, without claiming human equivalence."""
import json
import random
import tempfile
from pathlib import Path
from service import Service
from engine import decide


def run():
    with tempfile.TemporaryDirectory() as tmp:
        service=Service(Path(tmp)/'benchmark.db')
        agent=service.create('Benchmark',loci=4096,seed=17)
        arithmetic=[]
        for expression,expected in [('(41*17)+3',700),('2**10',1024),('(15-3)/4',3)]:
            actual=service.solve(agent['id'],'arithmetic',expression=expression)['result']
            arithmetic.append({'expression':expression,'expected':expected,'actual':actual,'passed':actual==expected})
        route=service.solve(agent['id'],'route',grid=['S...','.#.#','....','##.G'])
        learning=[]
        gym=[]
        for seed in range(5):
            a=service.create(f'Learner {seed}',4096,seed)
            result=service.train(a['id'],grid=['S..','.#.','..G'],episodes=200,seed=seed)
            learning.append({'seed':seed,'before':result['before']['success'],'after':result['after']['success'],
                             'training_success_rate':result['training_success_rate']})
            try:
                result=service.train(a['id'],kind='gym',episodes=1000,seed=seed)
                gym.append({'seed':seed,'before':result['before_success_rate'],'after':result['after_success_rate'],
                            'version':result['gymnasium_version']})
            except ValueError as exc:
                gym.append({'seed':seed,'unavailable':str(exc)})
        service.goal(agent['id'],'resources',100,priority=1)
        a=service.get(agent['id'])
        healthy=decide(a['traits'],a['state'],[],[],random.Random(0))
        recall_before=len(service.memories(agent['id']))
        service.lesion(agent['id'],['executive','hippocampus'])
        a=service.get(agent['id'])
        disabled=decide(a['traits'],a['state'],[],[],random.Random(0))
        ablation={'healthy_exploration_score':healthy['scores']['explore'],
                  'disabled_executive_exploration_score':disabled['scores']['explore'],
                  'healthy_recalled_memories':recall_before,
                  'disabled_hippocampus_recalled_memories':len(service.memories(agent['id']))}
        return {'arithmetic':arithmetic,'route':route,'grid_learning':learning,'gymnasium_learning':gym,
                'ablation':ablation,'limitations':'Small fixed environments and software interventions; no human behavioral validation or general intelligence measurement.'}


if __name__=='__main__':
    report=run()
    destination=Path(__file__).parent/'research'/'benchmark_results.json'
    destination.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
