"""Behavioral interventions and bounded community run; no human comparator."""
import copy
import json
import random
import tempfile
from pathlib import Path
from service import Service
from engine import ACTIONS,decide
import human_dynamics as dynamics


def run():
    with tempfile.TemporaryDirectory() as tmp:
        service=Service(Path(tmp)/'benchmark.db')
        agent=service.create('Synthetic intervention fixture',4096,51)
        facets=[]
        for name,(target,direction,signal,_) in dynamics.FACETS.items():
            state=copy.deepcopy(agent['state'])
            low={action:0 for action in ACTIONS};high=dict(low)
            state['personality']['dispositions'][name]=0
            dynamics.bias(low,state)
            state['personality']['dispositions'][name]=1
            dynamics.bias(high,state)
            facets.append({'trait':name,'target':target,'signal':signal,
                           'score_change':high[target]-low[target],
                           'expected_direction':direction,'causal_check':direction*(high[target]-low[target])>0})
        agents=[agent]+[service.create(f'Synthetic peer {i}',4096,i) for i in range(5)]
        before={a['id']:a['genome_hash'] for a in agents}
        # Two equal-genome agents with opposed configured profiles.
        creative=service.create('Creative configured fixture',4096,99)
        cautious=service.create('Cautious configured fixture',4096,99)
        for a,values in ((creative,{'creativity':1,'playfulness':1,'loss_aversion':0,'ambiguity_aversion':0}),
                         (cautious,{'creativity':0,'playfulness':0,'loss_aversion':1,'ambiguity_aversion':1})):
            service.cognition_operation(a['id'],'personality_configure',dispositions=values,source='synthetic_intervention')
        choices={}
        relations=[{'other_id':agent['id'],'trust':0.5}]
        for a in (creative,cautious):
            s=service.get(a['id'])['state']
            counts={action:0 for action in ACTIONS}
            for seed in range(1000):counts[decide(a['traits'],s,[],relations,random.Random(seed))['action']]+=1
            choices[a['name']]=counts
        emitted=service.tick(steps=100,seed=5)['events']
        action_counts={action:sum(e['outcome']['action']==action for e in emitted) for action in ACTIONS}
        current=service.list()
        return {'facet_interventions':facets,'all_new_facets_causal':all(f['causal_check'] for f in facets),
                'equal_genome_profile_intervention':{'same_genome':creative['genome_hash']==cautious['genome_hash'],
                    'samples_per_profile':1000,'action_counts':choices,'evaluation_scope':'choice distribution under one fixed synthetic context'},
                'community':{'agents':len(current),'ticks':100,'action_counts':action_counts,
                    'genomes_preserved':all(service.get(aid)['genome_hash']==digest for aid,digest in before.items()),
                    'artifact_count':sum(len(a['state']['artifacts']) for a in current),
                    'relationship_histories':sum(len(a['state']['social_dynamics']) for a in current)},
                'human_comparator':False,'human_fidelity_measured':False,
                'limitations':['Interventions test the engineered mechanisms, not their psychological validity.',
                               'Rewards, cooperation and environments are synthetic.',
                               'Choice frequencies are not a human personality score.']}


if __name__=='__main__':
    result=run()
    path=Path(__file__).parent/'research'/'behavior_benchmark_v5.json'
    path.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({'new_facets_causal':result['all_new_facets_causal'],'community':result['community']},indent=2))
