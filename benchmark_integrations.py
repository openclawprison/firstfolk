"""Upstream provider checks, real environments and synthetic participant fixtures."""
import copy
import json
import tempfile
from pathlib import Path
from service import Service
import actr_memory
import embodied
import embodied_learning

TRAIN=[{'context':'quiet reading alone at home','action':'rest'},
       {'context':'quiet books peaceful library','action':'rest'},
       {'context':'alone quiet evening','action':'rest'},
       {'context':'social party friends dancing','action':'socialize'},
       {'context':'social friends party crowd','action':'socialize'},
       {'context':'social dancing group evening','action':'socialize'}]
TEST=[{'context':'quiet peaceful reading','action':'rest'},
      {'context':'party dancing friends','action':'socialize'},
      {'context':'alone quiet library','action':'rest'},
      {'context':'social friends crowd','action':'socialize'}]


def run():
    with tempfile.TemporaryDirectory() as tmp:
        service=Service(Path(tmp)/'benchmark.db');agent=service.create('Synthetic fixture',4096,31)
        aid=agent['id'];state=agent['state']
        memories=[{'id':1,'tick':0,'action':'rest','text':'Remembered event'}]
        near=actr_memory.rank(copy.deepcopy(state),memories,1,rehearse=False)
        far=actr_memory.rank(copy.deepcopy(state),memories,100,rehearse=False)
        rehearsed=copy.deepcopy(state)
        for tick in (1,2,3,4):actr_memory.rank(rehearsed,memories,tick)
        repeat=actr_memory.rank(rehearsed,memories,100,rehearse=False)
        retrieval={'provider':near['provider'],'version':near['version'],
                   'near':near['memories'][0],'far':far['memories'][0],
                   'rehearsed':repeat['memories'][0],
                   'expected_forgetting_observed':far['memories'][0]['modeled_latency']>near['memories'][0]['modeled_latency'],
                   'expected_rehearsal_observed':repeat['memories'][0]['modeled_latency']<far['memories'][0]['modeled_latency'],
                   'human_response_time_data_used':False}
        rooms=[embodied.evaluate(state,environment=env,seeds=list(range(20)),max_steps=200)
               for env in ('MiniGrid-DoorKey-6x6-v0','MiniGrid-DoorKey-8x8-v0')]
        learning=[]
        for seed in (0,11,22):
            a=service.create(f'Learner {seed}',4096,seed)
            result=service.cognition_operation(a['id'],'embodied_train',environment='MiniGrid-Empty-Random-6x6-v0',episodes=600,seed=seed)
            evaluated=service.cognition_operation(a['id'],'embodied_policy_evaluate',environment='MiniGrid-Empty-Random-6x6-v0',seeds=list(range(910000,910020)))
            learning.append({'seed':seed,'training':result,'frozen_evaluation':evaluated})
        service.cognition_operation(aid,'participant_train',participant='fictional',examples=TRAIN,source='synthetic_fixture_not_human_data')
        participant=service.cognition_operation(aid,'participant_evaluate',participant='fictional',examples=TEST)
        return {'actr_memory':retrieval,'doorkey_planning':rooms,'partial_observation_learning':learning,
                'participant_fixture':participant,'human_fidelity_measured':False,'ninety_percent_human_claim_supported':False,
                'scope':'Provider and task checks. No whole-person benchmark or real participant dataset used.'}


if __name__=='__main__':
    report=run()
    (Path(__file__).parent/'research/integrations_benchmark_v6.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({'actr_forgetting':report['actr_memory']['expected_forgetting_observed'],
                      'actr_rehearsal':report['actr_memory']['expected_rehearsal_observed'],
                      'doorkey_success':[r['success_rate'] for r in report['doorkey_planning']],
                      'learning_before':[r['training']['before_success_rate'] for r in report['partial_observation_learning']],
                      'learning_after':[r['frozen_evaluation']['success_rate'] for r in report['partial_observation_learning']],
                      'synthetic_choice_agreement':report['participant_fixture']['agreement_percent'],
                      'human_fidelity_measured':False},indent=2))
