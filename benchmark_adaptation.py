"""Held-out transfer and small behavioral-model tests; no human equivalence score."""
import json
import tempfile
from pathlib import Path
import transfer
from service import Service

TRAIN=[{'text':'quiet reading alone at home','label':'quiet'},
       {'text':'quiet books peaceful library','label':'quiet'},
       {'text':'alone studying quiet evening','label':'quiet'},
       {'text':'social party friends dancing','label':'social'},
       {'text':'friends music party crowd','label':'social'},
       {'text':'social dancing group evening','label':'social'}]
HOLDOUT=[{'text':'quiet peaceful reading','label':'quiet'},
         {'text':'party dancing friends','label':'social'},
         {'text':'alone quiet library','label':'quiet'},
         {'text':'social friends crowd','label':'social'}]


def run():
    with tempfile.TemporaryDirectory() as tmp:
        service=Service(Path(tmp)/'agent.db')
        agent=service.create('Synthetic research participant',4096,3)
        aid=agent['id']
        service.profile(aid,'Fictional Mira',biography=['I study engineering.'],values=['fairness'],
                        preferences=['quiet reading'],source='synthetic_fixture_not_human_data')
        training=service.cognition_operation(aid,'transfer_train',grids=[transfer.generate_grid(1,7),transfer.generate_grid(2,7)],steps_per_grid=300,seed=0)
        maze11=service.cognition_operation(aid,'transfer_evaluate',size=11,seeds=list(range(1000,1020)),max_steps=300)
        maze15=service.cognition_operation(aid,'transfer_evaluate',size=15,seeds=list(range(2000,2020)),max_steps=600)
        open_grid=['#########','#S......#','#.......#','#.......#','#.......#','#.......#','#.......#','#......G#','#########']
        shifted=service.cognition_operation(aid,'navigate',grid=open_grid,control_order=[1,2,3,0],adapt=True,seed=11,max_steps=500)
        class_training=service.cognition_operation(aid,'skill_train',name='fixture_preferences',examples=TRAIN,epochs=60)
        class_evaluation=service.cognition_operation(aid,'skill_evaluate',name='fixture_preferences',examples=HOLDOUT)
        student=service.create('Student',4096,4)
        teaching=service.teach_skill(aid,student['id'],'fixture_preferences')
        student_evaluation=service.cognition_operation(student['id'],'skill_evaluate',name='fixture_preferences',examples=HOLDOUT)
        beliefs=[]
        for i in range(20):
            obj=f'object{i}'
            actor=f'actor{i}'
            service.cognition_operation(aid,'perspective_event',object_name=obj,location='original',observers=[actor])
            visible=i%2==0
            service.cognition_operation(aid,'perspective_event',object_name=obj,location='new',observers=[actor] if visible else [])
            result=service.cognition_operation(aid,'perspective_query',actor=actor,object_name=obj)
            expected='new' if visible else 'original'
            beliefs.append({'visible':visible,'expected':expected,'actual':result['believed_location'],'correct':expected==result['believed_location']})
        return {'movement_training':training,'heldout_11x11':maze11,'heldout_15x15':maze15,
                'changed_controls':shifted,'classifier_training':class_training,
                'classifier_holdout':class_evaluation,'teaching':teaching,'student_holdout':student_evaluation,
                'belief_tracking':{'trials':beliefs,'accuracy':sum(b['correct'] for b in beliefs)/len(beliefs)},
                'human_data_used':False,'human_fidelity_measured':False,
                'limitations':['Planner is engineered; only primitive transition dynamics are learned.',
                               'Classifiers learn lexical patterns from six synthetic examples.',
                               'Belief tracking uses declared event visibility, not inferred human mental states.',
                               'No human comparator, real participant data, whole-brain simulation or general intelligence result.']}


if __name__=='__main__':
    result=run()
    root=Path(__file__).parent/'research'
    (root/'adaptation_benchmark.json').write_text(json.dumps(result,indent=2))
    summary={'maze11':{k:result['heldout_11x11'][k] for k in ('baseline_success_rate','transfer_success_rate')},
             'maze15':{k:result['heldout_15x15'][k] for k in ('baseline_success_rate','transfer_success_rate')},
             'changed_controls_success':result['changed_controls']['success'],
             'classifier_holdout':result['classifier_holdout']['accuracy'],
             'student_holdout':result['student_holdout']['accuracy'],
             'belief_bookkeeping':result['belief_tracking']['accuracy'],
             'human_fidelity_measured':False}
    print(json.dumps(summary,indent=2))
