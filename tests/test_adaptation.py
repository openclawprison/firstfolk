import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from service import Service
import transfer
import skills
import social_model
from cognition import think
from engine import decide
import random


TRAIN_EXAMPLES=[
    {'text':'quiet reading alone at home','label':'quiet'},
    {'text':'quiet books peaceful library','label':'quiet'},
    {'text':'alone studying quiet evening','label':'quiet'},
    {'text':'social party friends dancing','label':'social'},
    {'text':'friends music party crowd','label':'social'},
    {'text':'social dancing group evening','label':'social'},
]
TEST_EXAMPLES=[{'text':'quiet peaceful reading','label':'quiet'},
               {'text':'party dancing friends','label':'social'},
               {'text':'alone quiet library','label':'quiet'},
               {'text':'social friends crowd','label':'social'}]


class AdaptationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.path=Path(self.tmp.name)/'agent.db'
        self.service=Service(self.path)
        self.a=self.service.create('Adaptive',4096,3)

    def tearDown(self):
        self.tmp.cleanup()

    def operation(self,operation,**arguments):
        return self.service.cognition_operation(self.a['id'],operation,**arguments)

    def train_movement(self):
        return self.operation('transfer_train',grids=[transfer.generate_grid(1,7),transfer.generate_grid(2,7)],steps_per_grid=300)

    def test_unseen_layout_transfer_with_frozen_model(self):
        self.train_movement()
        before=self.service.get(self.a['id'])['state']['transfer_memory']
        result=self.operation('transfer_evaluate',seeds=list(range(1000,1010)),max_steps=300)
        self.assertGreaterEqual(result['transfer_success_rate'],0.9)
        self.assertGreater(result['transfer_success_rate'],result['baseline_success_rate'])
        after=self.service.get(self.a['id'])['state']['transfer_memory']
        self.assertEqual(before,after)

    def test_local_observations_do_not_reveal_grid(self):
        grid=transfer.generate_grid(1000,11)
        obs=transfer.observe(grid,(1,1),(9,9))
        self.assertEqual(len(obs['cells']),5)
        self.assertNotIn('grid',obs)

    def test_control_shift_adaptation(self):
        self.train_movement()
        grid=['#########','#S......#','#.......#','#.......#','#.......#','#.......#','#.......#','#......G#','#########']
        result=self.operation('navigate',grid=grid,control_order=[1,2,3,0],adapt=True,seed=11,max_steps=500)
        self.assertTrue(result['success'])
        self.assertGreater(result['control_shifts_detected'],0)
        self.assertNotEqual(self.service.get(self.a['id'])['state']['transfer_memory']['movement-v1']['actions'],{})

    def test_holdout_layout_leakage_rejected(self):
        self.operation('transfer_train',grids=[transfer.generate_grid(1000,11)],steps_per_grid=100)
        with self.assertRaises(ValueError):
            self.operation('transfer_evaluate',seeds=[1000])

    def test_actual_weight_learning_and_heldout_examples(self):
        self.operation('skill_train',name='preference',examples=TRAIN_EXAMPLES,epochs=60)
        result=self.operation('skill_evaluate',name='preference',examples=TEST_EXAMPLES)
        self.assertGreaterEqual(result['accuracy'],0.75)
        state=self.service.get(self.a['id'])['state']
        self.assertTrue(any(v!=0 for row in state['learned_skills']['preference']['weights'] for v in row))
        fresh=Service(self.path)
        self.assertEqual(fresh.get(self.a['id'])['state']['learned_skills'],state['learned_skills'])

    def test_incremental_feedback_updates_weights(self):
        self.operation('skill_train',name='preference',examples=TRAIN_EXAMPLES,epochs=10)
        before=self.service.get(self.a['id'])['state']['learned_skills']['preference']
        self.operation('skill_train',name='preference',examples=[{'text':'peaceful quiet garden','label':'quiet'}],epochs=5)
        after=self.service.get(self.a['id'])['state']['learned_skills']['preference']
        self.assertNotEqual(before['weights'],after['weights'])
        self.assertGreater(after['updates'],before['updates'])
        self.assertEqual(after['labels'],before['labels'])

    def test_classifier_evaluation_leakage_rejected(self):
        self.operation('skill_train',name='preference',examples=TRAIN_EXAMPLES)
        with self.assertRaises(ValueError):
            self.operation('skill_evaluate',name='preference',examples=TRAIN_EXAMPLES)

    def test_false_belief_and_unknown_actor(self):
        self.operation('perspective_event',object_name='ball',location='box',observers=['A','B'])
        self.operation('perspective_event',object_name='ball',location='basket',observers=['B'])
        result=self.operation('perspective_query',actor='A',object_name='ball')
        self.assertEqual(result['believed_location'],'box')
        self.assertEqual(result['actual_location'],'basket')
        self.assertTrue(result['belief_differs_from_world'])
        result=self.operation('perspective_query',actor='C',object_name='ball')
        self.assertIsNone(result['believed_location'])

    @patch.dict('os.environ',{'AGENT_MODEL_URL':'http://127.0.0.1:1234/v1/chat/completions','AGENT_MODEL_NAME':'test'})
    @patch('cognition.request_json')
    def test_identity_profile_is_persisted_and_used(self,request):
        self.service.profile(self.a['id'],'Fictional Mira',biography=['I study engineering'],values=['fairness'],preferences=['quiet reading'],source='synthetic_fixture')
        request.return_value={'answer':'I prefer quiet reading.','uncertainty':'Profile based','cited_memory_ids':[]}
        think(self.service,self.a['id'],'What do you like?')
        prompt=request.call_args.args[0][0]['content']
        self.assertIn('Fictional Mira',prompt)
        self.assertIn('synthetic_fixture',prompt)
        self.assertIn('without claiming to be the actual person',prompt)

    def test_invalid_operation_rolls_back(self):
        before=self.service.get(self.a['id'])['state']
        with self.assertRaises(ValueError):
            self.operation('not_real')
        self.assertEqual(before,self.service.get(self.a['id'])['state'])

    def test_teaching_transfers_learned_skill(self):
        self.operation('skill_train',name='preference',examples=TRAIN_EXAMPLES)
        child=self.service.create('Student',4096,4)
        self.service.teach_skill(self.a['id'],child['id'],'preference')
        result=self.service.cognition_operation(child['id'],'skill_evaluate',name='preference',examples=TEST_EXAMPLES)
        self.assertGreaterEqual(result['accuracy'],0.75)
        self.assertEqual(self.service.get(child['id'])['state']['identity_profile'],{})

    def test_learned_preferences_affect_autonomous_decisions(self):
        a=self.service.get(self.a['id'])
        original=decide(a['traits'],a['state'],[],[],random.Random(1))
        context='energy_medium stress_low social_need_medium novelty_need_medium'
        examples=[{'text':context,'label':'rest'},
                  {'text':'energy_high stress_low social_need_low novelty_need_high','label':'explore'}]
        self.operation('skill_train',name='action_preferences',examples=examples,epochs=100)
        a=self.service.get(self.a['id'])
        # Match the taught state; this verifies the causal connection to runtime.
        a['state']['energy']=0.5
        original=decide(a['traits'],{**a['state'],'learned_skills':{}},[],[],random.Random(1))
        learned=decide(a['traits'],a['state'],[],[],random.Random(1))
        self.assertGreater(learned['scores']['rest'],original['scores']['rest'])
        self.assertIsNotNone(learned['learned_preference_evidence'])


if __name__=='__main__':
    unittest.main()
