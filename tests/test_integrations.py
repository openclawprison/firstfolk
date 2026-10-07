import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from service import Service
import actr_memory
import embodied
from unittest.mock import patch
from cognition import reason

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


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.path=Path(self.tmp.name)/'db'
        self.service=Service(self.path)
        self.a=self.service.create('Synthetic integration fixture',4096,17)
    def tearDown(self):self.tmp.cleanup()
    def op(self,name,**kw):return self.service.cognition_operation(self.a['id'],name,**kw)

    def test_participant_heldout_evaluation_has_no_human_score(self):
        self.op('participant_train',participant='fictional',examples=TRAIN,source='synthetic_fixture')
        before=self.service.get(self.a['id'])['state']['learned_skills']
        result=self.op('participant_evaluate',participant='fictional',examples=TEST)
        self.assertGreaterEqual(result['agreement_percent'],75)
        self.assertEqual(result['majority_baseline_percent'],50)
        self.assertIsNone(result['human_equivalence_percent'])
        self.assertFalse(result['source_verified'])
        self.assertTrue(result['model_unchanged'])
        self.assertEqual(before,self.service.get(self.a['id'])['state']['learned_skills'])

    def test_normalized_training_overlap_rejected(self):
        self.op('participant_train',participant='fictional',examples=TRAIN,source='synthetic')
        heldout=copy.deepcopy(TEST);heldout[0]['context']='  QUIET  reading alone at HOME '
        with self.assertRaises(ValueError):self.op('participant_evaluate',participant='fictional',examples=heldout)

    def test_duplicate_heldout_contexts_rejected(self):
        self.op('participant_train',participant='fictional',examples=TRAIN,source='synthetic')
        with self.assertRaises(ValueError):self.op('participant_evaluate',participant='fictional',examples=[TEST[0],TEST[0]])

    def test_invalid_record_rolls_back(self):
        before=self.service.get(self.a['id'])['state']
        with self.assertRaises(ValueError):self.op('participant_train',participant='fictional',examples=[{'context':'x','action':'shell'}]*2,source='synthetic')
        self.assertEqual(before,self.service.get(self.a['id'])['state'])

    def test_untrained_human_actions_are_scored_rather_than_excluded(self):
        self.op('participant_train',participant='fictional',examples=TRAIN,source='synthetic')
        unseen=copy.deepcopy(TEST);unseen[0]['action']='explore'
        result=self.op('participant_evaluate',participant='fictional',examples=unseen)
        self.assertEqual(result['unseen_action_count'],1)
        self.assertLessEqual(result['agreement_percent'],75)

    @patch.dict('os.environ',{'AGENT_MODEL_URL':'http://localhost:1234/v1/chat/completions','AGENT_MODEL_NAME':'fixture'})
    @patch('cognition.request_json')
    def test_reasoning_uses_learned_participant_prediction(self,request):
        self.op('participant_train',participant='fictional',examples=TRAIN,source='synthetic')
        request.side_effect=[{'calls':[{'kind':'participant','participant':'fictional','context':'quiet peaceful reading'}]},
                             {'answer':'The model predicts rest.','uncertainty':'Synthetic examples only','cited_memory_ids':[]}]
        result=reason(self.service,self.a['id'],'What would the fictional participant choose?')
        evidence=result['tool_evidence'][0]
        self.assertEqual(evidence['result']['predicted_action'],'rest')
        self.assertFalse(evidence['verified'])
        self.assertTrue(evidence['execution_verified'])

    @unittest.skipUnless(importlib.util.find_spec('minigrid'),'Optional MiniGrid absent')
    def test_actual_partial_observation_learning_persists_and_freezes(self):
        learned=self.op('embodied_train',episodes=600,seed=0)
        self.assertGreaterEqual(learned['after_success_rate'],0.9)
        before=Service(self.path).get(self.a['id'])['state']['procedural_memory']
        self.assertTrue(any(any(v!=0 for v in row) for row in before[learned['policy_id']]['q'].values()))
        evaluated=self.op('embodied_policy_evaluate',seeds=list(range(910000,910010)))
        self.assertTrue(evaluated['frozen_weights'])
        self.assertEqual(before,Service(self.path).get(self.a['id'])['state']['procedural_memory'])
        with self.assertRaises(ValueError):self.op('embodied_policy_evaluate',seeds=[1])

    @unittest.skipUnless(importlib.util.find_spec('pyactr'),'Optional pyactr absent')
    def test_upstream_actr_recency_activation_and_latency(self):
        memories=[{'id':1,'tick':0,'action':'rest','text':'old'},{'id':2,'tick':9,'action':'rest','text':'new'}]
        result=actr_memory.rank(self.a['state'],memories,10,2,rehearse=False)
        self.assertEqual(result['provider'],'pyactr')
        self.assertEqual(result['memories'][0]['id'],2)
        self.assertLess(result['memories'][0]['modeled_latency'],result['memories'][1]['modeled_latency'])

    @unittest.skipUnless(importlib.util.find_spec('pyactr'),'Optional pyactr absent')
    def test_actr_rehearsal_and_trace_persistence(self):
        self.service.experience(self.a['id'],'A remembered rest experience',0.5,'rest')
        self.op('memory_mode',mode='actr')
        self.service.memories(self.a['id'],'rest')
        self.service.memories(self.a['id'],'rest')
        state=Service(self.path).get(self.a['id'])['state']
        self.assertGreaterEqual(state['actr_memory']['retrievals'],2)
        self.assertTrue(any(len(times)>1 for times in state['actr_memory']['presentations'].values()))

    @unittest.skipUnless(importlib.util.find_spec('pyactr'),'Optional pyactr absent')
    def test_actr_high_threshold_fails_retrieval(self):
        result=actr_memory.rank(self.a['state'],[{'id':1,'tick':0,'action':'rest','text':'x'}],1000,threshold=10)
        self.assertEqual(result['memories'],[])

    @unittest.skipUnless(importlib.util.find_spec('pyactr'),'Optional pyactr absent')
    def test_actr_respects_ablation(self):
        self.service.lesion(self.a['id'],['hippocampus'])
        with self.assertRaises(ValueError):self.op('actr_recall')

    @unittest.skipUnless(importlib.util.find_spec('minigrid'),'Optional MiniGrid absent')
    def test_real_doorkey_requires_object_interactions(self):
        result=embodied.episode('MiniGrid-DoorKey-6x6-v0',seed=1,max_steps=200)
        self.assertTrue(result['success'])
        self.assertGreater(result['interaction_attempts']['pickup'],0)
        self.assertGreater(result['interaction_attempts']['toggle'],0)
        self.assertFalse(result['learned_policy'])

    @unittest.skipUnless(importlib.util.find_spec('minigrid'),'Optional MiniGrid absent')
    def test_controller_receives_no_hidden_grid_or_position(self):
        import gymnasium as gym
        import minigrid
        env=gym.make('MiniGrid-DoorKey-6x6-v0')
        try:
            observation,_=env.reset(seed=1)
            self.assertEqual(set(observation),{'image','direction','mission'})
            controller=embodied.Controller()
            action,_=controller.action(observation)
            self.assertIn(action,range(7))
            self.assertNotIn('env',controller.__dict__)
        finally:env.close()

    @unittest.skipUnless(importlib.util.find_spec('minigrid'),'Optional MiniGrid absent')
    def test_embodied_report_persists_separately_from_learning(self):
        result=self.op('embodied_evaluate',seeds=[1,2],max_steps=200)
        self.assertEqual(result['seed_count'],2)
        self.assertFalse(result['training_used'])
        self.assertTrue(Service(self.path).get(self.a['id'])['state']['embodied_evaluations'])
