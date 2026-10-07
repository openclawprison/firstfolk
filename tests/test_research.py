import copy
import importlib.util
import json
import random
import tempfile
import unittest
from pathlib import Path

import brain
import capabilities
import mind
from engine import decide
from service import Service


class ResearchTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.path=Path(self.temp.name)/'agents.db'
        self.service=Service(self.path)
        self.a=self.service.create('Research',4096,17)

    def tearDown(self):
        self.temp.cleanup()

    def test_atlas_integrity(self):
        atlas=brain.atlas()
        self.assertEqual(len(atlas['parcels']),100)
        self.assertEqual(len({p['network'] for p in atlas['parcels']}),7)
        self.assertEqual(len(atlas['provenance']['commit']),40)
        self.assertTrue(all(len(p['centroid_ras_mm'])==3 for p in atlas['parcels']))
        import hashlib
        root=Path(brain.__file__).parent/'research'/'third_party'
        for filename,expected in atlas['provenance']['sha256'].items():
            self.assertEqual(hashlib.sha256((root/filename).read_bytes()).hexdigest(),expected)

    def test_brain_genetic_variation_and_observation(self):
        b=self.service.create('Other',4096,19)
        self.assertNotEqual(self.a['state']['brain']['weights'],b['state']['brain']['weights'])
        self.service.observe(self.a['id'],'A surprising event',confidence=1)
        mapping=self.service.brain_map(self.a['id'])
        self.assertGreater(next(r for r in mapping['regions'] if r['id']=='sensory')['activity'],0)
        self.assertEqual(len(mapping['edges']),24)

    def test_conflicting_evidence_and_provenance(self):
        for value in ('red','blue'):
            self.service.observe(self.a['id'],f'The object is {value}',claims=[{'subject':'object','predicate':'color','value':value}])
        knowledge=self.service.get(self.a['id'])['state']['knowledge']
        item=next(iter(knowledge.values()))
        self.assertTrue(item['conflicted'])
        self.assertEqual(set(item['alternatives']),{'red','blue'})
        self.assertEqual(item['alternatives']['red']['relative_support'],0.5)
        self.assertTrue(item['alternatives']['red']['evidence'][0]['memory_id'])

    def test_goal_bias_and_ablation(self):
        self.service.goal(self.a['id'],'resources',100,priority=1)
        a=self.service.get(self.a['id'])
        healthy=decide(a['traits'],a['state'],[],[],random.Random(0))
        self.service.lesion(a['id'],['executive'])
        injured=self.service.get(a['id'])
        disabled=decide(injured['traits'],injured['state'],[],[],random.Random(0))
        self.assertGreater(healthy['scores']['explore'],disabled['scores']['explore'])
        self.assertEqual(injured['state']['brain']['activities']['executive'],0)

    def test_working_memory_capacity_and_lesion(self):
        for i in range(20):
            self.service.observe(self.a['id'],f'Observation {i}')
        state=self.service.get(self.a['id'])['state']
        self.assertLessEqual(len(state['workspace']),8)
        self.service.lesion(self.a['id'],['working_memory','hippocampus'])
        self.service.observe(self.a['id'],'Another observation')
        self.assertEqual(self.service.get(self.a['id'])['state']['workspace'],[])
        self.assertEqual(self.service.memories(self.a['id']),[])

    def test_sleep_replay_does_not_duplicate_evidence(self):
        self.service.tick(steps=8,seed=8)
        self.service.consolidate(self.a['id'])
        self.service.consolidate(self.a['id'])
        state=self.service.get(self.a['id'])['state']
        ids=state['consolidated_memory_ids']
        self.assertTrue(ids)
        self.assertEqual(len(ids),len(set(ids)))
        self.assertTrue(state['knowledge'])
        self.assertGreater(state['metacognition']['samples'],0)

    def test_learned_world_model_has_evidence(self):
        self.service.tick(steps=20,seed=2)
        state=self.service.get(self.a['id'])['state']
        self.assertTrue(state['world_model'])
        self.assertEqual(sum(m['samples'] for m in state['world_model'].values()),20)
        action=next(iter(state['world_model']))
        prediction=mind.forecast(state,action)
        self.assertTrue(prediction['known'])
        self.assertGreater(prediction['samples'],0)
        self.assertTrue(prediction['plan'])

    def test_calculator_and_injection_boundary(self):
        self.assertEqual(capabilities.calculate('(41*17)+3')['result'],700)
        for expression in ("__import__('os').system('whoami')",'10**1000000','1/0'):
            with self.assertRaises(ValueError):
                capabilities.calculate(expression)

    def test_route_and_unreachable(self):
        result=capabilities.plan_route(['S..','.#.','..G'])
        self.assertTrue(result['reachable'])
        self.assertEqual(result['steps'],4)
        self.assertFalse(capabilities.plan_route(['S#G'])['reachable'])

    def test_policy_learning_and_persistence(self):
        result=self.service.train(self.a['id'],grid=['S..','.#.','..G'],episodes=200,seed=11)
        self.assertTrue(result['after']['success'])
        self.assertEqual(len(result['after']['path']),5)
        self.assertIn(result['policy_id'],Service(self.path).get(self.a['id'])['state']['procedural_memory'])

    @unittest.skipUnless(importlib.util.find_spec('gymnasium'),'Optional Gymnasium dependency absent')
    def test_real_gymnasium_learning(self):
        result=self.service.train(self.a['id'],kind='gym',episodes=1000,seed=42)
        self.assertGreater(result['after_success_rate'],result['before_success_rate'])
        self.assertGreaterEqual(result['after_success_rate'],0.9)

    def test_message_delivery_with_source(self):
        b=self.service.create('Peer',4096,12)
        self.service.message(self.a['id'],b['id'],'We should cooperate')
        self.assertEqual(self.service.memories(b['id'])[0]['source'],'agent_message')
        self.assertIn('cooperate',self.service.memories(b['id'])[0]['text'])

    def test_legacy_state_migration(self):
        with self.service.store.connection() as db:
            db.execute('UPDATE agents SET state=? WHERE id=?',(json.dumps({k:v for k,v in self.a['state'].items() if k in ('age_ticks','energy','stress','social_need','novelty_need','confidence','resources','skills','learned_values','self_model')}),self.a['id']))
        migrated=Service(self.path).get(self.a['id'])
        self.assertEqual(migrated['genome_hash'],self.a['genome_hash'])
        self.assertIn('brain',migrated['state'])


if __name__=='__main__':
    unittest.main()
