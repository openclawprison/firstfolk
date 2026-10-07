import copy
import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from service import Service
from engine import ACTIONS,decide,transition,retrieve
import human_dynamics as dynamics
import mind


class DynamicsTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.path=Path(self.tmp.name)/'db.sqlite'
        self.service=Service(self.path)
        self.agent=self.service.create('Synthetic dynamics fixture',4096,5)

    def tearDown(self):self.tmp.cleanup()

    def test_each_new_facet_changes_its_target_score(self):
        for name,(target,direction,_,_) in dynamics.FACETS.items():
            with self.subTest(facet=name):
                state=copy.deepcopy(self.agent['state'])
                state['personality']['dispositions'][name]=0
                low={action:0 for action in ACTIONS}
                dynamics.bias(low,state)
                state['personality']['dispositions'][name]=1
                high={action:0 for action in ACTIONS}
                dynamics.bias(high,state)
                self.assertGreater(direction*(high[target]-low[target]),0)

    def test_appraisal_distinguishes_responsibility_and_fairness(self):
        own=copy.deepcopy(self.agent['state']);external=copy.deepcopy(own)
        dynamics.appraise(own,goal_congruence=-1,agency=1,fairness=0)
        dynamics.appraise(external,goal_congruence=-1,agency=0,fairness=0)
        self.assertGreater(own['affect']['shame'],external['affect']['shame'])
        self.assertGreater(external['affect']['anger'],own['affect']['anger'])

    def test_invalid_appraisal_rolls_back(self):
        aid=self.agent['id'];before=self.service.get(aid)['state']
        with self.assertRaises(ValueError):
            self.service.cognition_operation(aid,'appraise',goal_congruence=1,agency=float('nan'))
        self.assertEqual(self.service.get(aid)['state'],before)

    def test_v4_migration_preserves_configured_values(self):
        aid=self.agent['id'];s=self.agent['state']
        s['personality']['schema_version']=4
        s['personality']['dispositions']['altruism']=0.91
        for name in dynamics.FACETS:
            del s['personality']['dispositions'][name]
            del s['personality']['baseline'][name]
        with self.service.store.connection() as db:
            db.execute('UPDATE agents SET state=? WHERE id=?',(json.dumps(s),aid))
        migrated=Service(self.path).get(aid)
        self.assertEqual(migrated['state']['personality']['dispositions']['altruism'],0.91)
        self.assertEqual(len(migrated['state']['personality']['dispositions']),36)
        self.assertEqual(migrated['genome_hash'],self.agent['genome_hash'])

    def test_forgiveness_changes_partner_selection_and_resentment(self):
        state=self.agent['state']
        state['social_dynamics']['partner']={'closeness':0.2,'resentment':0.8,'gratitude':0,'interactions':1}
        relation={'other_id':'partner','trust':0.4}
        state['personality']['dispositions']['forgiveness']=0
        low=dynamics.partner_weight(state,relation)
        state['personality']['dispositions']['forgiveness']=1
        self.assertGreater(dynamics.partner_weight(state,relation),low)
        before=state['social_dynamics']['partner']['resentment']
        dynamics.update(state,self.agent['traits'],{'action':'socialize','reward':0.5,'partner_id':'partner','trust_delta':0.1,'text':'Observed cooperation'},1,1)
        self.assertLess(state['social_dynamics']['partner']['resentment'],before)

    def test_learning_progress_tracks_decreasing_prediction_error(self):
        s=self.agent['state'];outcome={'action':'practice','reward':0.1,'text':'Practice'}
        s['prediction_error']=0.8
        dynamics.update(s,self.agent['traits'],outcome,1,1)
        s['prediction_error']=0.2
        dynamics.update(s,self.agent['traits'],outcome,2,2)
        self.assertGreater(s['motivation']['action_progress']['practice'],0)

    def test_planning_forms_bounded_intention_and_nudges_choice(self):
        s=self.agent['state']
        dynamics.update(s,self.agent['traits'],{'action':'plan','reward':0.03,'text':'Planned'},1,1)
        self.assertEqual(s['intentions']['action'],'practice')
        self.assertTrue(1<=s['intentions']['remaining']<=4)
        committed={a:0 for a in ACTIONS};dynamics.bias(committed,s)
        s['intentions']['remaining']=0
        uncommitted={a:0 for a in ACTIONS};dynamics.bias(uncommitted,s)
        self.assertGreater(committed['practice'],uncommitted['practice'])

    def test_creative_artifacts_are_bounded_with_unique_indexes(self):
        s=self.agent['state'];s['resources']=20
        for i in range(40):
            s,_=transition(self.agent['traits'],s,{'action':'create','skill':'craft','partner_id':None},random.Random(i))
        self.assertEqual(len(s['artifacts']),32)
        self.assertEqual(s['creative_output_count'],40)
        self.assertEqual(len({a['index'] for a in s['artifacts']}),32)

    def test_mood_congruent_recall_keeps_original_evidence(self):
        memories=[{'id':1,'text':'same event','tick':0,'salience':0.5,'reward':1},
                  {'id':2,'text':'same event','tick':0,'salience':0.5,'reward':-1}]
        state=self.agent['state'];state['affect']['valence']=1
        positive=retrieve(memories,'same',self.agent['traits'],0,state=state)
        state['affect']['valence']=-1
        negative=retrieve(memories,'same',self.agent['traits'],0,state=state)
        self.assertEqual(positive[0]['id'],1);self.assertEqual(negative[0]['id'],2)
        self.assertEqual(positive[0]['text'],memories[0]['text'])

    def test_overload_reduces_workspace_capacity(self):
        rested=copy.deepcopy(self.agent['state']);tired=copy.deepcopy(rested)
        rested.update(energy=1,stress=0);rested['body']['sleep_pressure']=0
        tired.update(energy=0,stress=1);tired['body']['sleep_pressure']=1
        tired['affect']['arousal']=1
        for i in range(20):
            for s in (rested,tired):mind.attend(s,{'tick':i,'priority':1,'text':str(i)})
        self.assertLess(len(tired['workspace']),len(rested['workspace']))

    def test_missing_partners_and_resources_mask_new_actions(self):
        s=self.agent['state'];s['resources']=0
        for seed in range(100):
            decision=decide(self.agent['traits'],s,[],[],random.Random(seed))
            self.assertNotIn(decision['action'],('repair','create','socialize','help','eat','drink'))

    def test_simulation_persists_bounded_milestones_and_emotions(self):
        self.service.create('Peer',4096,7)
        self.service.tick(steps=100,seed=9)
        s=Service(self.path).get(self.agent['id'])['state']
        self.assertLessEqual(len(s['self_narrative']['milestones']),32)
        for name in dynamics.EMOTIONS:self.assertTrue(0<=s['affect'][name]<=1)
        self.assertTrue(s['motivation']['action_progress'])
