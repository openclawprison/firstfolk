import copy
import json
import random
import tempfile
import unittest
from pathlib import Path
from service import Service
from engine import decide,effective_traits
from mind import forecast
import personality


class PersonalityTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.path=Path(self.tmp.name)/'db.sqlite'
        self.service=Service(self.path)
        self.agent=self.service.create('Synthetic participant',4096,12)

    def tearDown(self):
        self.tmp.cleanup()

    def test_counts_and_persistent_configuration(self):
        aid=self.agent['id']
        self.assertEqual(len(self.agent['traits']),16)
        self.assertEqual(len(self.agent['state']['personality']['dispositions']),36)
        self.service.cognition_operation(aid,'personality_configure',dispositions={'altruism':0.9},source='test')
        state=Service(self.path).get(aid)['state']
        self.assertEqual(state['personality']['dispositions']['altruism'],0.9)
        self.assertEqual(len(state['psychological_needs']),5)

    def test_inventory_distinguishes_traits_from_numeric_values(self):
        result=self.service.inventory(self.agent['id'])
        self.assertEqual(result['named_trait_controls'],52)
        self.assertEqual(result['stored_numeric_state_values'],len(result['numeric_state_paths']))
        self.assertGreater(result['stored_numeric_state_values'],28)
        self.assertFalse(result['human_fidelity_measured'])

    def test_altruism_and_belonging_change_action_scores(self):
        state=self.agent['state']
        low={name:0.0 for name in ('rest','reflect','explore','practice','help','socialize')}
        high=dict(low)
        state['personality']['dispositions']['altruism']=0
        state['psychological_needs']['belonging']=1
        personality.bias(low,state)
        state['personality']['dispositions']['altruism']=1
        state['psychological_needs']['belonging']=0
        personality.bias(high,state)
        self.assertGreater(high['help'],low['help'])
        self.assertGreater(high['socialize'],low['socialize'])

    def test_fatigue_reduces_planning_and_attention(self):
        rested=copy.deepcopy(self.agent['state'])
        rested.update(age_ticks=100,energy=1,stress=0)
        rested['body']['sleep_pressure']=0
        tired=copy.deepcopy(rested)
        tired.update(energy=0,stress=1)
        tired['body']['sleep_pressure']=1
        for trait in ('planning','attention'):
            self.assertGreater(effective_traits(self.agent['traits'],rested)[trait],effective_traits(self.agent['traits'],tired)[trait])

    def test_regulation_changes_sustained_emotion(self):
        low=copy.deepcopy(self.agent['state'])
        high=copy.deepcopy(low)
        for state,value in ((low,0),(high,1)):
            state['personality']['dispositions']['emotional_regulation']=value
            state['affect'].update(fear=0.7,frustration=0.8)
            personality.update(state,self.agent['traits'],{'action':'explore','reward':-0.5})
        self.assertLess(high['affect']['fear'],low['affect']['fear'])
        self.assertLess(high['affect']['frustration'],low['affect']['frustration'])

    def test_slow_development_preserves_genome(self):
        before=copy.deepcopy(self.agent)
        self.service.tick(steps=20,seed=7)
        after=self.service.get(before['id'])
        self.assertEqual(after['genome_hash'],before['genome_hash'])
        self.assertEqual(after['traits'],before['traits'])
        self.assertEqual(after['state']['personality']['updates'],20)
        self.assertNotEqual(after['state']['personality']['dispositions'],before['state']['personality']['dispositions'])
        for value in after['state']['psychological_needs'].values():self.assertTrue(0<=value<=1)

    def test_legacy_migration_preserves_individual(self):
        aid=self.agent['id']
        state=self.agent['state']
        for key in ('personality','psychological_needs','appraisal','behavior_repetition'):state.pop(key)
        with self.service.store.connection() as db:
            db.execute('UPDATE agents SET state=? WHERE id=?',(json.dumps(state),aid))
        after=Service(self.path).get(aid)
        self.assertEqual(after['genome_hash'],self.agent['genome_hash'])
        self.assertEqual(len(after['state']['personality']['dispositions']),36)

    def test_invalid_configuration_rolls_back(self):
        before=self.service.get(self.agent['id'])['state']
        with self.assertRaises(ValueError):
            self.service.cognition_operation(self.agent['id'],'personality_configure',dispositions={'altruism':float('nan')})
        self.assertEqual(before,self.service.get(self.agent['id'])['state'])

    def test_forecast_drink_cost_and_bounded_projection(self):
        state=self.agent['state']
        state['resources']=0.2
        state['energy']=0.95
        state['world_model']={'rest':{'samples':10,'reward_mean':0.1,'reward_m2':0,'effects':{'energy':0.5}},
                              'drink':{'samples':10,'reward_mean':0.8,'reward_m2':0,'effects':{'resources':-0.1}}}
        result=forecast(state,'rest',depth=3)
        self.assertEqual(result['plan'],['rest','drink','drink'])
        self.assertEqual(result['projected_state']['energy'],1)
        self.assertGreaterEqual(result['projected_state']['resources'],0)
