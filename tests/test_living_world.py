import copy
import json
from pathlib import Path
import random
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from urllib.request import Request, urlopen
from urllib.error import HTTPError

from living_world import World, ACTIONS, choose
from world_server import Simulation, handler


class LivingWorldTests(unittest.TestCase):
    def setUp(self):
        self.w = World(seed=7, width=12, height=12)

    def test_founders_have_inherited_variation_and_no_experience(self):
        agents = self.w.living()
        self.assertEqual(len(agents), 4)
        self.assertEqual(len({a['dna'] for a in agents}), 4)
        self.assertEqual(len(agents[0]['traits'])+len(agents[0]['dispositions']), 52)
        self.assertTrue(all(not a['memories'] and not a['learner']['q'] and not a['lexicon'] for a in agents))

    def test_agent_observation_excludes_administrator_and_internal_information(self):
        a = self.w.living()[0]
        obs = self.w.observe(a)
        encoded = json.dumps(obs).lower()
        for field in ('dna', 'organs', 'source_code', 'events', 'artificial', 'ai', 'parents'):
            self.assertNotIn('"'+field+'"', encoded)
        self.assertNotIn('x', obs)
        self.assertNotIn('y', obs)
        self.assertTrue(all(abs(v['dx']) <= 3 for v in obs['visible']))
        saved = copy.deepcopy(obs)
        choose(obs, a['learner'], a['traits'], a['dispositions'], random.Random(1))
        self.assertEqual(obs, saved)

    def test_innate_needs_have_causal_action_bias(self):
        a = self.w.living()[0]
        a['energy'] = .1
        self.w.cell(a['x'], a['y'])['food'] = 1.
        obs = self.w.observe(a)
        _, decision = choose(obs, a['learner'], a['traits'], a['dispositions'], random.Random(3))
        self.assertGreater(decision['instinct_bias']['eat'], 1.)
        obs['sensations']['hunger'] = 0.
        _, fed = choose(obs, a['learner'], a['traits'], a['dispositions'], random.Random(3))
        self.assertEqual(fed['instinct_bias']['eat'], 0.)

    def test_real_learning_changes_values_and_memories(self):
        self.w.step(30)
        a = self.w.living()[0]
        self.assertEqual(a['learner']['updates'], 30)
        self.assertTrue(any(abs(v) > 1e-7 for row in a['learner']['q'].values() for v in row))
        self.assertEqual(len(a['memories']), 30)
        self.assertFalse(a['decision']['human_thought'])

    def test_frozen_control_does_not_learn(self):
        self.w.learning = False
        self.w.data['learning'] = False
        self.w.step(30)
        self.assertTrue(all(not a['learner']['q'] and a['learner']['updates'] == 0 for a in self.w.living()))

    def test_save_restore_continuation_is_exact(self):
        self.w.step(20)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'save.json'
            self.w.save(path)
            loaded = World.load(path)
            self.w.step(35)
            loaded.step(35)
            self.assertEqual(self.w.data, loaded.data)
            self.assertEqual(self.w.rng.getstate(), loaded.rng.getstate())

    def test_naming_game_learns_shared_tokens_without_given_words(self):
        a, b = self.w.living()[:2]
        b['x'], b['y'] = a['x'], a['y']
        self.w._signal(a, b)
        self.assertTrue(a['lexicon'])
        self.assertEqual(a['lexicon'], b['lexicon'])
        self.w._signal(a, b)
        self.assertEqual(self.w.data['stats']['agreements'], 1)

    def test_birth_is_recombined_and_has_no_parent_policy_or_memories(self):
        a, b = self.w.living()[:2]
        b['x'], b['y'] = a['x'], a['y']
        a['memories'].append({'text': 'parent experience'})
        a['learner']['q']['test'] = [1.]*len(ACTIONS)
        intentions = {a['id']: 'pair', b['id']: 'pair'}
        self.w._act(a, ACTIONS.index('pair'), intentions)
        carrier = b if b['sex'] == 'B' else a
        self.assertIsNotNone(carrier['gestation'])
        carrier['gestation']['remaining'] = 1
        self.w._body(carrier)
        child = self.w.data['agents'][-1]
        self.assertEqual(child['generation'], 1)
        self.assertEqual(set(child['parents']), {a['id'], b['id']})
        self.assertNotIn(child['dna'], (a['dna'], b['dna']))
        self.assertFalse(child['learner']['q'])
        self.assertFalse(child['memories'])
        self.assertFalse(child['lexicon'])

    def test_pairing_requires_both_local_intentions_and_maturity(self):
        a, b = self.w.living()[:2]
        b['x'], b['y'] = a['x'], a['y']
        self.w._act(a, ACTIONS.index('pair'), {b['id']: 'rest'})
        self.assertFalse(a['gestation'] or b['gestation'])
        b['age'] = 1
        self.w._act(a, ACTIONS.index('pair'), {b['id']: 'pair'})
        self.assertFalse(a['gestation'] or b['gestation'])

    def test_generic_placement_changes_cover_without_a_construction_recipe(self):
        a = self.w.living()[0]
        c = self.w.cell(a['x'], a['y'])
        c['terrain'], c['shelter'] = 'grass', 0.
        self.w._act(a, ACTIONS.index('place'), {})
        self.assertEqual(c['shelter'], 0.)
        a['inventory'] = 1
        a['held_materials'] = ['wood']
        self.w._act(a, ACTIONS.index('place'), {})
        self.w._physics()
        self.assertGreater(c['shelter'], 0.)
        self.assertFalse(self.w.data['events'][-1]['explicit_goal'])

    def test_friction_can_ignite_material_without_a_fire_action(self):
        a = self.w.living()[0]
        c = self.w.cell(a['x'], a['y'])
        c['terrain'] = 'grass'
        self.w.data['tick'] = 60
        while self.w.sky()['rain']:
            self.w.data['tick'] += 1
        a['inventory'], a['held_materials'] = 1, ['wood']
        self.w._act(a, ACTIONS.index('place'), {})
        for _ in range(10):
            self.w._act(a, ACTIONS.index('rub'), {})
            self.w._physics()
        self.assertGreater(c['fire'], 0.)
        self.assertEqual(self.w.data['stats']['ignitions'], 1)
        self.assertNotIn('build_house', ACTIONS)
        self.assertNotIn('make_fire', ACTIONS)

    def test_parent_can_neglect_child_and_care_is_optional(self):
        parent = self.w.living()[0]
        child = self.w._birth(bytes.fromhex(parent['dna']), parent['x'], parent['y'], [parent['id']], 1)
        child['energy'] = .1
        self.w._act(parent, ACTIONS.index('rest'), {})
        self.assertEqual(child['energy'], .1)
        self.w._act(parent, ACTIONS.index('care'), {})
        self.assertGreater(child['energy'], .1)

    def test_organ_failure_causes_permanent_death(self):
        a = self.w.living()[0]
        a['infection'] = {'strain': 'test', 'load': 1., 'age': 0}
        a['organs']['heart'] = .01
        self.w._body(a)
        self.assertFalse(a['alive'])
        self.assertEqual(a['death']['cause'], 'infection_and_organ_failure')
        frozen = copy.deepcopy(a)
        self.w.step(20)
        self.assertEqual(a, frozen)

    def test_immunity_and_recovery_are_body_transitions(self):
        a = self.w.living()[0]
        a['infection'] = {'strain': 'test', 'load': .001, 'age': 10}
        self.w._body(a)
        self.assertIsNone(a['infection'])
        self.assertIn('test', a['immunity'])

    def test_day_night_and_weather_change_observations(self):
        self.w.data['tick'] = 60
        day = self.w.observe(self.w.living()[0])
        self.w.data['tick'] = 180
        night = self.w.observe(self.w.living()[0])
        self.assertGreater(day['light'], night['light'])
        self.assertGreater(len(day['visible']), len(night['visible']))

    def test_step_validation_is_non_mutating(self):
        before = copy.deepcopy(self.w.data)
        for value in (-1, 0, 10001, True, '2'):
            with self.assertRaises(ValueError):
                self.w.step(value)
        self.assertEqual(self.w.data, before)

    def test_http_observer_controls_and_origin_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            sim = Simulation(Path(directory)/'world.json', 7)
            server = ThreadingHTTPServer(('127.0.0.1', 0), handler(sim))
            worker = threading.Thread(target=server.serve_forever, daemon=True)
            worker.start()
            base = 'http://127.0.0.1:'+str(server.server_port)
            try:
                self.assertIn(b'Firstfolk', urlopen(base).read())
                request = Request(base+'/api/step', data=b'{"steps":2}', headers={'Content-Type':'application/json'})
                result = json.loads(urlopen(request).read())
                self.assertEqual(result['tick'], 2)
                request = Request(base+'/api/control', data=b'{"running":true}',
                                  headers={'Content-Type':'application/json', 'Origin':'https://example.org'})
                with self.assertRaises(HTTPError) as error:
                    urlopen(request)
                self.assertEqual(error.exception.code, 403)
                error.exception.close()
                self.assertFalse(sim.running)
            finally:
                server.shutdown()
                server.server_close()
                worker.join(2)


if __name__ == '__main__':
    unittest.main()
