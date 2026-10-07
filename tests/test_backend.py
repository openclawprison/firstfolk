import json
import random
import tempfile
import threading
import unittest
import urllib.request
import urllib.error
from pathlib import Path
from http.server import ThreadingHTTPServer

import genome
from engine import decide, initial_state
from service import Service
from server import handler


class BackendTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / 'test.sqlite3'
        self.service = Service(self.path)

    def tearDown(self):
        self.temp.cleanup()

    def create(self, name='A', seed=1):
        return self.service.create(name, loci=4096, seed=seed)

    def test_million_locus_genome(self):
        a = self.service.create('Large', loci=3_000_000, seed=7)
        self.assertEqual(a['loci'], 3_000_000)
        self.assertEqual(len(a['traits']), 16)
        self.assertTrue(all(0 < v < 1 for v in a['traits'].values()))

    def test_heredity_and_no_memory_inheritance(self):
        a, b = self.create(), self.create('B', 2)
        self.service.experience(a['id'], 'Unique parental experience', 0.8, 'explore')
        c = self.service.create('Child', parents=[a['id'], b['id']], seed=3)
        self.assertEqual(c['parents'], [a['id'], b['id']])
        self.assertEqual(c['loci'], 4096)
        self.assertEqual(self.service.memories(c['id']), [])
        self.assertNotEqual(c['genome_hash'], a['genome_hash'])
        with self.service.store.connection() as db:
            aa = self.service.agent(db, a['id'])['genome']
            bb = self.service.agent(db, b['id'])['genome']
        child, _ = genome.reproduce(aa, bb, 3, 0)
        for i in range(4096):
            self.assertIn(child[i], (aa[i], aa[4096 + i]))
            self.assertIn(child[4096 + i], (bb[i], bb[4096 + i]))

    def test_tick_persistence_and_bounds(self):
        a, b = self.create(), self.create('B', 2)
        result = self.service.tick(steps=30, seed=6)
        self.assertEqual(result['tick'], 30)
        self.assertEqual(len(result['events']), 60)
        fresh = Service(self.path)
        state = fresh.get(a['id'])['state']
        self.assertEqual(state['age_ticks'], 30)
        for key in ('energy', 'stress', 'confidence', 'social_need', 'novelty_need'):
            self.assertTrue(0 <= state[key] <= 1)
        self.assertTrue(fresh.events(a['id']))
        self.assertTrue(fresh.memories(a['id']))
        self.assertTrue(fresh.relationships(b['id']))

    def test_experience_changes_learning_and_decisions(self):
        a = self.create()
        for _ in range(10):
            self.service.experience(a['id'], 'explore discovered resources', 1, 'explore')
        learned = self.service.get(a['id'])['state']['learned_values']['explore']
        self.assertGreater(learned, 0.5)
        memories = self.service.memories(a['id'], 'explore')
        d = decide(a['traits'], self.service.get(a['id'])['state'], memories, [], random.Random(1))
        self.assertTrue(d['memory_evidence'])

    def test_genome_trait_has_causal_effect(self):
        base = {name: 0.5 for name in genome.TRAITS}
        state = initial_state(base)
        low = decide(base | {'curiosity': 0}, state, [], [], random.Random(1))
        high = decide(base | {'curiosity': 1}, state, [], [], random.Random(1))
        self.assertGreater(high['scores']['explore'], low['scores']['explore'])

    def test_invalid_parent_transaction_rolls_back(self):
        a = self.create()
        with self.assertRaises(LookupError):
            self.service.create('Bad', parents=[a['id'], 'missing'])
        self.assertEqual(len(self.service.list()), 1)

    def test_teaching(self):
        a, b = self.create(), self.create('B', 2)
        with self.service.store.connection() as db:
            teacher = self.service.agent(db, a['id'])
            teacher['state']['skills']['reasoning'] = 0.9
            db.execute('UPDATE agents SET state=? WHERE id=?', (json.dumps(teacher['state']), a['id']))
        result = self.service.teach(a['id'], b['id'], 'reasoning')
        self.assertGreater(result['after'], result['before'])
        self.assertEqual(self.service.memories(b['id'])[0]['source'], 'teaching')

    def test_http(self):
        server = ThreadingHTTPServer(('127.0.0.1', 0), handler(self.service))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f'http://127.0.0.1:{server.server_port}'
        try:
            request = urllib.request.Request(base + '/agents', json.dumps({'name': 'HTTP', 'loci': 64}).encode(), {'Content-Type': 'application/json'})
            with urllib.request.urlopen(request) as response:
                a = json.load(response)
            with urllib.request.urlopen(base + '/agents/' + a['id']) as response:
                self.assertEqual(json.load(response)['name'], 'HTTP')
            request = urllib.request.Request(base + '/simulation/tick', b'{"steps":0}', {'Content-Type': 'application/json'})
            with self.assertRaises(urllib.error.HTTPError) as error:
                urllib.request.urlopen(request)
            self.assertEqual(error.exception.code, 400)
            error.exception.close()
            with urllib.request.urlopen(base + '/health') as response:
                self.assertEqual(json.load(response)['status'], 'ok')
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == '__main__':
    unittest.main()
