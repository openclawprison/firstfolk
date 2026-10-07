"""Persistent artificial-life experiment. Engineered physiology, not human biology.

Policy consumes an observation DTO only. It never receives the world, genome,
organ implementation, admin events, or a language model's external knowledge.
"""
from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path
import random
from collections import Counter

import genome
import engine
import personality
import human_dynamics

ACTIONS = ('north', 'east', 'south', 'west', 'eat', 'drink', 'rest',
           'gather', 'place', 'signal', 'pair', 'care', 'rub')
DIRECTIONS = ((0, -1), (1, 0), (0, 1), (-1, 0))
ORGANS = ('heart', 'lungs', 'gut', 'liver', 'kidneys', 'immune', 'brain',
          'muscles', 'skin', 'reproductive')
TERRAINS = ('grass', 'water', 'wood', 'rock', 'cave')


def clamp(value, lo=0., hi=1.):
    return max(lo, min(hi, value))


def tuplify(value):
    return tuple(tuplify(x) for x in value) if isinstance(value, list) else value


def state_key(observation):
    """Coarse local context. No absolute coordinates or hidden world data."""
    body = observation['sensations']
    return '|'.join(map(str, [observation['underfoot'],
        *(int(body[k] * 3) for k in ('hunger', 'thirst', 'tiredness', 'cold', 'pain')),
        int(bool(observation['neighbors'])), min(2, observation['held_material']),
        int(observation['light'] < .3)]))


def choose(observation, learner, traits, dispositions, rng, learning=True):
    """Untrained Q values start at zero; all actions are available to try.

    Local directional generalization reuses learned food/water outcomes. It
    searches visible offsets, never a hidden map. These inductive biases and
    the primitive action set are engineered, not evolved anatomy/intelligence.
    """
    key = state_key(observation)
    values = learner['q'].get(key, [0.] * len(ACTIONS))
    scores = list(values)
    senses = observation['sensations']
    predictions = learner['transition_model']
    for i, action in enumerate(ACTIONS):
        model = predictions.get(action)
        if model and model['n'] >= 3:
            scores[i] += .12*traits['planning']*model['mean_reward']
    for i, (dx, dy) in enumerate(DIRECTIONS):
        sector = [t for t in observation['visible']
                  if t['dx'] * dx + t['dy'] * dy > 0]
        if sector:
            for modality, action, need in [('food', 'eat', 'hunger'), ('water', 'drink', 'thirst')]:
                learned = max(0., learner['affordances'].get(action, 0.))
                candidates = [t for t in sector if t[modality]]
                if candidates:
                    distance = min(abs(t['dx']) + abs(t['dy']) for t in candidates)
                    scores[i] += learned * senses[need] / max(1, distance)
    # Innate homeostatic drives are explicitly supplied. They are not learned
    # evidence or proof of human instincts. No explicit construction goal.
    instincts = {'eat': 1.3*senses['hunger'], 'drink': 1.5*senses['thirst'],
        'rest': .85*senses['tiredness']+.25*senses['pain'],
        'signal': .4*traits['sociability']*senses['loneliness']*bool(observation['neighbors']),
        'pair': .6*observation['mating_drive']*bool(observation['neighbors'])}
    food_here = observation['held_food'] > .01 or any(v['dx'] == v['dy'] == 0 and v['food'] for v in observation['visible'])
    water_here = any(abs(v['dx'])+abs(v['dy']) <= 1 and v['water'] for v in observation['visible'])
    if not food_here:
        instincts['eat'] = 0.
    if not water_here:
        instincts['drink'] = 0.
    # Innate sensory orientation towards visible nourishment. No hidden map.
    for i, (dx, dy) in enumerate(DIRECTIONS):
        for modality, need, present in [('food', 'hunger', food_here), ('water', 'thirst', water_here)]:
            targets = [v for v in observation['visible'] if v[modality]
                       and v['dx']*dx+v['dy']*dy > 0]
            if targets and not present:
                distance = min(abs(v['dx'])+abs(v['dy']) for v in targets)
                scores[i] += 1.5*senses[need]/max(1, distance)
    proxy = engine.initial_state(traits)
    proxy.update({'affect': {'fear': senses['pain'], 'frustration': senses['pain'], 'arousal': senses['pain']},
                  'body': {'sleep_pressure': senses['tiredness']}, 'prediction_error': 0.})
    personality.initialize(proxy, traits)
    proxy['personality']['dispositions'] = dispositions
    proxy['energy'] = 1-senses['hunger']
    proxy['stress'] = max(senses['pain'], senses['cold'])
    proxy['psychological_needs']['belonging'] = 1-senses['loneliness']
    proxy['psychological_needs']['safety'] = 1-senses['pain']
    abstract = dict.fromkeys(engine.ACTIONS, 0.)
    personality.bias(abstract, proxy)
    facets = human_dynamics.bias(abstract, proxy)
    mappings = {'rest': ('rest', 'sleep', 'reflect'), 'signal': ('socialize', 'repair'),
                'care': ('help',), 'gather': ('practice',), 'place': ('create', 'plan'), 'rub': ('practice', 'play')}
    for action, sources in mappings.items():
        scores[ACTIONS.index(action)] += .3*sum(abstract[s] for s in sources)
    for i in range(4):
        scores[i] += .15*(abstract['explore']+abstract['play'])
    dependent = any(n['small'] for n in observation['neighbors'])
    instincts['care'] = .35*traits['empathy']*dispositions['altruism']*dependent*(1-senses['hunger'])
    for action, bias in instincts.items():
        scores[ACTIONS.index(action)] += bias
    epsilon = .06 + .24 * traits['curiosity']
    epsilon += .15 * dispositions['impulsivity'] * senses['pain']
    epsilon += .1*(1-observation['cognitive_capacity'])
    explore = rng.random() < epsilon
    if explore:
        selected = rng.randrange(len(ACTIONS))
    else:
        best = max(scores)
        selected = rng.choice([i for i, x in enumerate(scores) if abs(x-best) < 1e-9])
    return selected, {'context': key, 'exploration': explore,
        'action_values': dict(zip(ACTIONS, map(lambda x: round(x, 5), scores))),
        'instinct_bias': instincts, 'facet_evidence': facets,
        'learned_predictions': copy.deepcopy(predictions),
        'source': 'innate_drives_personality_and_learned_local_outcomes', 'human_thought': False}


class World:
    VERSION = 7

    def __init__(self, seed=7, width=32, height=24, founders=4, learning=True):
        if not 12 <= width <= 64 or not 12 <= height <= 64 or founders not in (3, 4):
            raise ValueError('World size must be 12..64; founders must be 3 or 4')
        self.rng = random.Random(seed)
        self.learning = learning
        self.data = {'version': self.VERSION, 'seed': seed, 'width': width,
            'height': height, 'tick': 0, 'agents': [], 'cells': [], 'events': [],
            'next_id': 1, 'stats': {'births': 0, 'deaths': 0, 'conversations': 0,
            'agreements': 0, 'material_placements': 0, 'ignitions': 0, 'recoveries': 0}, 'learning': learning}
        for y in range(height):
            for x in range(width):
                r = self.rng.random()
                terrain = 'water' if x == width//2 or r < .035 else (
                    'wood' if r < .24 else 'rock' if r < .31 else 'cave' if r < .335 else 'grass')
                self.data['cells'].append({'terrain': terrain,
                    'food': round(self.rng.uniform(.5, 1.), 3) if terrain in ('grass', 'wood') else 0.,
                    'material': 5 if terrain in ('wood', 'rock') else 0,
                    'shelter': .85 if terrain == 'cave' else .3 if terrain == 'wood' else 0.,
                    'objects': [], 'fire': 0.})
        for i in range(founders):
            x, y = width//2 - 1 + i % 2, height//2 + i//2
            self._birth(genome.founder(4096, seed*100+i), x, y, [], 0, adult=True)
            self.data['agents'][-1]['sex'] = 'A' if i % 2 == 0 else 'B'

    def event(self, kind, **fields):
        self.data['events'].append({'tick': self.data['tick'], 'kind': kind, **fields})
        self.data['events'] = self.data['events'][-1000:]

    def _birth(self, dna, x, y, parents, generation, adult=False):
        traits = genome.express(dna)
        initial = engine.initial_state(traits)
        initial['affect'] = {'fear': 0., 'frustration': 0.}
        personality.initialize(initial, traits)
        identifier = self.data['next_id']
        self.data['next_id'] += 1
        a = {'id': identifier, 'name': f'Individual {identifier}', 'x': x, 'y': y,
             'dna': dna.hex(), 'traits': traits,
             'dispositions': initial['personality']['dispositions'],
             'sex': 'A' if self.rng.random() < .5 else 'B',
             'parents': parents, 'generation': generation, 'born': self.data['tick'],
             'age': 600 if adult else 0, 'alive': True, 'death': None,
             'energy': .85, 'hydration': .9, 'sleep': .1, 'temperature': .5,
             'health': 1., 'social': .25, 'organs': {k: 1. for k in ORGANS},
             'oxygen': 1., 'infection': None, 'immunity': [], 'gestation': None,
             'cooldown': 0, 'inventory': 0, 'held_materials': [], 'food_store': 0., 'lexicon': {},
             'mating_drive': .45 if adult else 0.,
             'bonds': {}, 'learner': {'q': {}, 'updates': 0, 'affordances': {},
               'actions': {}, 'visited': [], 'transition_model': {}},
             'memories': [], 'last_action': None, 'decision': None, 'last_signal': None,
             'reward_total': 0., 'lifespan': int(6500 + traits['recovery']*3500)}
        self.data['agents'].append(a)
        if parents:
            self.data['stats']['births'] += 1
        self.event('birth', agent=identifier, parents=parents, generation=generation,
                   inherited_memories=False, inherited_policy=False)
        return a

    def cell(self, x, y):
        return self.data['cells'][y*self.data['width']+x]

    def living(self):
        return [a for a in self.data['agents'] if a['alive']]

    def sky(self):
        tick = self.data['tick']
        phase = tick % 240 / 240
        light = clamp(math.sin(2*math.pi*phase) * .65 + .35)
        rain = math.sin(tick/113 + self.data['seed']) > .65
        season = .1 * math.sin(tick/1800)
        return {'phase': phase, 'light': light, 'day': tick//240,
                'rain': rain, 'clouds': .8 if rain else .25,
                'temperature': .32 + .23*light + season}

    def observe(self, a):
        sky = self.sky()
        radius = 3 if sky['light'] > .2 else 1
        visible = []
        for dy in range(-radius, radius+1):
            for dx in range(-radius, radius+1):
                x, y = a['x']+dx, a['y']+dy
                if 0 <= x < self.data['width'] and 0 <= y < self.data['height']:
                    c = self.cell(x, y)
                    visible.append({'dx': dx, 'dy': dy, 'terrain': TERRAINS.index(c['terrain']),
                        'food': c['food'] > .08, 'water': c['terrain'] == 'water',
                        'cover': c['shelter'] > .25, 'material': c['material'] > 0,
                        'hot': c['fire'] > .01, 'objects': len(c['objects'])})
        neighbors = [{'id': b['id'], 'dx': b['x']-a['x'], 'dy': b['y']-a['y'],
                      'small': b['age'] < 480, 'signal': b['last_signal']}
                     for b in self.living() if b['id'] != a['id']
                     and abs(b['x']-a['x']) + abs(b['y']-a['y']) <= radius]
        return {'visible': visible, 'underfoot': TERRAINS.index(self.cell(a['x'], a['y'])['terrain']),
                'neighbors': neighbors, 'light': sky['light'], 'wet': sky['rain'],
                'held_material': a['inventory'], 'held_food': a['food_store'],
                'mating_drive': a['mating_drive'] if a['age'] >= 480 and not a['gestation'] else 0.,
                'cognitive_capacity': a['organs']['brain']*a['oxygen']*(1-.5*a['sleep']),
                'sensations': {'hunger': 1-a['energy'], 'thirst': 1-a['hydration'],
                    'tiredness': a['sleep'], 'cold': clamp((.48-a['temperature'])*3),
                    'pain': 1-a['health'], 'breathlessness': 1-a['oxygen'],
                    'loneliness': a['social']}}

    def utility(self, a):
        t = a['traits']
        return (2*a['energy'] + 2.5*a['hydration'] + 3*a['health']
                + .7*(1-a['sleep']) + .6*(1-abs(a['temperature']-.5)*2)
                + (.3+.6*t['sociability'])*(1-a['social']))

    def _death(self, a, cause):
        if not a['alive']:
            return
        a['alive'] = False
        a['death'] = {'tick': self.data['tick'], 'cause': cause}
        a['gestation'] = None
        self.data['stats']['deaths'] += 1
        self.event('death', agent=a['id'], cause=cause, generation=a['generation'])

    def _body(self, a):
        a['age'] += 1
        a['cooldown'] = max(0, a['cooldown']-1)
        o, t = a['organs'], a['traits']
        infection = a['infection']
        burden = infection['load'] if infection else 0.
        exertion = .002 if a['last_action'] in ACTIONS[:4] else 0.
        a['energy'] = clamp(a['energy']-.0025-exertion-.004*burden)
        a['hydration'] = clamp(a['hydration']-.0025-.003*burden)
        a['sleep'] = clamp(a['sleep']+.0025)
        a['social'] = clamp(a['social']+.0015)
        if a['age'] >= 480 and not a['gestation'] and a['cooldown'] == 0:
            a['mating_drive'] = clamp(a['mating_drive']+.003*(1-a['sleep'])*a['organs']['reproductive'])
        shelter = self.cell(a['x'], a['y'])['shelter']
        sky = self.sky()
        ambient = sky['temperature'] - (.08 if sky['rain'] else 0.)
        a['temperature'] += .025*((ambient*(1-shelter)+.5*shelter)-a['temperature'])
        a['temperature'] = clamp(a['temperature']+.002*burden)
        a['temperature'] = clamp(a['temperature']+.035*self.cell(a['x'], a['y'])['fire'])
        overheating = max(0., a['temperature']-.7)
        age_load = max(0., (a['age']-a['lifespan']*.7)/(a['lifespan']*.3))
        deprivation = max(0., .12-a['energy'])+max(0., .12-a['hydration'])
        cold = max(0., .22-a['temperature'])
        for organ in ORGANS:
            damage = (.018*deprivation + .002*age_load + .01*cold + .03*overheating)
            if organ in ('lungs', 'heart', 'brain'):
                damage += .0015*burden
            recovery = .0006*t['recovery'] if a['energy'] > .3 and a['hydration'] > .3 else 0.
            o[organ] = clamp(o[organ]+recovery-damage)
        # Oxygen delivery couples respiratory, cardiac and nutritional status.
        a['oxygen'] = clamp(o['lungs']*o['heart']*(.5+.5*a['hydration'])*(1-.4*burden))
        o['brain'] = clamp(o['brain']-.004*max(0., .3-a['oxygen']))
        o['kidneys'] = clamp(o['kidneys']-.008*max(0., .15-a['hydration']))
        a['health'] = min(o['heart'], o['lungs'], o['brain'], sum(o.values())/len(o))
        if infection:
            infection['age'] += 1
            defense = (.035+.035*t['recovery'])*o['immune']*(.5+.5*a['energy'])
            if infection['strain'] in a['immunity']:
                defense += .09
            infection['load'] = clamp(burden + .027 - defense, 0., 1.)
            if infection['load'] <= .01:
                if infection['strain'] not in a['immunity']:
                    a['immunity'].append(infection['strain'])
                a['infection'] = None
                self.data['stats']['recoveries'] += 1
                self.event('recovery', agent=a['id'])
        if a['health'] <= .035 or a['oxygen'] <= .025:
            self._death(a, 'infection_and_organ_failure' if infection else
                        'deprivation_or_organ_failure')
        elif a['age'] >= a['lifespan']:
            self._death(a, 'aging')
        if a['alive'] and a['gestation']:
            a['gestation']['remaining'] -= 1
            a['energy'] = clamp(a['energy']-.001)
            if a['gestation']['remaining'] <= 0:
                pregnancy = a['gestation']
                a['gestation'] = None
                if len(self.living()) < 64 and len(self.data['agents']) < 512:
                    child = self._birth(bytes.fromhex(pregnancy['dna']), a['x'], a['y'],
                         pregnancy['parents'], pregnancy['generation'])
                    child['mutation_count'] = pregnancy['mutations']
                    a['energy'] = clamp(a['energy']-.12)

    def nearby(self, a, radius=1):
        return [b for b in self.living() if b['id'] != a['id']
                and abs(b['x']-a['x'])+abs(b['y']-a['y']) <= radius]

    def _concept(self, a):
        c = self.cell(a['x'], a['y'])
        # Joint attention to perceptible categories, not a shared vocabulary.
        if c['food'] > .08:
            return 'edible'
        if c['terrain'] == 'water':
            return 'liquid'
        if c['shelter'] > .3:
            return 'cover'
        return c['terrain']

    def _signal(self, a, b):
        concept = self._concept(a)
        inventory = a['lexicon'].setdefault(concept, [])
        if not inventory:
            inventory.append(''.join(self.rng.choice(('ba','ku','mi','ta','lo','si','na','po'))
                                     for _ in range(3)))
        token = self.rng.choice(inventory)
        a['last_signal'] = {'token': token, 'tick': self.data['tick']}
        # Hearer must independently perceive the referent in its own local view.
        visible = self.observe(b)['visible']
        dx, dy = a['x']-b['x'], a['y']-b['y']
        if not any(v['dx'] == dx and v['dy'] == dy for v in visible):
            return
        other = b['lexicon'].setdefault(concept, [])
        success = token in other
        if self.learning:
            if success:
                a['lexicon'][concept] = [token]
                b['lexicon'][concept] = [token]
            elif token not in other:
                other.append(token)
                del other[:-32]
        a['social'] = clamp(a['social']-.15)
        b['social'] = clamp(b['social']-.1)
        self.data['stats']['conversations'] += 1
        self.data['stats']['agreements'] += int(success)
        for p, q in ((a, b), (b, a)):
            p['bonds'][str(q['id'])] = clamp(p['bonds'].get(str(q['id']), .3)+.04)
        self.event('communication', speaker=a['id'], hearer=b['id'], token=token,
                   referent=concept, understood=success)

    def _act(self, a, index, intentions):
        action = ACTIONS[index]
        a['last_action'] = action
        c = self.cell(a['x'], a['y'])
        peers = self.nearby(a)
        if index < 4:
            dx, dy = DIRECTIONS[index]
            if a['oxygen'] > .12 and a['organs']['muscles'] > .08:
                a['x'] = max(0, min(self.data['width']-1, a['x']+dx))
                a['y'] = max(0, min(self.data['height']-1, a['y']+dy))
        elif action == 'eat':
            amount = min(.25, a['food_store']+c['food'])
            stored = min(a['food_store'], amount)
            a['food_store'] -= stored
            c['food'] -= amount-stored
            a['energy'] = clamp(a['energy']+amount*a['organs']['gut']*a['organs']['liver'])
        elif action == 'drink':
            if any(0 <= a['x']+dx < self.data['width'] and 0 <= a['y']+dy < self.data['height']
                   and self.cell(a['x']+dx, a['y']+dy)['terrain'] == 'water'
                   for dx, dy in ((0,0), *DIRECTIONS)):
                a['hydration'] = clamp(a['hydration']+.3*a['organs']['kidneys'])
        elif action == 'rest':
            a['sleep'] = clamp(a['sleep']-.09)
            a['energy'] = clamp(a['energy']+.0015)
        elif action == 'gather':
            if c['material'] > 0 and a['inventory'] < 12:
                c['material'] -= 1
                a['inventory'] += 1
                a['held_materials'].append('wood' if c['terrain'] == 'wood' else 'stone')
            if c['food'] > .05 and a['food_store'] < 1.:
                taken = min(.1, c['food'])
                c['food'] -= taken
                a['food_store'] += taken
        elif action == 'place' and a['held_materials'] and len(c['objects']) < 20:
            kind = a['held_materials'].pop()
            a['inventory'] -= 1
            c['objects'].append({'material': kind, 'temperature': self.sky()['temperature'],
                                 'integrity': 1., 'burning': False})
            self.data['stats']['material_placements'] += 1
            self.event('material_placed', agent=a['id'], x=a['x'], y=a['y'], material=kind,
                       explicit_goal=False)
        elif action == 'rub' and c['objects']:
            target = self.rng.choice(c['objects'])
            target['temperature'] = min(1.5, target['temperature']+.16*a['organs']['muscles'])
            a['energy'] = clamp(a['energy']-.006)
        elif action == 'signal' and peers:
            self._signal(a, self.rng.choice(peers))
        elif action == 'care' and peers and a['energy'] > .2:
            recipient = min(peers, key=lambda b: b['energy'])
            amount = min(.06, a['energy']-.2)
            a['energy'] -= amount
            recipient['energy'] = clamp(recipient['energy']+amount)
            a['social'] = clamp(a['social']-.08*a['traits']['empathy'])
            a['bonds'][str(recipient['id'])] = clamp(a['bonds'].get(str(recipient['id']), .3)+.03)
        elif action == 'pair' and a['age'] >= 480 and a['cooldown'] == 0:
            candidates = [b for b in peers if b['sex'] != a['sex'] and b['age'] >= 480
                and b['cooldown'] == 0 and intentions.get(b['id']) == 'pair'
                and b['energy'] > .4 and b['hydration'] > .4]
            if candidates and a['energy'] > .4 and a['hydration'] > .4:
                b = self.rng.choice(candidates)
                carrier = a if a['sex'] == 'B' else b
                if not carrier['gestation'] and len(self.living()) < 64 and len(self.data['agents']) < 512:
                    dna, heredity = genome.reproduce(bytes.fromhex(a['dna']), bytes.fromhex(b['dna']),
                         self.rng.randrange(2**31), mutation_rate=.001)
                    carrier['gestation'] = {'remaining': 120, 'dna': dna.hex(),
                        'parents': [a['id'], b['id']], 'mutations': heredity['mutations'],
                        'generation': max(a['generation'], b['generation'])+1}
                    a['cooldown'] = b['cooldown'] = 600
                    for p in (a, b):
                        p['social'] = clamp(p['social']-.4)
                        p['mating_drive'] = 0.
                    self.event('conception', parents=[a['id'], b['id']], carrier=carrier['id'])

    def _transmission(self):
        living = self.living()
        infected = [a for a in living if a['infection']]
        for a in infected:
            for b in self.nearby(a, 2):
                if not b['infection'] and self.rng.random() < .018*a['infection']['load']:
                    strain = a['infection']['strain']
                    b['infection'] = {'strain': strain, 'load': .2, 'age': 0}
                    self.event('infection', agent=b['id'], source=a['id'], strain=strain)
        if living and self.data['tick'] % 800 == 0:
            a = self.rng.choice(living)
            if not a['infection']:
                strain = 'strain-'+str(self.data['tick']//2400 % 3)
                a['infection'] = {'strain': strain, 'load': .4, 'age': 0}
                self.event('infection', agent=a['id'], source='environment', strain=strain)

    def _physics(self):
        """Toy material laws. No fire recipe, construction goal or policy reward.

        Placement supplies physical cover; friction supplies heat; dry combustible
        matter above an ignition threshold releases heat and is consumed.
        These scalar approximations are not a validated combustion simulator.
        """
        sky = self.sky()
        for i, c in enumerate(self.data['cells']):
            c['fire'] = 0.
            base_cover = .85 if c['terrain'] == 'cave' else .3 if c['terrain'] == 'wood' else 0.
            for obj in c['objects']:
                if obj['material'] == 'wood' and obj['temperature'] >= .8 and not sky['rain']:
                    if not obj['burning']:
                        self.data['stats']['ignitions'] += 1
                        self.event('ignition', x=i%self.data['width'], y=i//self.data['width'],
                                   source='material_temperature', explicit_goal=False)
                    obj['burning'] = True
                if obj['burning']:
                    if sky['rain']:
                        obj['burning'] = False
                    else:
                        c['fire'] = min(1., c['fire']+.2)
                        obj['temperature'] = 1.
                        obj['integrity'] -= .01
                if not obj['burning']:
                    obj['temperature'] += .03*(sky['temperature']-obj['temperature'])
            c['objects'] = [o for o in c['objects'] if o['integrity'] > 0.]
            c['shelter'] = min(.85, base_cover+sum(.055*o['integrity'] for o in c['objects']))

    def step(self, steps=1):
        if type(steps) is not int or not 1 <= steps <= 10000:
            raise ValueError('steps must be an integer in 1..10000')
        for _ in range(steps):
            self.data['tick'] += 1
            living = self.living()
            self.rng.shuffle(living)
            prepared = []
            for a in living:
                obs = self.observe(a)
                index, evidence = choose(obs, a['learner'], a['traits'], a['dispositions'], self.rng, self.learning)
                prepared.append((a, obs, index, evidence, self.utility(a)))
            intentions = {a['id']: ACTIONS[i] for a, _, i, _, _ in prepared}
            for a, obs, index, evidence, before in prepared:
                self._act(a, index, intentions)
                self._body(a)
                learner = a['learner']
                location = f"{a['x']},{a['y']}"
                novel = location not in learner['visited']
                if novel:
                    learner['visited'].append(location)
                reward = self.utility(a)-before + .018*a['traits']['curiosity']*novel
                if not a['alive']:
                    reward -= 5.
                if self.learning:
                    key = state_key(obs)
                    if key not in learner['q'] and len(learner['q']) >= 2048:
                        del learner['q'][next(iter(learner['q']))]
                    row = learner['q'].setdefault(key, [0.] * len(ACTIONS))
                    future = max(learner['q'].get(state_key(self.observe(a)), [0.] * len(ACTIONS))) if a['alive'] else 0.
                    alpha = (.08+.27*a['traits']['plasticity'])*obs['cognitive_capacity']
                    discount = .82+.15*a['traits']['planning']*(1-.5*a['dispositions']['delay_discounting'])
                    row[index] += alpha*(reward+discount*future-row[index])
                    action = ACTIONS[index]
                    learner['affordances'][action] = learner['affordances'].get(action, 0.)*.95+.05*reward
                    learner['updates'] += 1
                    # Empirical action-effect model, not a scripted prediction.
                    model = learner['transition_model'].setdefault(action, {'n': 0, 'mean_reward': 0.})
                    model['n'] += 1
                    model['mean_reward'] += (reward-model['mean_reward'])/model['n']
                learner['actions'][ACTIONS[index]] = learner['actions'].get(ACTIONS[index], 0)+1
                a['reward_total'] += reward
                a['decision'] = {**evidence, 'selected': ACTIONS[index],
                    'reward': round(reward, 5), 'sensations': obs['sensations']}
                a['memories'].append({'tick': self.data['tick'], 'action': ACTIONS[index],
                    'reward': round(reward, 5), 'observed': obs['underfoot']})
                a['memories'] = a['memories'][-80:]
            self._transmission()
            self._physics()
            if self.data['tick'] % 12 == 0:
                for c in self.data['cells']:
                    if c['terrain'] in ('grass', 'wood'):
                        c['food'] = min(1., c['food']+.03)
                    if c['terrain'] in ('wood', 'rock') and self.data['tick'] % 120 == 0:
                        c['material'] = min(5, c['material']+1)
        return self.snapshot()

    def snapshot(self):
        agents = [{k: copy.deepcopy(v) for k, v in a.items() if k not in ('dna', 'learner')}
                  for a in self.data['agents']]
        for target, source in zip(agents, self.data['agents']):
            target['learning'] = {'updates': source['learner']['updates'],
                'contexts': len(source['learner']['q']), 'visited': len(source['learner']['visited']),
                'actions': source['learner']['actions'],
                'action_effects': source['learner']['transition_model']}
        return {'tick': self.data['tick'], 'seed': self.data['seed'],
            'width': self.data['width'], 'height': self.data['height'], 'sky': self.sky(),
            'agents': agents, 'cells': copy.deepcopy(self.data['cells']),
            'events': copy.deepcopy(self.data['events'][-100:]),
            'stats': {**self.data['stats'], 'alive': len(self.living()),
                      'max_generation': max(a['generation'] for a in self.data['agents'])},
            'claims': {'human_iq_measured': False, 'consciousness_established': False,
                'language': 'naming_game_shared_tokens_not_human_grammar',
                'physiology': 'engineered_coupled_organ_proxies',
                'origin_information_supplied': False, 'learning_enabled': self.learning}}

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {**self.data, 'rng_state': self.rng.getstate()}
        temporary = path.with_suffix(path.suffix+'.tmp')
        temporary.write_text(json.dumps(payload, allow_nan=False), encoding='utf-8')
        os.replace(temporary, path)

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding='utf-8'))
        if data.get('version') != cls.VERSION:
            raise ValueError('Unsupported world save version')
        result = cls.__new__(cls)
        result.data = data
        result.learning = data['learning']
        result.rng = random.Random()
        result.rng.setstate(tuplify(data.pop('rng_state')))
        return result
