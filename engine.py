"""Causal state, memory, learning, relationships and inspectable action selection."""
import math
import copy
import personality
import human_dynamics
import skills as learned_skills
from mind import bias_scores, forecast

ACTIONS = ('rest', 'explore', 'practice', 'socialize', 'help', 'sleep', 'eat', 'drink', 'reflect',
           'create','play','plan','repair')
SKILLS = ('reasoning', 'communication', 'craft')


def clamp(x, low=0.0, high=1.0):
    return max(low, min(high, x))


def initial_state(traits):
    return {
        'age_ticks': 0, 'energy': 0.8, 'stress': 0.1, 'social_need': 0.3,
        'novelty_need': 0.4, 'confidence': 0.5, 'resources': 5.0,
        'skills': {k: 0.05 for k in SKILLS},
        'learned_values': {a: 0.0 for a in ACTIONS},
        'self_model': {'successful_actions': 0, 'failed_actions': 0},
    }


def effective_traits(base, state):
    # Gradual maturation of planning and attention, with lifelong plasticity.
    maturity = 1 - math.exp(-state['age_ticks'] / 100)
    result = dict(base)
    for name in ('planning', 'attention'):
        result[name] *= 0.35 + 0.65 * maturity
        result[name] *= 1-0.45*personality.fatigue(state)
    return result


def retrieve(memories, query, traits, tick, limit=8,state=None):
    words = set(query.lower().split())
    ranked = []
    for memory in memories:
        overlap = len(words & set(memory['text'].lower().split())) / max(1, len(words))
        age = max(0, tick - memory['tick'])
        decay = math.exp(-age / (20 + 300 * traits['memory_retention']))
        score = (0.35+0.15*traits['attention']) * overlap + 0.3 * memory['salience'] + 0.2 * decay
        if state and 'personality' in state:
            mood=state['affect']['valence']
            disposition=state['personality']['dispositions']['negative_bias' if mood<0 else 'positive_bias']
            congruence=mood*memory['reward']
            score*=1+0.2*disposition*congruence
        ranked.append((score, memory))
    ranked.sort(key=lambda item: (-item[0], -item[1]['id']))
    return [{**m, 'retrieval_score': round(s, 4)} for s, m in ranked[:limit]]


def decide(traits, state, memories, relations, rng):
    t = effective_traits(traits, state)
    scores = {
        'rest': 1.8 * (1 - state['energy']) + state['stress'] * 0.6,
        'explore': t['curiosity'] * state['novelty_need'] + t['novelty_preference'] * 0.3 - t['caution'] * state['stress'],
        'practice': t['persistence'] * 0.6 + t['planning'] * 0.4 + (1 - max(state['skills'].values())) * 0.25,
        'socialize': t['sociability'] * state['social_need'] + 0.15,
        'help': t['cooperation'] * 0.35 + t['empathy'] * 0.25,
        'sleep': state.get('body',{}).get('sleep_pressure',0)*1.5,
        'eat': state.get('body',{}).get('hunger',0)*1.3 if state['resources']>=0.5 else -10,
        'drink': state.get('body',{}).get('hydration_need',0)*1.2 if state['resources']>=0.1 else -10,
        'reflect': state.get('affect',{}).get('frustration',0)*0.5+t['planning']*0.15,
        'create':0.15+0.2*state['novelty_need'] if state['resources']>=0.2 else -10,
        'play':0.1+0.2*state['stress'],
        'plan':0.1+0.15*t['planning'],
        'repair':0.05,
    }
    evidence = []
    for memory in memories:
        action = memory.get('action')
        if action in scores:
            influence = memory['reward'] * memory['retrieval_score'] * 0.15
            scores[action] += influence
            evidence.append({'memory_id': memory['id'], 'action': action, 'influence': round(influence, 4)})
    for action in scores:
        scores[action] += state['learned_values'].get(action,0) * t['reward_sensitivity'] * 0.4
        if action != 'rest':
            scores[action] -= (1 - state['energy']) * 0.5
    bias_scores(scores,state)
    personality.bias(scores,state)
    facet_evidence=human_dynamics.bias(scores,state)
    preference_evidence=None
    if 'action_preferences' in state.get('learned_skills',{}):
        words=[]
        for field in ('energy','stress','social_need','novelty_need'):
            value=state[field]
            words.append(field+'_'+('low' if value<0.3 else 'high' if value>0.7 else 'medium'))
        context=' '.join(words)
        preference_evidence=learned_skills.predict(state,'action_preferences',context)
        preference_evidence['context']=context
        if not preference_evidence['abstain']:
            for label,score in preference_evidence['scores'].items():
                if label in scores:
                    scores[label]+=0.5*score
    if 'brain' in state:
        threshold=0.3+0.4*state['brain']['parameters']['sleep_threshold']
        pressure=state['body']['sleep_pressure']
        scores['sleep']+=max(0,pressure-threshold)*1.5
    available = [r for r in relations if r['trust'] > 0.15]
    if not available:
        scores['socialize'] = scores['help'] = -10
    repair_partners=[r for r in relations if r['trust']<0.6 and r['other_id'] in state.get('social_dynamics',{})]
    if not repair_partners:scores['repair']=-10
    forecasts={}
    if 'brain' in state and 'executive' not in state['brain']['lesions']:
        for candidate in scores:
            prediction=forecast(state,candidate,depth=2+int(t['planning']>0.6))
            forecasts[candidate]=prediction
            if prediction['known'] and scores[candidate]>-9:
                delay=state.get('personality',{}).get('dispositions',{}).get('delay_discounting',0)
                scores[candidate]+=t['planning']*0.15*((1-0.5*delay)*prediction['expected_return']-t['caution']*prediction['uncertainty'])
    # Softmax implements exploration; planning sharpens the choice distribution.
    temperature = 0.15 + 0.35 * t['curiosity'] * (1 - t['planning'])
    temperature+=0.08*state.get('personality',{}).get('dispositions',{}).get('impulsivity',0)*state['stress']
    if 'brain' in state:
        temperature *= 1-0.3*state['brain']['parameters']['inhibition']
    maximum = max(scores.values())
    # Mask impossible actions; a softmax penalty alone still allows rare invalid choices.
    weights = [0 if scores[a]<=-9 else math.exp((scores[a] - maximum) / temperature) for a in ACTIONS]
    action = rng.choices(ACTIONS, weights=weights)[0]
    partner = None
    if action in ('socialize', 'help','repair'):
        candidates=repair_partners if action=='repair' else available
        if state.get('personality',{}).get('schema_version',0)>=5:
            partner_weights=[human_dynamics.partner_weight(state,r) for r in candidates]
        else:partner_weights=[0.1+r['trust'] for r in candidates]
        partner = rng.choices(candidates, weights=partner_weights)[0]['other_id']
    skill = min(state['skills'], key=state['skills'].get)
    return {'action': action, 'partner_id': partner, 'skill': skill,
            'scores': {k: round(v, 5) for k, v in scores.items()},
            'memory_evidence': evidence, 'effective_traits': t,
            'forecasts':forecasts,
            'learned_preference_evidence':preference_evidence,
            'facet_evidence':facet_evidence or [],
            'temperature': temperature}


def transition(traits, state, decision, rng, partner_traits=None):
    s = copy.deepcopy(state)
    action = decision['action']
    trust_delta = 0.0
    reward = 0.0
    detail = ''
    s['age_ticks'] += 1
    s['energy'] -= 0.035
    s['social_need'] += 0.025
    s['novelty_need'] += 0.02
    if action in ('rest','sleep'):
        s['energy'] += 0.2 + traits['recovery'] * 0.15
        s['stress'] -= 0.12
        reward = 0.25
        detail = 'Recovered energy and reduced stress.'
        if action == 'sleep':
            detail = 'Slept; memory replay is scheduled.'
    elif action in ('eat','drink'):
        cost = 0.5 if action=='eat' else 0.1
        if s['resources'] < cost:
            raise ValueError('Insufficient resources')
        s['resources'] -= cost
        s['energy'] += 0.1
        reward = 0.3
        detail = 'Consumed simulated food.' if action=='eat' else 'Consumed simulated water.'
    elif action == 'reflect':
        s['stress'] -= 0.05
        reward = 0.1
        detail = 'Reviewed recent outcomes; memory replay is scheduled.'
    elif action=='plan':
        s['energy']-=0.02
        reward=0.03
        detail='Compared remembered outcomes and prepared a short-lived intention.'
    elif action=='play':
        s['energy']-=0.04
        s['stress']-=0.08
        s['novelty_need']-=0.15
        reward=0.15
        detail='Engaged in simulated play; reduced stress and novelty need.'
    elif action=='create':
        if s['resources']<0.2:raise ValueError('Insufficient resources')
        s['resources']-=0.2
        s['energy']-=0.07
        p=s.get('personality',{}).get('dispositions',{})
        gain=0.01*traits['plasticity']*(1-s['skills']['craft'])
        s['skills']['craft']=clamp(s['skills']['craft']+gain)
        s['novelty_need']-=0.3
        # Recombination creates a real structured object; aesthetic merit unmeasured.
        shape=rng.choice(('spiral','branch','grid','wave'))
        material=rng.choice(('paper','wood','glass','clay'))
        artifact={'index':s.get('creative_output_count',0), 'tick':s['age_ticks'],
                  'kind':'combinatorial_design_sketch','title':f'{material} {shape}',
                  'elements':[material,shape,rng.choice(('warm','cool','contrasting'))],
                  'origin':'synthetic recombination; not model-generated or human-evaluated'}
        s.setdefault('artifacts',[]).append(artifact)
        s['artifacts']=s['artifacts'][-32:]
        s['creative_output_count']=artifact['index']+1
        reward=0.1+0.1*p.get('creativity',0)
        detail=f'Created structured design sketch: {artifact["title"]}.'
    elif action == 'explore':
        success = rng.random() < 0.4 + 0.35 * s['skills']['reasoning']
        reward = 0.7 if success else -0.35
        s['resources'] = max(0, s['resources'] + (1.0 if success else -0.3))
        s['novelty_need'] -= 0.5
        s['stress'] += -0.04 if success else 0.12
        s['energy'] -= 0.08
        detail = 'Found resources.' if success else 'Exploration failed.'
    elif action == 'practice':
        skill = decision['skill']
        gain = (0.01 + 0.04 * traits['plasticity']) * max(0,s['energy']) * (1 - s['skills'][skill])
        s['skills'][skill] = clamp(s['skills'][skill] + gain)
        s['energy'] -= 0.07
        reward = gain * 10
        detail = f'Practiced {skill}; proficiency increased by {gain:.4f}.'
    elif action in ('socialize', 'help','repair'):
        if partner_traits is None:
            raise ValueError('Social actions require a present partner')
        reciprocal = rng.random() < partner_traits['cooperation']
        reward = (0.55 if reciprocal else -0.4)
        if action == 'help':
            spent = min(0.5, s['resources'])
            s['resources'] -= spent
            reward += traits['empathy'] * spent * 0.3
        s['social_need'] -= 0.4
        s['energy'] -= 0.05
        trust_delta = (0.12 if reciprocal else -0.18) * (0.5 + traits['reciprocity'])
        detail = 'Partner cooperated.' if reciprocal else 'Partner did not reciprocate.'
        if action=='repair':
            trust_delta*=0.5
            reward*=0.5
            detail='Attempted relationship repair; '+detail
    reward = clamp(reward, -1, 1)
    alpha = 0.05 + 0.25 * traits['plasticity']
    alpha*=1-0.4*personality.fatigue(s)
    s['learned_values'][action] += alpha * (reward - s['learned_values'][action])
    s['confidence'] = clamp(s['confidence'] + alpha * reward * 0.1)
    s['self_model']['successful_actions' if reward >= 0 else 'failed_actions'] += 1
    for field in ('energy', 'stress', 'social_need', 'novelty_need'):
        s[field] = clamp(s[field])
    return s, {'action': action, 'reward': reward, 'text': f'{action}: {detail}',
               'salience': clamp(abs(reward) + traits['threat_sensitivity'] * max(0, -reward)),
               'trust_delta': trust_delta, 'partner_id': decision['partner_id']}
