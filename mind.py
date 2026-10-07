"""Working memory, semantic evidence, appraisal, goals, replay and social inference."""
import copy
import math
import brain
import personality
import human_dynamics


def clamp(v, low=0, high=1):
    return max(low,min(high,v))


def upgrade(state, dna):
    s = copy.deepcopy(state)
    s.setdefault('brain', brain.compile_program(dna) if 'brain' not in s else s['brain'])
    defaults = {
        'body': {'hunger':0.2,'sleep_pressure':0.1,'hydration_need':0.1},
        'affect': {'valence':0.0,'arousal':0.2,'fear':0.0,'frustration':0.0},
        'workspace': [], 'knowledge': {}, 'goals': [], 'social_beliefs': {},
        'habits': {}, 'prediction_error':0.0,
        'metacognition': {'samples':0,'squared_error':0.0,'last_prediction':0.0},
        'consolidated_memory_ids': [], 'replay_count':0,
        'world_model': {},
        'identity_profile': {}, 'learned_skills': {}, 'transfer_memory': {},
        'perspective_memory': {'events':[],'actual':{},'beliefs':{}},
    }
    for key,value in defaults.items():
        s.setdefault(key,value)
    for action in ('sleep','eat','drink','reflect'):
        s['learned_values'].setdefault(action,0.0)
    return s


def attend(state, item):
    capacity = 2 + int(state['brain']['parameters']['workspace_capacity']*6)
    if 'personality' in state:
        sensitivity=state['personality']['dispositions'].get('sensory_sensitivity',0)
        capacity=max(1,int(capacity*(1-0.35*personality.fatigue(state)-0.2*sensitivity*state['affect']['arousal'])))
    if 'working_memory' in state['brain']['lesions']:
        capacity = 0
    state['workspace'].append(item)
    state['workspace'].sort(key=lambda x:(-x.get('priority',0),-x.get('tick',0)))
    state['workspace'] = state['workspace'][:capacity]


def evidence(state, subject, predicate, value, confidence, source, memory_id):
    key = json_key(subject,predicate)
    record = state['knowledge'].setdefault(key,{'subject':subject,'predicate':predicate,'alternatives':{}})
    entry = record['alternatives'].setdefault(value,{'support':0.0,'evidence':[]})
    # Repeated replay cannot manufacture independent supporting evidence.
    if memory_id not in [e['memory_id'] for e in entry['evidence']]:
        entry['support'] += confidence
        entry['evidence'].append({'memory_id':memory_id,'source':source,'confidence':confidence})
    total = sum(e['support'] for e in record['alternatives'].values())
    for candidate in record['alternatives'].values():
        candidate['relative_support'] = candidate['support']/total if total else 0
    record['conflicted'] = len(record['alternatives']) > 1
    return record


def json_key(subject,predicate):
    import json
    return json.dumps([subject,predicate],separators=(',',':'))


def learn(state, traits, outcome, memory_id, tick, previous_state=None):
    reward, action = outcome['reward'], outcome['action']
    model=state.setdefault('world_model',{}).setdefault(action,{'samples':0,'reward_mean':0.0,'reward_m2':0.0,'effects':{}})
    model['samples']+=1
    difference=reward-model['reward_mean']
    model['reward_mean']+=difference/model['samples']
    model['reward_m2']+=difference*(reward-model['reward_mean'])
    if previous_state:
        for field in ('energy','stress','resources','social_need','novelty_need'):
            observed=state[field]-previous_state[field]
            mean=model['effects'].get(field,0.0)
            model['effects'][field]=mean+(observed-mean)/model['samples']
    expected = state['metacognition']['last_prediction']
    error = reward-expected
    state['prediction_error'] = error
    # Bounded, reward-modulated local plasticity: inherited routing develops
    # through experience while the original genome stays immutable.
    if 'prediction_error' not in state['brain']['lesions']:
        activity=state['brain']['activities']
        rate=0.002*state['brain']['parameters']['learning_rate']
        for i,(source,target) in enumerate(brain.EDGES):
            delta=rate*error*activity[source]*activity[target]
            state['brain']['weights'][i]=clamp(state['brain']['weights'][i]+delta,0.02,0.8)
    stats = state['metacognition']
    stats['samples'] += 1
    stats['squared_error'] += error*error
    affect = state['affect']
    affect['valence'] = clamp(0.8*affect['valence']+0.2*reward,-1,1)
    affect['arousal'] = clamp(0.7*affect['arousal']+0.3*abs(error))
    affect['fear'] = clamp(0.8*affect['fear']+max(0,-reward)*traits['threat_sensitivity']*0.3)
    affect['frustration'] = clamp(0.85*affect['frustration']+max(0,-error)*0.15)
    habit = state['habits'].setdefault(action, {'count':0,'reward_sum':0.0})
    habit['count'] += 1
    habit['reward_sum'] += reward
    partner = outcome.get('partner_id')
    if partner:
        belief = state['social_beliefs'].setdefault(partner,{'alpha':1.0,'beta':1.0,'evidence_ids':[]})
        positive = outcome.get('trust_delta',0) > 0
        belief['alpha' if positive else 'beta'] += 0.5+state['brain']['parameters']['social_update_rate']
        belief['evidence_ids'].append(memory_id)
        belief['evidence_ids'] = belief['evidence_ids'][-100:]
        belief['cooperation_expectation'] = belief['alpha']/(belief['alpha']+belief['beta'])
    state['body']['hunger'] = clamp(state['body']['hunger']+0.025)
    state['body']['sleep_pressure'] = clamp(state['body']['sleep_pressure']+0.04)
    state['body']['hydration_need'] = clamp(state['body']['hydration_need']+0.02)
    if action in ('rest','sleep'):
        state['body']['sleep_pressure'] = clamp(state['body']['sleep_pressure']-0.15)
    if action == 'eat':
        state['body']['hunger'] = clamp(state['body']['hunger']-0.5)
    if action == 'drink':
        state['body']['hydration_need'] = clamp(state['body']['hydration_need']-0.5)
    attend(state,{'text':outcome['text'],'memory_id':memory_id,'tick':tick,'priority':abs(error)})
    for goal in state['goals']:
        if goal['status'] != 'active':
            continue
        current = goal_value(state,goal['metric'])
        goal['progress'] = current
        achieved = current >= goal['target'] if goal['direction']=='at_least' else current <= goal['target']
        if achieved:
            goal['status'] = 'achieved'
    personality.update(state,traits,outcome)
    human_dynamics.update(state,traits,outcome,memory_id,tick)
    brain.cycle(state,sensory=min(1,abs(error)),retrieval=min(1,len(state['workspace'])/8))


def goal_value(state, metric):
    if not isinstance(metric,str):
        raise ValueError('metric must be a string')
    if metric in ('energy','resources'):
        return state[metric]
    if metric.startswith('skill:') and metric[6:] in state['skills']:
        return state['skills'][metric[6:]]
    raise ValueError('metric must be energy, resources, or skill:<supported skill>')


def consolidate(state, memories):
    if 'hippocampus' in state['brain']['lesions'] or 'semantic' in state['brain']['lesions']:
        return {'replayed':0,'reason':'memory module disabled'}
    seen = set(state['consolidated_memory_ids'])
    pending = [m for m in memories if m['id'] not in seen and m['source']=='simulation']
    pending.sort(key=lambda m:(-m['salience'],m['id']))
    limit = 1+int(state['brain']['parameters']['replay_fraction']*15)
    replayed = []
    for m in pending[:limit]:
        evidence(state,m['action'],'observed_outcome',m['text'],0.5,'simulation',m['id'])
        replayed.append(m['id'])
        # Replay gradually changes action values without inventing new observations.
        if m['action'] in state['learned_values']:
            alpha = 0.01+0.04*state['brain']['parameters']['learning_rate']
            state['learned_values'][m['action']] += alpha*(m['reward']-state['learned_values'][m['action']])
    state['consolidated_memory_ids'].extend(replayed)
    state['replay_count'] += len(replayed)
    return {'replayed':len(replayed),'memory_ids':replayed}


def bias_scores(scores, state):
    if 'brain' not in state:
        return
    activity = state['brain']['activities']
    parameters = state['brain']['parameters']
    if 'executive' not in state['brain']['lesions']:
        for goal in state['goals']:
            if goal['status'] != 'active':
                continue
            if goal['direction']=='at_most':
                action={'energy':'practice','resources':'help'}.get(goal['metric'],'rest')
            else:
                action = {'energy':'rest','resources':'explore'}.get(goal['metric'],'practice')
            scores[action] += goal['priority']*(0.3+activity['executive'])
    if 'striatum' not in state['brain']['lesions']:
        for action,habit in state['habits'].items():
            if action in scores:
                scores[action] += parameters['habit_weight']*habit['reward_sum']/max(1,habit['count'])*0.3
    scores['explore'] += activity['salience']*0.1-state['affect']['fear']*0.2
    scores['rest'] += state['body']['sleep_pressure']*0.2
    if 'socialize' in scores and state['social_beliefs']:
        expected = sum(b.get('cooperation_expectation',0.5) for b in state['social_beliefs'].values())/len(state['social_beliefs'])
        scores['socialize'] += (expected-0.5)*0.3


def forecast(state,action,depth=2):
    """Bounded lookahead over learned action effects, with explicit unknowns."""
    models=state.get('world_model',{})
    initial=models.get(action)
    if not initial:
        return {'known':False,'samples':0,'expected_return':0.0,'uncertainty':1.0,'plan':[action]}
    projected={key:state[key] for key in ('energy','stress','resources','social_need','novelty_need')}
    def apply_effects(model):
        for key,value in model['effects'].items():
            projected[key]=max(0,projected[key]+value) if key=='resources' else clamp(projected[key]+value)
    apply_effects(initial)
    discount=0.8+0.19*state['brain']['parameters']['discount']
    value=initial['reward_mean']
    plan=[action]
    for step in range(1,depth):
        feasible=[(name,model) for name,model in models.items()
                  if not (name in ('eat','drink','create') and projected['resources']<{'eat':0.5,'drink':0.1,'create':0.2}[name])]
        if not feasible:
            break
        # A low-energy projected state makes recovery more valuable.
        name,model=max(feasible,key=lambda item:item[1]['reward_mean']+
                       (max(0,0.3-projected.get('energy',0.5)) if item[0] in ('rest','sleep') else 0))
        value+=discount**step*model['reward_mean']
        plan.append(name)
        apply_effects(model)
    variance=initial['reward_m2']/max(1,initial['samples']-1)
    return {'known':True,'samples':initial['samples'],'expected_return':value,
            'uncertainty':math.sqrt(variance/max(1,initial['samples']))+1/math.sqrt(initial['samples']),
            'plan':plan,'projected_state':projected,
            'scope':'learned average effects; not a causal or complete environmental model'}
