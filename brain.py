"""Research-informed functional graph; activities are NOT neural measurements."""
import json
import math
import csv
from pathlib import Path

# Roles are distributed across the real brain; these are engineering analogies.
REGIONS = {
    'sensory': ('Sensory association systems', 'structured observation encoding', 'CoALA'),
    'thalamus': ('Thalamocortical systems', 'input routing', 'engineering'),
    'salience': ('Insula / anterior cingulate networks', 'priority and conflict signals', 'engineering'),
    'hippocampus': ('Hippocampal formation', 'fast episodic retrieval', 'CLS'),
    'semantic': ('Distributed association cortex', 'slow consolidation of evidence', 'CLS'),
    'working_memory': ('Frontoparietal systems', 'capacity-limited workspace', 'PFC'),
    'executive': ('Prefrontal control systems', 'goal and plan bias', 'PFC'),
    'valuation': ('Orbitofrontal / ventromedial systems', 'outcome value estimates', 'RPE'),
    'striatum': ('Corticostriatal systems', 'learned action selection', 'RPE'),
    'prediction_error': ('Midbrain reward-related systems', 'reward prediction error', 'RPE'),
    'threat': ('Amygdala-related systems', 'threat appraisal proxy', 'engineering'),
    'homeostasis': ('Hypothalamic / brainstem systems', 'resource and sleep drives', 'engineering'),
    'social': ('Distributed social cognition systems', 'beliefs about partner cooperation', 'CoALA'),
    'self_model': ('Distributed self-related systems', 'performance calibration', 'CoALA'),
    'cerebellum': ('Cerebellar systems', 'action-outcome prediction proxy', 'engineering'),
    'motor': ('Motor / premotor systems', 'selected action commitment', 'engineering'),
}
EDGES = (
    ('sensory','thalamus'), ('thalamus','salience'), ('thalamus','hippocampus'),
    ('salience','working_memory'), ('hippocampus','working_memory'),
    ('semantic','working_memory'), ('working_memory','executive'),
    ('executive','working_memory'), ('executive','striatum'),
    ('valuation','striatum'), ('prediction_error','valuation'),
    ('prediction_error','striatum'), ('threat','salience'),
    ('homeostasis','valuation'), ('social','valuation'), ('self_model','executive'),
    ('striatum','motor'), ('executive','motor'), ('motor','cerebellum'),
    ('cerebellum','prediction_error'), ('hippocampus','semantic'),
    ('social','working_memory'), ('threat','valuation'), ('salience','executive'),
)
PARAMETERS = ('workspace_capacity', 'learning_rate', 'discount', 'habit_weight',
              'replay_fraction', 'sleep_threshold', 'inhibition', 'social_update_rate')


def compile_program(dna):
    n = len(dna) // 2
    channels = []
    count = len(EDGES) + len(PARAMETERS)
    for i in range(count):
        left, right = i*n//count, (i+1)*n//count
        size = 2*(right-left)
        z = (sum(dna[left:right])+sum(dna[n+left:n+right])-127.5*size)/math.sqrt(5461.25*size)
        channels.append(1/(1+math.exp(-max(-20,min(20,z)))))
    return {'version': 'regulatory-functional-graph-v2',
            'weights': [round(0.1+0.5*v,6) for v in channels[:len(EDGES)]],
            'parameters': dict(zip(PARAMETERS, channels[len(EDGES):])),
            'activities': {key: 0.0 for key in REGIONS}, 'lesions': [], 'trace': []}


def cycle(state, sensory=0, retrieval=0, conflict=0):
    brain = state['brain']
    body = state['body']
    emotion = state['affect']
    inputs = {key: 0.0 for key in REGIONS}
    inputs.update(sensory=sensory, salience=max(sensory, conflict),
                  hippocampus=retrieval, semantic=min(1,len(state['knowledge'])/20),
                  executive=min(1,len([g for g in state['goals'] if g['status']=='active'])/3),
                  homeostasis=max(body['hunger'],body['sleep_pressure']),
                  threat=emotion['fear'], valuation=max(0,emotion['valence']),
                  prediction_error=abs(state['prediction_error']),
                  self_model=state['confidence'], social=min(1,len(state['social_beliefs'])/5))
    activity = dict(brain['activities'])
    # Stable leaky graph propagation. No empirical connectome claim.
    for _ in range(3):
        drive = dict(inputs)
        degree = {key: 1 for key in REGIONS}
        for (source, target), weight in zip(EDGES, brain['weights']):
            drive[target] += weight * activity[source]
            degree[target] += 1
        activity = {key: (0 if key in brain['lesions'] else
                         round(0.4*activity[key]+0.6*math.tanh(drive[key]/math.sqrt(degree[key])),6))
                    for key in REGIONS}
    brain['activities'] = activity
    brain['trace'] = [{'region': key, 'activity': value} for key,value in sorted(activity.items(),key=lambda x:-x[1])]
    return activity


def atlas():
    root = Path(__file__).parent/'research'/'third_party'
    path = root/'Schaefer2018_100Parcels_7Networks_order.txt'
    parcels = []
    if path.exists():
        for line in path.read_text().splitlines():
            fields = line.split()
            if len(fields) >= 2:
                label = fields[1]
                parts = label.split('_')
                parcels.append({'id':int(fields[0]), 'label':label, 'hemisphere':parts[1],
                                'network':parts[2], 'color':[int(v) for v in fields[2:5]]})
    centroid_path = root/'Schaefer2018_centroids.csv'
    if centroid_path.exists():
        with centroid_path.open(newline='') as stream:
            coords={int(row['ROI Label']):[float(row[k]) for k in ('R','A','S')] for row in csv.DictReader(stream)}
        for parcel in parcels:
            parcel['centroid_ras_mm']=coords.get(parcel['id'])
    return {'name':'Schaefer2018 100 parcels / 7 networks', 'parcels':parcels,
            'provenance':json.loads((root/'provenance.json').read_text()) if (root/'provenance.json').exists() else {},
            'scope':'Published cortical labels and MNI centroid coordinates; no measured connectivity or voxel geometry bundled.'}


def mapping(state=None):
    return {'kind':'functional engineering analogy',
            'regions':[{'id':key,'anatomical_analogy':value[0],'implemented_role':value[1],
                        'research_key':value[2], 'activity': state['brain']['activities'][key] if state else None}
                       for key,value in REGIONS.items()],
            'edges':[{'source':a,'target':b,'origin':'hand-designed synthetic routing',
                      'weight': state['brain']['weights'][i] if state else None}
                     for i,(a,b) in enumerate(EDGES)],
            'atlas':atlas(), 'limitations':'Not a connectome, neuron simulation, fMRI, or validated localization of human traits.'}
