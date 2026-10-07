"""Task-specific replication of supplied choices; never an overall human score."""
import hashlib
import json
import math
import skills
from engine import ACTIONS


def records(examples):
    if not isinstance(examples,list) or not 2<=len(examples)<=500:
        raise ValueError('Provide 2–500 context/action records')
    result=[]
    for row in examples:
        if not isinstance(row,dict) or set(row)!={'context','action'}:
            raise ValueError('Records require exactly context and action')
        if not isinstance(row['context'],str) or not 1<=len(row['context'])<=2000 or row['action'] not in ACTIONS:
            raise ValueError('Invalid context or unsupported action')
        context=' '.join(row['context'].lower().split())
        if not context:raise ValueError('Context must contain text')
        result.append({'text':context,'label':row['action']})
    return result


def train(state,participant,examples,source,epochs=80,seed=0):
    if not isinstance(participant,str) or not 1<=len(participant)<=100:
        raise ValueError('participant must be a short label')
    if not isinstance(source,str) or not 1<=len(source)<=200:
        raise ValueError('source must identify the supplied dataset')
    samples=records(examples)
    name='participant_'+hashlib.sha256(participant.encode()).hexdigest()[:16]
    result=skills.train(state,name,samples,epochs,seed)
    registry=state.setdefault('participant_models',{})
    model=registry.setdefault(participant,{'name':name,'training_contexts':[],'action_counts':{}})
    model['source']=source
    for sample in samples:
        digest=hashlib.sha256(sample['text'].encode()).hexdigest()
        if digest not in model['training_contexts']:
            model['training_contexts'].append(digest)
        label=sample['label']
        model['action_counts'][label]=model['action_counts'].get(label,0)+1
    return {**result,'participant':participant,'source':source,'source_verified':False,
            'scope':'Train lexical action predictor from supplied demonstrations; not a psychological clone'}


def predict(state,participant,context):
    if not isinstance(context,str) or not 1<=len(context)<=2000 or not context.strip():
        raise ValueError('context must contain 1–2000 characters')
    model=state.get('participant_models',{}).get(participant)
    if model is None:raise LookupError('Participant model not found')
    result=skills.predict(state,model['name'],' '.join(context.lower().split()))
    return {**result,'predicted_action':result['label'],'participant':participant,
            'declared_source':model['source'],'source_verified':False,
            'scope':'Prediction from supplied context-choice examples; not knowledge of the real person'}


def evaluate(state,participant,examples):
    samples=records(examples)
    model=state.get('participant_models',{}).get(participant)
    if model is None:raise LookupError('Participant model not found')
    if any(hashlib.sha256(s['text'].encode()).hexdigest() in model['training_contexts'] for s in samples):
        raise ValueError('Held-out contexts overlap normalized training contexts')
    if len({s['text'] for s in samples})!=len(samples):
        raise ValueError('Held-out contexts must be distinct')
    learner=state['learned_skills'][model['name']]
    checkpoint=hashlib.sha256(json.dumps(learner,sort_keys=True).encode()).hexdigest()
    baseline=max(model['action_counts'],key=lambda k:(model['action_counts'][k],k))
    predictions=[{'context':s['text'],'expected':s['label'],**skills.predict(state,model['name'],s['text'])} for s in samples]
    correct=sum(p['label']==p['expected'] for p in predictions)
    n=len(predictions);rate=correct/n;z=1.96
    center=(rate+z*z/(2*n))/(1+z*z/n)
    margin=z*math.sqrt(rate*(1-rate)/n+z*z/(4*n*n))/(1+z*z/n)
    answered=[p for p in predictions if not p['abstain']]
    after=hashlib.sha256(json.dumps(learner,sort_keys=True).encode()).hexdigest()
    return {'participant':participant,'declared_source':model['source'],'source_verified':False,
            'samples':n,'agreement_percent':100*rate,'agreement_wilson95_percent':[100*(center-margin),100*(center+margin)],
            'unseen_action_count':sum(s['label'] not in learner['labels'] for s in samples),
            'majority_baseline_percent':100*sum(p['expected']==baseline for p in predictions)/n,
            'non_abstaining_coverage_percent':100*len(answered)/n,
            'non_abstaining_agreement_percent':100*sum(p['label']==p['expected'] for p in answered)/len(answered) if answered else None,
            'predictions':predictions,'frozen_checkpoint':checkpoint,'model_unchanged':checkpoint==after,
            'human_equivalence_percent':None,
            'scope':'Exact action agreement on supplied held-out records only. Interval assumes independent trials; repeated-person contexts may violate this.'}
