"""Small trainable text classifiers: actual lifetime weight updates, not LLM fine-tuning."""
import hashlib
import math
import random
import re

DIMENSIONS=128


def vector(text):
    words=re.findall(r'[a-z0-9]+',text.lower())
    terms=words+[a+' '+b for a,b in zip(words,words[1:])]
    values=[0.0]*DIMENSIONS
    for term in terms:
        digest=hashlib.sha256(term.encode()).digest()
        index=int.from_bytes(digest[:4],'big')%DIMENSIONS
        values[index]+=1 if digest[4]%2 else -1
    norm=math.sqrt(sum(v*v for v in values)) or 1
    return [1.0]+[v/norm for v in values]


def probabilities(model,text):
    features=vector(text)
    logits=[sum(a*b for a,b in zip(weights,features)) for weights in model['weights']]
    maximum=max(logits)
    scores=[math.exp(v-maximum) for v in logits]
    total=sum(scores)
    return [v/total for v in scores],features


def validate_examples(examples):
    if not isinstance(examples,list) or not 1<=len(examples)<=500:
        raise ValueError('examples must contain 1–500 text/label records')
    for e in examples:
        if not isinstance(e,dict) or set(e)!=set(('text','label')):
            raise ValueError('Each example requires exactly text and label')
        if not isinstance(e['text'],str) or not 1<=len(e['text'])<=2000 or not isinstance(e['label'],str) or not 1<=len(e['label'])<=100:
            raise ValueError('Invalid example text or label')


def train(state,name,examples,epochs=60,seed=0):
    if not isinstance(name,str) or not 1<=len(name)<=100:
        raise ValueError('name must contain 1–100 characters')
    validate_examples(examples)
    if type(epochs) is not int or not 1<=epochs<=200 or type(seed) is not int:
        raise ValueError('epochs must be 1–200 and seed an integer')
    models=state.setdefault('learned_skills',{})
    if name not in models and len(models)>=30:
        raise ValueError('Skill capacity reached')
    supplied_labels=sorted(set(e['label'] for e in examples))
    if name in models:
        model=models[name]
        labels=model['labels']
        if not set(supplied_labels)<=set(labels):
            raise ValueError('Continued training must use existing class labels')
    else:
        labels=supplied_labels
        if not 2<=len(labels)<=20:
            raise ValueError('Initial training requires 2–20 classes')
        model={'labels':labels,'weights':[[0.0]*(DIMENSIONS+1) for _ in labels],
               'training_examples':[],'updates':0,'replay_examples':[],
               'feature_version':'signed-word-bigram-hash-v1'}
    rng=random.Random(seed)
    # Bounded rehearsal helps retain prior classes during incremental feedback.
    rehearsal=examples+model.get('replay_examples',[])[:64]
    indexes=list(range(len(rehearsal)))
    rate=0.08+0.12*state['brain']['parameters']['learning_rate']
    for _ in range(epochs):
        rng.shuffle(indexes)
        for i in indexes:
            example=rehearsal[i]
            scores,features=probabilities(model,example['text'])
            target=labels.index(example['label'])
            for label,weights in enumerate(model['weights']):
                error=(1 if label==target else 0)-scores[label]
                for j,value in enumerate(features):
                    weights[j]+=rate*(error*value-0.0001*weights[j])
            model['updates']+=1
    for e in examples:
        digest=hashlib.sha256(e['text'].encode()).hexdigest()
        if digest not in model['training_examples']:
            model['training_examples'].append(digest)
        if e not in model.setdefault('replay_examples',[]):
            model['replay_examples'].append(dict(e))
    model['replay_examples']=model['replay_examples'][-128:]
    models[name]=model
    correct=sum(predict(state,name,e['text'])['label']==e['label'] for e in examples)
    return {'name':name,'updates':model['updates'],'training_accuracy':correct/len(examples),
            'labels':labels,'algorithm':'linear softmax classifier, supervised SGD',
            'scope':'lexical feature learning; training accuracy is not held-out performance'}


def predict(state,name,text):
    if not isinstance(text,str) or not 1<=len(text)<=2000:
        raise ValueError('text must contain 1–2000 characters')
    model=state.get('learned_skills',{}).get(name)
    if model is None:
        raise LookupError('Learned skill not found')
    scores,_=probabilities(model,text)
    index=max(range(len(scores)),key=scores.__getitem__)
    return {'label':model['labels'][index],'scores':dict(zip(model['labels'],scores)),
            'calibrated':False,'training_updates':model['updates'],
            'abstain':scores[index]<0.6}


def evaluate(state,name,examples):
    validate_examples(examples)
    model=state.get('learned_skills',{}).get(name)
    if model is None:
        raise LookupError('Learned skill not found')
    if any(hashlib.sha256(e['text'].encode()).hexdigest() in model['training_examples'] for e in examples):
        raise ValueError('Held-out examples overlap recorded training text')
    if any(e['label'] not in model['labels'] for e in examples):
        raise ValueError('Evaluation label was not trained')
    predictions=[{'text':e['text'],'expected':e['label'],**predict(state,name,e['text'])} for e in examples]
    return {'accuracy':sum(e['label']==e['expected'] for e in predictions)/len(predictions),
            'examples':predictions,'held_out':True,'count':len(examples),
            'scope':'exact label agreement on supplied held-out examples, not human fidelity'}
