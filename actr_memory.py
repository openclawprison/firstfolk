"""Optional upstream pyactr declarative retrieval with persistent rehearsal traces."""
import math
import threading
from importlib.metadata import version
LOCK=threading.RLock()
_declared=False


def provider():
    try:
        import pyactr
    except ImportError as exc:
        raise ValueError('Install requirements-cognition.txt to use pyactr') from exc
    global _declared
    if not _declared:
        pyactr.chunktype('backend_episode','identifier action')
        _declared=True
    return pyactr


def rank(state,memories,tick,limit=8,action=None,delay=1.0,threshold=-5.0,rehearse=True):
    if type(limit) is not int or not 1<=limit<=20:
        raise ValueError('limit must be 1–20')
    if type(delay) not in (int,float) or not math.isfinite(delay) or not 0.01<=delay<=100000:
        raise ValueError('delay must be finite in [0.01,100000]')
    if type(threshold) not in (int,float) or not math.isfinite(threshold) or not -20<=threshold<=10:
        raise ValueError('threshold must be finite in [-20,10]')
    from engine import ACTIONS
    if action is not None and action not in ACTIONS:raise ValueError('Unknown action cue')
    if type(rehearse) is not bool:raise ValueError('rehearse must be boolean')
    with LOCK:
        actr=provider()
        trace=state.setdefault('actr_memory',{'clock':0.0,'presentations':{},'retrievals':0})
        now=max(float(tick),trace['clock'])+delay
        trace['clock']=now
        retention=state['brain']['parameters']['replay_fraction']
        decay=0.25+0.5*(1-retention)
        model=actr.ACTRModel(subsymbolic=True,decay=decay,retrieval_threshold=threshold,
                             instantaneous_noise=0,latency_factor=0.1)
        index={}
        for record in memories:
            identifier=f'm{record["id"]}'
            chunk=actr.makechunk(typename='backend_episode',identifier=identifier,action=record.get('action') or 'none')
            times=trace['presentations'].setdefault(str(record['id']),[min(float(record['tick']),now-0.01)])
            model.decmem.add(chunk,time=times)
            index[chunk]=record
        model.retrieval.decmem=model.decmem
        model.retrieval.finst=limit+1
        cue=actr.makechunk(typename='backend_episode',**({'action':action} if action else {}))
        selected=[]
        for _ in range(limit):
            chunk,latency=model.retrieval.retrieve(now,cue,{}, {},{'recently_retrieved':False},model.model_parameters)
            if chunk is None:break
            memory=index[chunk]
            activation=float(model.retrieval.activation)
            selected.append({**memory,'actr_activation':activation,'modeled_latency':float(latency),
                             'retrieval_score':1/(1+math.exp(-activation))})
            if rehearse:
                times=trace['presentations'][str(memory['id'])]
                times.append(now);trace['presentations'][str(memory['id'])]=times[-100:]
                trace['retrievals']+=1
        return {'memories':selected,'provider':'pyactr','version':version('pyactr'),
                'clock':now,'decay':decay,'threshold':threshold,
                'scope':'ACT-R base-level declarative retrieval; time units are simulated, not fitted human response times'}


def configure(state,mode):
    if mode not in ('heuristic','actr'):raise ValueError('mode must be heuristic or actr')
    if mode=='actr':
        with LOCK:provider()
    state['memory_retrieval_mode']=mode
    return {'mode':mode,'scope':'Optional retrieval policy; does not replace the complete cognitive architecture'}
