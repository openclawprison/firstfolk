"""Explicit visibility-based belief tracking; not learned human theory of mind."""


def event(state,object_name,location,observers):
    if any(not isinstance(v,str) or not 1<=len(v)<=100 for v in (object_name,location)):
        raise ValueError('object_name and location must be strings of 1–100 characters')
    if not isinstance(observers,list) or len(observers)>30 or any(not isinstance(v,str) or not 1<=len(v)<=100 for v in observers):
        raise ValueError('observers must be at most 30 actor names')
    world=state.setdefault('perspective_memory',{'events':[],'actual':{},'beliefs':{}})
    if len(world['events'])>=1000:
        raise ValueError('Perspective event capacity reached')
    record={'index':len(world['events']),'object':object_name,'location':location,'observers':list(set(observers))}
    world['events'].append(record)
    world['actual'][object_name]=location
    for observer in observers:
        world['beliefs'].setdefault(observer,{})[object_name]={'location':location,'event_index':record['index']}
    return record


def query(state,actor,object_name):
    if not isinstance(actor,str) or not isinstance(object_name,str):
        raise ValueError('actor and object_name must be strings')
    world=state.get('perspective_memory',{})
    belief=world.get('beliefs',{}).get(actor,{}).get(object_name)
    actual=world.get('actual',{}).get(object_name)
    return {'actor':actor,'object':object_name,'believed_location':belief['location'] if belief else None,
            'actual_location':actual,'evidence_event_index':belief['event_index'] if belief else None,
            'belief_differs_from_world':belief['location']!=actual if belief else None,
            'uncertainty':'Actor may have other information; only declared observations are modeled.',
            'mechanism':'symbolic observation-based bookkeeping'}
