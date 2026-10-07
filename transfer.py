"""Reusable action models and navigation with local observations only.

The planner is engineered. Action displacements are learned from transitions.
It never receives the full grid, unlike the legacy route tool.
"""
import copy
import hashlib
import json
import random
from collections import deque
from capabilities import grid_spec, move, MOVES


def observe(grid,position,goal):
    x,y=position
    cells=[]
    # Local sensory field, expressed in spatial coordinates rather than action IDs.
    for dx,dy in ((0,0),)+MOVES:
        nx,ny=x+dx,y+dy
        wall=not (0<=ny<len(grid) and 0<=nx<len(grid[0])) or grid[ny][nx]=='#'
        cells.append({'position':[nx,ny],'blocked':wall})
    return {'position':list(position),'goal':list(goal),'cells':cells}


def model_for(state):
    return state.setdefault('transfer_memory',{}).setdefault('movement-v1',{
        'version':'movement-v1','actions':{},'training_layout_hashes':[],
        'observations':0,'training_steps':0})


def update(model,action,before,after):
    dx,dy=after[0]-before[0],after[1]-before[1]
    model['observations']+=1
    if (dx,dy)==(0,0):
        return
    if abs(dx)+abs(dy)!=1:
        raise ValueError('Only cardinal unit movement transitions are supported')
    data=model['actions'].setdefault(str(action),{})
    key=f'{dx},{dy}'
    data[key]=data.get(key,0)+1


def displacements(model):
    result={}
    for action,counts in model['actions'].items():
        if counts:
            result[int(action)]=tuple(map(int,max(counts,key=counts.get).split(',')))
    return result


def path_action(known,position,goal,actions,visits):
    """BFS over sensed free cells; select a frontier when the goal is unseen."""
    queue=deque([position])
    routes={position:[]}
    while queue:
        here=queue.popleft()
        for action,(dx,dy) in actions.items():
            target=(here[0]+dx,here[1]+dy)
            if known.get(target) is False and target not in routes:
                routes[target]=routes[here]+[action]
                queue.append(target)
    if goal in routes and routes[goal]:
        return routes[goal][0],'observed route to goal'
    frontier=[]
    for p,route in routes.items():
        if route and any((p[0]+dx,p[1]+dy) not in known for dx,dy in MOVES):
            distance=abs(p[0]-goal[0])+abs(p[1]-goal[1])
            frontier.append((distance+0.25*len(route)+0.5*visits.get(p,0),p,route))
    if frontier:
        frontier.sort()
        return frontier[0][2][0],'explore observed frontier'
    return None,'no observed path'


def navigate(model,grid,seed=0,max_steps=None,adapt=False,control_order=None,trace_limit=20):
    start,goal=grid_spec(grid)
    if type(seed) is not int or type(adapt) is not bool:
        raise ValueError('seed must be integer and adapt boolean')
    if max_steps is None:
        max_steps=min(1500,len(grid)*len(grid[0])*8)
    if type(max_steps) is not int or not 1<=max_steps<=2000:
        raise ValueError('max_steps must be 1–2000')
    order=list(range(4)) if control_order is None else control_order
    if not isinstance(order,list) or len(order)!=4 or any(type(v) is not int for v in order) or sorted(order)!=list(range(4)):
        raise ValueError('control_order must be a permutation of 0,1,2,3')
    working=model if adapt else copy.deepcopy(model)
    rng=random.Random(seed)
    known={}
    position=start
    visits={}
    trace=[]
    collisions=0
    shifts=0
    for step in range(max_steps):
        observation=observe(grid,position,goal)
        for cell in observation['cells']:
            known[tuple(cell['position'])]=cell['blocked']
        if position==goal:
            return {'success':True,'steps':step,'collisions':collisions,'known_cells':len(known),
                    'trace':trace,'adapted':adapt,'control_shifts_detected':shifts,
                    'observation_scope':'position, goal coordinate and adjacent cells only'}
        visits[position]=visits.get(position,0)+1
        actions=displacements(working)
        if len(actions)==4:
            action,reason=path_action(known,position,goal,actions,visits)
        else:
            action,reason=None,'unknown controls'
        if action is None:
            action=rng.randrange(4)
        target=move(grid,position,order[action])
        if target==position:
            collisions+=1
        if adapt:
            expected=actions.get(action)
            actual=(target[0]-position[0],target[1]-position[1])
            unexpected_collision=expected and actual==(0,0) and known.get((position[0]+expected[0],position[1]+expected[1])) is False
            if expected and ((actual!=(0,0) and actual!=expected) or unexpected_collision):
                # Evidence of changed controls invalidates only that action's model.
                working['actions'][str(action)]={}
                shifts+=1
            update(working,action,position,target)
        if len(trace)<trace_limit:
            trace.append({'position':list(position),'action':action,'next_position':list(target),
                          'reason':reason,'model_known_actions':len(actions)})
        position=target
    return {'success':position==goal,'steps':max_steps,'collisions':collisions,
            'known_cells':len(known),'trace':trace,'adapted':adapt,'control_shifts_detected':shifts,
            'observation_scope':'position, goal coordinate and adjacent cells only'}


def train(state,grids,steps_per_grid=200,seed=0):
    if not isinstance(grids,list) or not 1<=len(grids)<=30:
        raise ValueError('grids must contain 1–30 layouts')
    if type(steps_per_grid) is not int or not 10<=steps_per_grid<=1000 or type(seed) is not int:
        raise ValueError('steps_per_grid must be 10–1000 and seed an integer')
    for grid in grids:
        grid_spec(grid)
    model=model_for(state)
    rng=random.Random(seed)
    previous=model['training_steps']
    for grid in grids:
        start,goal=grid_spec(grid)
        position=start
        for _ in range(steps_per_grid):
            action=rng.randrange(4)
            target=move(grid,position,action)
            update(model,action,position,target)
            model['training_steps']+=1
            position=start if target==goal else target
        key=hashlib.sha256(json.dumps(grid).encode()).hexdigest()
        if key not in model['training_layout_hashes']:
            model['training_layout_hashes'].append(key)
    return {'skill':'movement-v1','new_training_steps':model['training_steps']-previous,
            'learned_displacements':{str(k):list(v) for k,v in displacements(model).items()},
            'algorithm':'transition counting + learned movement model + engineered frontier planner',
            'scope':'transfer within cardinal grid movement; not human general intelligence'}


def generate_grid(seed,size=9):
    """Random maze with carved paths; generation has no access to agent policy."""
    if type(size) is not int or size<5 or size>21 or size%2==0:
        raise ValueError('size must be odd and between 5 and 21')
    rng=random.Random(seed)
    cells=[['#']*size for _ in range(size)]
    stack=[(1,1)]
    cells[1][1]='.'
    while stack:
        x,y=stack[-1]
        candidates=[(x+2*dx,y+2*dy) for dx,dy in MOVES
                    if 0<x+2*dx<size-1 and 0<y+2*dy<size-1 and cells[y+2*dy][x+2*dx]=='#']
        if not candidates:
            stack.pop()
            continue
        nx,ny=rng.choice(candidates)
        cells[(y+ny)//2][(x+nx)//2]='.'
        cells[ny][nx]='.'
        stack.append((nx,ny))
    cells[1][1]='S'
    cells[size-2][size-2]='G'
    return [''.join(row) for row in cells]


def evaluate(state,seeds=None,size=11,max_steps=300):
    seeds=[1000+i for i in range(20)] if seeds is None else seeds
    if not isinstance(seeds,list) or not 1<=len(seeds)<=100 or any(type(s) is not int for s in seeds):
        raise ValueError('seeds must contain 1–100 integers')
    model=model_for(state)
    trials=[]
    for seed in seeds:
        grid=generate_grid(seed,size)
        digest=hashlib.sha256(json.dumps(grid).encode()).hexdigest()
        if digest in model['training_layout_hashes']:
            raise ValueError('Evaluation layout overlaps recorded training layouts')
        baseline=navigate({'actions':{}},grid,seed,max_steps,adapt=False,trace_limit=0)
        transferred=navigate(model,grid,seed,max_steps,adapt=False,trace_limit=0)
        trials.append({'seed':seed,'layout_hash':digest,'baseline':baseline,'transferred':transferred})
    return {'trials':trials,'baseline_success_rate':sum(t['baseline']['success'] for t in trials)/len(trials),
            'transfer_success_rate':sum(t['transferred']['success'] for t in trials)/len(trials),
            'frozen_weights':True,'held_out_layouts':True,
            'baseline':'random actions, same local observation interface and step budget',
            'scope':'learned primitive dynamics reused by an engineered planner on unseen layouts'}
