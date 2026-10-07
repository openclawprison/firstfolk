"""Grounded tools, symbolic planning and persistent reinforcement learning."""
import ast
import hashlib
import heapq
import json
import math
import operator
import random

OPS = {ast.Add:operator.add, ast.Sub:operator.sub, ast.Mult:operator.mul,
       ast.Div:operator.truediv, ast.Mod:operator.mod, ast.Pow:operator.pow}


def calculate(expression):
    if not isinstance(expression,str) or not 1<=len(expression)<=300:
        raise ValueError('expression must contain 1–300 characters')
    try:
        tree = ast.parse(expression,mode='eval')
        if sum(1 for _ in ast.walk(tree)) > 60:
            raise ValueError('Expression is too complex')
        def visit(node):
            if isinstance(node,ast.Constant) and type(node.value) in (int,float):
                result = node.value
            elif isinstance(node,ast.UnaryOp) and isinstance(node.op,(ast.UAdd,ast.USub)):
                result = visit(node.operand)*(1 if isinstance(node.op,ast.UAdd) else -1)
            elif isinstance(node,ast.BinOp) and type(node.op) in OPS:
                a,b = visit(node.left),visit(node.right)
                if isinstance(node.op,ast.Pow) and abs(b)>12:
                    raise ValueError('Exponent limit is 12')
                result = OPS[type(node.op)](a,b)
            else:
                raise ValueError('Only numeric arithmetic is supported')
            if type(result) not in (int,float) or not math.isfinite(result) or abs(result)>1e15:
                raise ValueError('Result is outside numeric limits')
            return result
        return {'result':visit(tree.body),'verified_by':'bounded arithmetic interpreter'}
    except (SyntaxError,ArithmeticError) as exc:
        raise ValueError('Invalid arithmetic expression') from exc


def grid_spec(grid):
    if not isinstance(grid,list) or not 1<=len(grid)<=30 or any(not isinstance(row,str) for row in grid):
        raise ValueError('grid must be 1–30 text rows')
    width=len(grid[0])
    if not 1<=width<=30 or any(len(row)!=width or set(row)-set('.#SG') for row in grid):
        raise ValueError('Rectangular grid, width 1–30, symbols . # S G required')
    starts=[(x,y) for y,row in enumerate(grid) for x,c in enumerate(row) if c=='S']
    goals=[(x,y) for y,row in enumerate(grid) for x,c in enumerate(row) if c=='G']
    if len(starts)!=1 or len(goals)!=1:
        raise ValueError('Exactly one S and G required')
    return starts[0],goals[0]


MOVES = ((0,-1),(1,0),(0,1),(-1,0))


def move(grid,position,action):
    dx,dy = MOVES[action]
    x,y = position[0]+dx,position[1]+dy
    return (x,y) if 0<=y<len(grid) and 0<=x<len(grid[0]) and grid[y][x]!='#' else position


def plan_route(grid):
    start,goal = grid_spec(grid)
    queue=[(0,0,start)]
    costs={start:0}
    parent={}
    expanded=0
    while queue:
        _,cost,current=heapq.heappop(queue)
        if cost!=costs[current]:
            continue
        expanded+=1
        if current==goal:
            path=[current]
            while path[-1]!=start:
                path.append(parent[path[-1]])
            path.reverse()
            return {'reachable':True,'path':[list(p) for p in path],'steps':len(path)-1,'expanded':expanded}
        for action in range(4):
            target=move(grid,current,action)
            next_cost=cost+1
            if next_cost<costs.get(target,float('inf')):
                costs[target]=next_cost
                parent[target]=current
                heuristic=abs(target[0]-goal[0])+abs(target[1]-goal[1])
                heapq.heappush(queue,(next_cost+heuristic,next_cost,target))
    return {'reachable':False,'path':[],'steps':None,'expanded':expanded}


def train_grid(state,grid,episodes=200,seed=0):
    start,goal=grid_spec(grid)
    if type(episodes) is not int or not 1<=episodes<=2000 or type(seed) is not int:
        raise ValueError('episodes must be 1–2000 and seed an integer')
    key='grid:'+hashlib.sha256(json.dumps(grid).encode()).hexdigest()[:20]
    model=state.setdefault('procedural_memory',{}).setdefault(key,{'q':{},'episodes':0,'grid':grid})
    q=model['q']
    rng=random.Random(seed)
    parameters=state['brain']['parameters']
    alpha=0.1+0.4*parameters['learning_rate']
    gamma=0.8+0.19*parameters['discount']
    def row(p):
        return q.setdefault(f'{p[0]},{p[1]}',[0.0]*4)
    def greedy(p):
        values=row(p)
        best=max(values)
        return rng.choice([i for i,v in enumerate(values) if v==best])
    def evaluate():
        p=start
        visited=[]
        for _ in range(min(300,len(grid)*len(grid[0])*4)):
            visited.append(list(p))
            if p==goal:
                return {'success':True,'path':visited}
            p=move(grid,p,greedy(p))
        return {'success':False,'path':visited}
    before=evaluate()
    wins=0
    for episode in range(episodes):
        p=start
        epsilon=max(0.05,0.8*(1-episode/max(1,episodes)))
        for _ in range(min(300,len(grid)*len(grid[0])*4)):
            action=rng.randrange(4) if rng.random()<epsilon else greedy(p)
            target=move(grid,p,action)
            done=target==goal
            reward=1.0 if done else (-0.08 if target==p else -0.02)
            error=reward+(0 if done else gamma*max(row(target)))-row(p)[action]
            row(p)[action]+=alpha*error
            p=target
            if done:
                wins+=1
                break
    model['episodes']+=episodes
    after=evaluate()
    return {'policy_id':key,'episodes':episodes,'total_episodes':model['episodes'],
            'training_success_rate':wins/episodes,'before':before,'after':after,
            'algorithm':'tabular Q-learning','scope':'performance on this supplied grid; no general human ability claim'}


def train_gym(state,episodes=500,seed=0):
    if type(episodes) is not int or not 1<=episodes<=3000 or type(seed) is not int:
        raise ValueError('episodes must be 1–3000 and seed an integer')
    try:
        import gymnasium as gym
    except ImportError as exc:
        raise ValueError('Gymnasium is optional; install requirements-optional.txt in the local venv') from exc
    env=gym.make('FrozenLake-v1',is_slippery=False)
    model=state.setdefault('procedural_memory',{}).setdefault('gym:FrozenLake-v1:deterministic',{'q':[[0.0]*4 for _ in range(16)],'episodes':0})
    q=model['q']
    rng=random.Random(seed)
    params=state['brain']['parameters']
    alpha=0.1+params['learning_rate']*0.4
    gamma=0.85+params['discount']*0.14
    def greedy(position):
        best=max(q[position])
        return rng.choice([i for i,v in enumerate(q[position]) if v==best])
    def evaluate(rounds):
        wins=0
        for i in range(rounds):
            position,_=env.reset(seed=seed+10_000+i)
            for _ in range(100):
                position,reward,terminated,truncated,_=env.step(greedy(position))
                if terminated or truncated:
                    wins+=int(reward>0)
                    break
        return wins/rounds
    try:
        before=evaluate(30)
        for episode in range(episodes):
            position,_=env.reset(seed=seed+episode)
            epsilon=max(0.05,0.9*(1-episode/max(1,episodes)))
            for _ in range(100):
                action=rng.randrange(4) if rng.random()<epsilon else greedy(position)
                target,reward,terminated,truncated,_=env.step(action)
                # Small step penalty is explicitly a training reward modification.
                signal=reward if reward else (-0.5 if terminated else -0.01)
                bootstrap=0 if terminated else gamma*max(q[target])
                q[position][action]+=alpha*(signal+bootstrap-q[position][action])
                position=target
                if terminated or truncated:
                    break
        model['episodes']+=episodes
        after=evaluate(30)
        return {'environment':'FrozenLake-v1','gymnasium_version':gym.__version__,
                'is_slippery':False,'episodes':episodes,'before_success_rate':before,
                'after_success_rate':after,'evaluation_episodes':30,
                'training_reward':'goal +1, hole -0.5, step -0.01',
                'evaluation_reward':'native Gymnasium goal reward',
                'policy_id':'gym:FrozenLake-v1:deterministic'}
    finally:
        env.close()
