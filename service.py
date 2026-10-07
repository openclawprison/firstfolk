import json
import random
import threading
import uuid

import genome
from engine import initial_state, retrieve, decide, transition, clamp, SKILLS
from store import Store, decode, public
import mind
import brain
import capabilities
import transfer
import skills
import social_model
import personality
import human_dynamics
import actr_memory
import embodied
import embodied_learning
import human_validation
from pathlib import Path


class Service:
    def __init__(self, path):
        self.store = Store(path)
        self.lock = threading.RLock()
        # Upgrade existing individuals without changing their genomes or biography.
        with self.store.connection() as db:
            for row in db.execute('SELECT id,state,genome FROM agents').fetchall():
                state = json.loads(row['state'])
                if 'brain' not in state or 'world_model' not in state or 'identity_profile' not in state or state.get('personality',{}).get('schema_version',0)<human_dynamics.VERSION:
                    state = mind.upgrade(state,row['genome'])
                    personality.initialize(state,json.loads(db.execute('SELECT traits FROM agents WHERE id=?',(row['id'],)).fetchone()[0]))
                    db.execute('UPDATE agents SET state=? WHERE id=?',(json.dumps(state),row['id']))

    @staticmethod
    def agent(db, agent_id):
        return decode(db.execute('SELECT * FROM agents WHERE id=?', (agent_id,)).fetchone())

    def get(self, agent_id):
        with self.store.connection() as db:
            return public(self.agent(db, agent_id))

    def list(self):
        with self.store.connection() as db:
            return [public(decode(r)) for r in db.execute('SELECT * FROM agents ORDER BY birth_tick,id')]

    def create(self, name, loci=3_000_000, seed=0, parents=None, mutation_rate=0.00001):
        if not isinstance(name, str) or not 1 <= len(name.strip()) <= 100:
            raise ValueError('name must contain 1–100 characters')
        if type(loci) is not int or not 64 <= loci <= 10_000_000:
            raise ValueError('loci must be an integer from 64 to 10000000')
        if type(seed) is not int:
            raise ValueError('seed must be an integer')
        if not isinstance(mutation_rate, (int, float)) or not 0 <= mutation_rate <= 0.01:
            raise ValueError('mutation_rate must be between 0 and 0.01')
        parents = [] if parents is None else parents
        if not isinstance(parents, list) or len(parents) not in (0, 2) or any(not isinstance(p, str) for p in parents):
            raise ValueError('parents must be an array of two agent IDs')
        if len(parents) == 2 and parents[0] == parents[1]:
            raise ValueError('Choose two distinct parents')
        with self.lock, self.store.connection() as db:
            tick = int(db.execute("SELECT value FROM meta WHERE key='tick'").fetchone()[0])
            lineage = {'seed': seed}
            if parents:
                a, b = [self.agent(db, p) for p in parents]
                dna, history = genome.reproduce(a['genome'], b['genome'], seed, mutation_rate)
                lineage.update(history)
            else:
                dna = genome.founder(loci, seed)
            traits = genome.express(dna)
            state=mind.upgrade(initial_state(traits),dna)
            personality.initialize(state,traits)
            agent_id = str(uuid.uuid4())
            db.execute('INSERT INTO agents VALUES(?,?,?,?,?,?,?,?,?,?)',
                       (agent_id, name.strip(), dna, genome.identity(dna), genome.VERSION,
                        json.dumps(traits), json.dumps(state), json.dumps(parents), tick, json.dumps(lineage)))
            for other in db.execute('SELECT id,traits FROM agents WHERE id<>?', (agent_id,)).fetchall():
                db.execute('INSERT INTO relationships VALUES(?,?,?,0)', (agent_id, other['id'], traits['trust_prior']))
                db.execute('INSERT INTO relationships VALUES(?,?,?,0)', (other['id'], agent_id, json.loads(other['traits'])['trust_prior']))
            return public(self.agent(db, agent_id))

    def memories(self, agent_id, query='experience', limit=8):
        with self.lock,self.store.connection() as db:
            a = self.agent(db, agent_id)
            if 'hippocampus' in a['state']['brain']['lesions']:
                return []
            tick = int(db.execute("SELECT value FROM meta WHERE key='tick'").fetchone()[0])
            rows = [dict(r) for r in db.execute('SELECT * FROM memories WHERE agent_id=? ORDER BY id DESC LIMIT 2000', (agent_id,))]
            if a['state'].get('memory_retrieval_mode')=='actr':
                from engine import ACTIONS
                cue=next((action for action in ACTIONS if action in query.lower().split()),None)
                candidates=retrieve(rows,query,a['traits'],tick,limit=64,state=a['state'])
                result=actr_memory.rank(a['state'],candidates,tick,limit,action=cue)
                db.execute('UPDATE agents SET state=? WHERE id=?',(json.dumps(a['state']),agent_id))
                return result['memories']
            return retrieve(rows, query, a['traits'], tick, limit,state=a['state'])

    def experience(self, agent_id, text, reward=0, action=None):
        if not isinstance(text, str) or not 1 <= len(text) <= 8000:
            raise ValueError('text must contain 1–8000 characters')
        if type(reward) not in (int, float) or not -1 <= reward <= 1:
            raise ValueError('reward must be a finite number between -1 and 1')
        from engine import ACTIONS
        if action is not None and action not in ACTIONS:
            raise ValueError('Unknown action')
        with self.lock, self.store.connection() as db:
            a = self.agent(db, agent_id)
            tick = int(db.execute("SELECT value FROM meta WHERE key='tick'").fetchone()[0])
            if action:
                values = a['state']['learned_values']
                alpha = 0.05 + 0.25 * a['traits']['plasticity']
                values[action] += alpha * (reward - values[action])
                db.execute('UPDATE agents SET state=? WHERE id=?', (json.dumps(a['state']), agent_id))
            cursor = db.execute('INSERT INTO memories(agent_id,tick,text,action,reward,salience,source) VALUES(?,?,?,?,?,?,?)',
                               (agent_id, tick, text, action, reward, max(0.1, abs(reward)), 'external'))
            mind.attend(a['state'],{'text':text,'tick':tick,'memory_id':cursor.lastrowid,'priority':max(0.1,abs(reward))})
            db.execute('UPDATE agents SET state=? WHERE id=?',(json.dumps(a['state']),agent_id))
            return {'memory_id': cursor.lastrowid}

    def relationships(self, agent_id):
        with self.store.connection() as db:
            self.agent(db, agent_id)
            return [dict(r) for r in db.execute('SELECT * FROM relationships WHERE agent_id=?', (agent_id,))]

    def events(self, agent_id, limit=50):
        with self.store.connection() as db:
            self.agent(db, agent_id)
            return [dict(r) | {'payload': json.loads(r['payload'])} for r in db.execute(
                'SELECT * FROM events WHERE agent_id=? ORDER BY id DESC LIMIT ?', (agent_id, limit))]

    def tick(self, steps=1, seed=0):
        if type(steps) is not int or not 1 <= steps <= 100:
            raise ValueError('steps must be an integer from 1 to 100')
        if type(seed) is not int:
            raise ValueError('seed must be an integer')
        emitted = []
        with self.lock, self.store.connection() as db:
            # One atomic transaction; all agents decide from the same pre-tick snapshot.
            for _ in range(steps):
                tick = int(db.execute("SELECT value FROM meta WHERE key='tick'").fetchone()[0]) + 1
                agents = [decode(r) for r in db.execute('SELECT * FROM agents ORDER BY id')]
                lookup = {a['id']: a for a in agents}
                resource_transfers = {}
                social_deliveries = []
                for a in agents:
                    rng = random.Random(f"{seed}:{tick}:{a['genome_hash']}:{a['id']}")
                    rows = [dict(r) for r in db.execute('SELECT * FROM memories WHERE agent_id=? ORDER BY id DESC LIMIT 2000', (a['id'],))]
                    if a['state'].get('memory_retrieval_mode')=='actr' and 'hippocampus' not in a['state']['brain']['lesions']:
                        memories=actr_memory.rank(a['state'],rows,tick)['memories']
                    else:
                        memories = retrieve(rows, 'explore practice socialize help rest', a['traits'], tick,state=a['state'])
                    if 'hippocampus' in a['state']['brain']['lesions']:
                        memories=[]
                    relations = [dict(r) for r in db.execute('SELECT * FROM relationships WHERE agent_id=?', (a['id'],))]
                    decision = decide(a['traits'], a['state'], memories, relations, rng)
                    # Record the expectation before observing the outcome.
                    a['state']['metacognition']['last_prediction'] = a['state']['learned_values'][decision['action']]
                    partner = lookup.get(decision['partner_id'])
                    state, outcome = transition(a['traits'], a['state'], decision, rng, partner['traits'] if partner else None)
                    if outcome['action'] == 'help':
                        spent = max(0, a['state']['resources'] - state['resources'])
                        resource_transfers[partner['id']] = resource_transfers.get(partner['id'], 0) + spent
                    db.execute('UPDATE agents SET state=? WHERE id=?', (json.dumps(state), a['id']))
                    if partner:
                        current = next(r for r in relations if r['other_id'] == partner['id'])
                        db.execute('UPDATE relationships SET trust=?,interactions=interactions+1 WHERE agent_id=? AND other_id=?',
                                   (clamp(current['trust'] + outcome['trust_delta']), a['id'], partner['id']))
                        if outcome['action']=='repair':
                            social_deliveries.append((partner['id'],f'{a["id"]} attempted relationship repair.'))
                        # Teaching transfers a bounded amount of skill, separately from heredity.
                        if outcome['reward'] > 0:
                            skill = decision['skill']
                            difference = max(0, partner['state']['skills'][skill] - state['skills'][skill])
                            state['skills'][skill] = clamp(state['skills'][skill] + difference * a['traits']['plasticity'] * 0.1)
                            db.execute('UPDATE agents SET state=? WHERE id=?', (json.dumps(state), a['id']))
                    cursor = db.execute('INSERT INTO memories(agent_id,tick,text,action,reward,salience,source) VALUES(?,?,?,?,?,?,?)',
                               (a['id'], tick, outcome['text'], outcome['action'], outcome['reward'], outcome['salience'], 'simulation'))
                    mind.learn(state,a['traits'],outcome,cursor.lastrowid,tick,previous_state=a['state'])
                    if outcome['action'] in ('sleep','reflect','plan'):
                        rows.append({'id':cursor.lastrowid,'text':outcome['text'],'reward':outcome['reward'],
                                     'action':outcome['action'],'salience':outcome['salience'],'source':'simulation'})
                        outcome['consolidation'] = mind.consolidate(state,rows)
                    db.execute('UPDATE agents SET state=? WHERE id=?',(json.dumps(state),a['id']))
                    event = {'decision': decision, 'outcome': outcome, 'state_before': a['state'], 'state_after': state}
                    db.execute('INSERT INTO events(tick,agent_id,payload) VALUES(?,?,?)', (tick, a['id'], json.dumps(event)))
                    emitted.append({'tick': tick, 'agent_id': a['id'], **event})
                for recipient, amount in resource_transfers.items():
                    a = self.agent(db, recipient)
                    a['state']['resources'] += amount
                    db.execute('UPDATE agents SET state=? WHERE id=?', (json.dumps(a['state']), recipient))
                    db.execute('INSERT INTO memories(agent_id,tick,text,action,reward,salience,source) VALUES(?,?,?,?,?,?,?)',
                               (recipient, tick, f'Received {amount:.3f} resources from peers.', None, 0.3, 0.3, 'simulation'))
                for recipient,text in social_deliveries:
                    db.execute('INSERT INTO memories(agent_id,tick,text,action,reward,salience,source) VALUES(?,?,?,?,?,?,?)',
                               (recipient,tick,text,None,0,0.5,'simulation_social_event'))
                # Final snapshots include incoming resource transfers as well as
                # each agent's own action, so event history matches stored state.
                for event in emitted[-len(agents):] if agents else []:
                    event['state_after'] = self.agent(db, event['agent_id'])['state']
                    db.execute('UPDATE events SET payload=? WHERE tick=? AND agent_id=?',
                               (json.dumps({k: v for k, v in event.items() if k not in ('tick', 'agent_id')}), tick, event['agent_id']))
                db.execute("UPDATE meta SET value=? WHERE key='tick'", (str(tick),))
        return {'tick': tick, 'events': emitted}

    def teach(self, teacher_id, student_id, skill):
        if skill not in SKILLS or teacher_id == student_id:
            raise ValueError('Choose different agents and a supported skill')
        with self.lock, self.store.connection() as db:
            teacher, student = self.agent(db, teacher_id), self.agent(db, student_id)
            before = student['state']['skills'][skill]
            gain = max(0, teacher['state']['skills'][skill] - before) * student['traits']['plasticity'] * 0.2
            student['state']['skills'][skill] = clamp(before + gain)
            db.execute('UPDATE agents SET state=? WHERE id=?', (json.dumps(student['state']), student_id))
            tick = int(db.execute("SELECT value FROM meta WHERE key='tick'").fetchone()[0])
            db.execute('INSERT INTO memories(agent_id,tick,text,action,reward,salience,source) VALUES(?,?,?,?,?,?,?)',
                       (student_id, tick, f'Learned {skill} from {teacher_id}; gain {gain:.4f}.', None, gain, 0.4, 'teaching'))
            return {'skill': skill, 'before': before, 'after': student['state']['skills'][skill]}

    def brain_map(self, agent_id=None):
        return brain.mapping(self.get(agent_id)['state'] if agent_id else None)

    def inventory(self,agent_id=None):
        return personality.inventory(self.get(agent_id)['state'] if agent_id else None)

    def sources(self):
        return json.loads((Path(__file__).parent/'research'/'sources.json').read_text())

    def observe(self,agent_id,text,claims=None,confidence=0.8,source='user_observation'):
        if not isinstance(text,str) or not 1<=len(text)<=8000:
            raise ValueError('text must contain 1–8000 characters')
        if type(confidence) not in (int,float) or not 0<confidence<=1:
            raise ValueError('confidence must be in (0,1]')
        if not isinstance(source,str) or not 1<=len(source)<=200:
            raise ValueError('source must contain 1–200 characters')
        claims=[] if claims is None else claims
        if not isinstance(claims,list) or len(claims)>20:
            raise ValueError('claims must contain at most 20 entries')
        for claim in claims:
            if not isinstance(claim,dict) or set(claim)!=set(('subject','predicate','value')) or any(not isinstance(v,str) or not 1<=len(v)<=200 for v in claim.values()):
                raise ValueError('Claims require subject, predicate, value strings of 1–200 characters')
        with self.lock,self.store.connection() as db:
            a=self.agent(db,agent_id)
            s=a['state']
            if len(s['knowledge'])+len(claims)>500:
                raise ValueError('Knowledge capacity reached for this prototype')
            tick=int(db.execute("SELECT value FROM meta WHERE key='tick'").fetchone()[0])
            cursor=db.execute('INSERT INTO memories(agent_id,tick,text,action,reward,salience,source) VALUES(?,?,?,?,?,?,?)',
                              (agent_id,tick,text,None,0,confidence,source))
            for claim in claims:
                mind.evidence(s,**claim,confidence=confidence,source=source,memory_id=cursor.lastrowid)
            mind.attend(s,{'text':text,'tick':tick,'memory_id':cursor.lastrowid,'priority':confidence})
            activity=brain.cycle(s,sensory=confidence,conflict=float(any(k['conflicted'] for k in s['knowledge'].values())))
            db.execute('UPDATE agents SET state=? WHERE id=?',(json.dumps(s),agent_id))
            return {'memory_id':cursor.lastrowid,'claims_recorded':len(claims),'brain_activity':activity}

    def goal(self,agent_id,metric,target,direction='at_least',priority=0.5):
        if type(target) not in (int,float) or not 0<=target<=1e6:
            raise ValueError('target must be finite and between 0 and 1000000')
        if direction not in ('at_least','at_most') or type(priority) not in (int,float) or not 0<=priority<=1:
            raise ValueError('Invalid direction or priority')
        with self.lock,self.store.connection() as db:
            a=self.agent(db,agent_id)
            progress=mind.goal_value(a['state'],metric)
            if len(a['state']['goals'])>=100:
                raise ValueError('Goal capacity reached')
            g={'id':str(uuid.uuid4()),'metric':metric,'target':target,'direction':direction,
               'priority':priority,'progress':progress,'status':'active'}
            if (progress>=target if direction=='at_least' else progress<=target):
                g['status']='achieved'
            a['state']['goals'].append(g)
            brain.cycle(a['state'])
            db.execute('UPDATE agents SET state=? WHERE id=?',(json.dumps(a['state']),agent_id))
            return g

    def consolidate(self,agent_id):
        with self.lock,self.store.connection() as db:
            a=self.agent(db,agent_id)
            rows=[dict(r) for r in db.execute('SELECT * FROM memories WHERE agent_id=? ORDER BY id DESC LIMIT 2000',(agent_id,))]
            result=mind.consolidate(a['state'],rows)
            brain.cycle(a['state'],retrieval=min(1,result['replayed']/10))
            db.execute('UPDATE agents SET state=? WHERE id=?',(json.dumps(a['state']),agent_id))
            return result

    def lesion(self,agent_id,regions):
        if not isinstance(regions,list) or len(regions)>len(brain.REGIONS) or any(r not in brain.REGIONS for r in regions):
            raise ValueError('regions must be a list of known functional module IDs')
        with self.lock,self.store.connection() as db:
            a=self.agent(db,agent_id)
            a['state']['brain']['lesions']=list(set(regions))
            if 'working_memory' in regions:
                a['state']['workspace']=[]
            brain.cycle(a['state'])
            db.execute('UPDATE agents SET state=? WHERE id=?',(json.dumps(a['state']),agent_id))
            return {'disabled_modules':regions,'scope':'software ablation; not a model of clinical brain injury'}

    def solve(self,agent_id,kind,**arguments):
        self.get(agent_id)
        if kind=='arithmetic':
            result=capabilities.calculate(**arguments)
        elif kind=='route':
            result=capabilities.plan_route(**arguments)
        elif kind=='skill':
            result=self.cognition_operation(agent_id,'skill_predict',**arguments)
        elif kind=='belief':
            result=self.cognition_operation(agent_id,'perspective_query',**arguments)
        elif kind=='participant':
            result=self.cognition_operation(agent_id,'participant_predict',**arguments)
        else:
            raise ValueError('kind must be arithmetic, route, skill, belief or participant')
        self.observe(agent_id,json.dumps({'task':kind,'arguments':arguments,'result':result}),
                     source='verified_tool' if kind in ('arithmetic','route') else 'computed_prediction',
                     confidence=1 if kind in ('arithmetic','route') else 0.5)
        return result

    def train(self,agent_id,kind='grid',**arguments):
        with self.lock,self.store.connection() as db:
            a=self.agent(db,agent_id)
            if 'striatum' in a['state']['brain']['lesions']:
                raise ValueError('Reinforcement learning module is disabled')
            if kind=='grid':
                result=capabilities.train_grid(a['state'],**arguments)
            elif kind=='gym':
                result=capabilities.train_gym(a['state'],**arguments)
            else:
                raise ValueError('kind must be grid or gym')
            db.execute('UPDATE agents SET state=? WHERE id=?',(json.dumps(a['state']),agent_id))
            tick=int(db.execute("SELECT value FROM meta WHERE key='tick'").fetchone()[0])
            db.execute('INSERT INTO memories(agent_id,tick,text,action,reward,salience,source) VALUES(?,?,?,?,?,?,?)',
                       (agent_id,tick,json.dumps(result),None,0,0.8,'evaluated_learning'))
            return result

    def message(self,sender_id,recipient_id,text):
        if sender_id==recipient_id or not isinstance(text,str) or not 1<=len(text)<=8000:
            raise ValueError('Choose distinct agents and text of 1–8000 characters')
        with self.lock,self.store.connection() as db:
            sender,recipient=self.agent(db,sender_id),self.agent(db,recipient_id)
            tick=int(db.execute("SELECT value FROM meta WHERE key='tick'").fetchone()[0])
            ids=[]
            for a,description in ((sender,f'Sent to {recipient_id}: {text}'),(recipient,f'Received from {sender_id}: {text}')):
                cursor=db.execute('INSERT INTO memories(agent_id,tick,text,action,reward,salience,source) VALUES(?,?,?,?,?,?,?)',
                                  (a['id'],tick,description,None,0,0.5,'agent_message'))
                mind.attend(a['state'],{'text':description,'tick':tick,'memory_id':cursor.lastrowid,'priority':0.5})
                brain.cycle(a['state'],sensory=0.5)
                db.execute('UPDATE agents SET state=? WHERE id=?',(json.dumps(a['state']),a['id']))
                ids.append(cursor.lastrowid)
            return {'sender_memory_id':ids[0],'recipient_memory_id':ids[1],'delivery':'internal simulation only'}

    def cognition_operation(self,agent_id,operation,**arguments):
        """Atomic, persisted learning and cognition operations."""
        with self.lock,self.store.connection() as db:
            a=self.agent(db,agent_id)
            state=a['state']
            if operation=='memory_mode':
                result=actr_memory.configure(state,**arguments)
            elif operation=='actr_recall':
                if 'hippocampus' in state['brain']['lesions']:
                    raise ValueError('Episodic retrieval module is disabled')
                tick=int(db.execute("SELECT value FROM meta WHERE key='tick'").fetchone()[0])
                rows=[dict(r) for r in db.execute('SELECT * FROM memories WHERE agent_id=? ORDER BY id DESC LIMIT 2000',(agent_id,))]
                result=actr_memory.rank(state,rows,tick,**arguments)
            elif operation=='embodied_evaluate':
                result=embodied.evaluate(state,**arguments)
            elif operation=='embodied_train':
                if 'striatum' in state['brain']['lesions']:raise ValueError('Reinforcement learning module is disabled')
                result=embodied_learning.train(state,**arguments)
            elif operation=='embodied_policy_evaluate':
                result=embodied_learning.evaluate(state,**arguments)
            elif operation=='participant_train':
                result=human_validation.train(state,**arguments)
            elif operation=='participant_evaluate':
                result=human_validation.evaluate(state,**arguments)
            elif operation=='participant_predict':
                result=human_validation.predict(state,**arguments)
            elif operation=='appraise':
                result=human_dynamics.appraise(state,**arguments)
            elif operation=='personality_configure':
                result=personality.configure(state,**arguments)
            elif operation=='transfer_train':
                result=transfer.train(state,**arguments)
            elif operation=='navigate':
                result=transfer.navigate(transfer.model_for(state),**arguments)
            elif operation=='transfer_evaluate':
                result=transfer.evaluate(state,**arguments)
            elif operation=='skill_train':
                result=skills.train(state,**arguments)
            elif operation=='skill_predict':
                result=skills.predict(state,**arguments)
            elif operation=='skill_evaluate':
                result=skills.evaluate(state,**arguments)
            elif operation=='perspective_event':
                result=social_model.event(state,**arguments)
            elif operation=='perspective_query':
                result=social_model.query(state,**arguments)
            else:
                raise ValueError('Unknown cognition operation')
            db.execute('UPDATE agents SET state=? WHERE id=?',(json.dumps(state),agent_id))
            tick=int(db.execute("SELECT value FROM meta WHERE key='tick'").fetchone()[0])
            # Large evaluation traces are stored separately from prompt context.
            summary=json.dumps({'operation':operation,'result':result})[:8000]
            db.execute('INSERT INTO memories(agent_id,tick,text,action,reward,salience,source) VALUES(?,?,?,?,?,?,?)',
                       (agent_id,tick,summary,None,0,0.7,'cognitive_operation'))
            return result

    def profile(self,agent_id,label,biography=None,values=None,preferences=None,style_examples=None,source='supplied_profile'):
        if not isinstance(label,str) or not 1<=len(label)<=100 or not isinstance(source,str) or not 1<=len(source)<=200:
            raise ValueError('Profile label/source must be short strings')
        fields={key:[] if value is None else value for key,value in
                {'biography':biography,'values':values,'preferences':preferences,'style_examples':style_examples}.items()}
        for key,items in fields.items():
            if not isinstance(items,list) or len(items)>40 or any(not isinstance(v,str) or not 1<=len(v)<=1000 for v in items):
                raise ValueError(f'{key} must be up to 40 strings of 1–1000 characters')
        with self.lock,self.store.connection() as db:
            a=self.agent(db,agent_id)
            tick=int(db.execute("SELECT value FROM meta WHERE key='tick'").fetchone()[0])
            cursor=db.execute('INSERT INTO memories(agent_id,tick,text,action,reward,salience,source) VALUES(?,?,?,?,?,?,?)',
                              (agent_id,tick,json.dumps({'label':label,**fields}),None,0,1,'identity_profile'))
            profile={'label':label,**fields,'source':source,'evidence_memory_id':cursor.lastrowid,
                     'representation':'simulation based on supplied records; completeness and fidelity unverified'}
            a['state']['identity_profile']=profile
            db.execute('UPDATE agents SET state=? WHERE id=?',(json.dumps(a['state']),agent_id))
            return profile

    def teach_skill(self,teacher_id,student_id,name,epochs=30,seed=0):
        if teacher_id==student_id:
            raise ValueError('Choose distinct teacher and student')
        with self.lock,self.store.connection() as db:
            teacher,student=self.agent(db,teacher_id),self.agent(db,student_id)
            model=teacher['state'].get('learned_skills',{}).get(name)
            if model is None:
                raise LookupError('Teacher does not have this learned skill')
            examples=model.get('replay_examples',[])[:128]
            if not examples:
                raise ValueError('Teacher has no retained demonstrations')
            result=skills.train(student['state'],name,examples,epochs=epochs,seed=seed)
            db.execute('UPDATE agents SET state=? WHERE id=?',(json.dumps(student['state']),student_id))
            tick=int(db.execute("SELECT value FROM meta WHERE key='tick'").fetchone()[0])
            db.execute('INSERT INTO memories(agent_id,tick,text,action,reward,salience,source) VALUES(?,?,?,?,?,?,?)',
                       (student_id,tick,json.dumps({'teacher_id':teacher_id,'skill':name,'examples':len(examples)}),None,0,0.8,'skill_teaching'))
            return {**result,'teacher_id':teacher_id,'student_id':student_id,
                    'mechanism':'supervised rehearsal of teacher-retained demonstrations; no DNA or biography copying'}
