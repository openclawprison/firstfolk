import json
import sqlite3
from contextlib import contextmanager

SCHEMA = '''
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
INSERT OR IGNORE INTO meta VALUES('tick', '0');
CREATE TABLE IF NOT EXISTS agents(
 id TEXT PRIMARY KEY, name TEXT NOT NULL, genome BLOB NOT NULL,
 genome_hash TEXT NOT NULL, genome_version TEXT NOT NULL, traits TEXT NOT NULL,
 state TEXT NOT NULL, parents TEXT NOT NULL, birth_tick INTEGER NOT NULL,
 lineage TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS memories(
 id INTEGER PRIMARY KEY AUTOINCREMENT, agent_id TEXT NOT NULL REFERENCES agents(id),
 tick INTEGER NOT NULL, text TEXT NOT NULL, action TEXT, reward REAL NOT NULL,
 salience REAL NOT NULL, source TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS memory_agent ON memories(agent_id, tick DESC);
CREATE TABLE IF NOT EXISTS relationships(
 agent_id TEXT NOT NULL REFERENCES agents(id), other_id TEXT NOT NULL REFERENCES agents(id),
 trust REAL NOT NULL, interactions INTEGER NOT NULL DEFAULT 0,
 PRIMARY KEY(agent_id,other_id));
CREATE TABLE IF NOT EXISTS events(
 id INTEGER PRIMARY KEY AUTOINCREMENT, tick INTEGER NOT NULL,
 agent_id TEXT NOT NULL REFERENCES agents(id), payload TEXT NOT NULL);
'''


class Store:
    def __init__(self, path):
        self.path = str(path)
        with self.connection() as db:
            db.executescript(SCHEMA)

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        db.execute('PRAGMA foreign_keys=ON')
        db.execute('PRAGMA journal_mode=WAL')
        try:
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()


def decode(row):
    if row is None:
        raise LookupError('Agent not found')
    obj = dict(row)
    for key in ('traits', 'state', 'parents', 'lineage'):
        obj[key] = json.loads(obj[key])
    return obj


def public(agent):
    return {k: v for k, v in agent.items() if k != 'genome'} | {'loci': len(agent['genome']) // 2}
