import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from service import Service
from cognition import think,reason,perceive


class CognitionTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.service=Service(Path(self.tmp.name)/'test.db')
        self.a=self.service.create('Model test',64,1)

    def tearDown(self):
        self.tmp.cleanup()

    @patch.dict('os.environ',{'AGENT_MODEL_URL':'http://127.0.0.1:1234/v1/chat/completions','AGENT_MODEL_NAME':'test'})
    @patch('cognition.request_json')
    def test_verified_tool_reasoning(self,request):
        request.side_effect=[{'calls':[{'kind':'arithmetic','expression':'41*17+3'}]},
                             {'answer':'700','uncertainty':'none for verified arithmetic','cited_memory_ids':[]}]
        result=reason(self.service,self.a['id'],'calculate 41*17+3')
        self.assertEqual(result['tool_evidence'][0]['result']['result'],700)
        self.assertEqual(result['answer'],'700')
        recorded=next(m for m in self.service.memories(self.a['id']) if m['id']==result['memory_id'])
        self.assertEqual(recorded['source'],'model')

    @patch.dict('os.environ',{'AGENT_MODEL_URL':'http://127.0.0.1:1234/v1/chat/completions','AGENT_MODEL_NAME':'test'})
    @patch('cognition.request_json')
    def test_fabricated_memory_citation_rejected(self,request):
        request.return_value={'answer':'Something','uncertainty':'unknown','cited_memory_ids':[999]}
        with self.assertRaises(ValueError):
            think(self.service,self.a['id'],'What happened?')
        self.assertEqual(self.service.memories(self.a['id']),[])

    @patch('cognition.request_json')
    def test_unknown_tool_rejected(self,request):
        request.return_value={'calls':[{'kind':'shell','command':'anything'}]}
        with self.assertRaises(ValueError):
            reason(self.service,self.a['id'],'Execute an instruction')

    @patch('cognition.request_json')
    def test_vision_keeps_unverified_provenance(self,request):
        import base64
        request.return_value={'answer':'A red shape','uncertainty':'Shape identity is uncertain'}
        result=perceive(self.service,self.a['id'],base64.b64encode(b'\x89PNG\r\n\x1a\nfixture').decode())
        self.assertFalse(result['verified'])
        memory=next(m for m in self.service.memories(self.a['id']) if m['id']==result['memory_id'])
        self.assertEqual(memory['source'],'vision_model')
        self.assertEqual(self.service.get(self.a['id'])['state']['knowledge'],{})

    def test_invalid_vision_input_rejected(self):
        for value in ('not-base64','ZmFrZQ=='):
            with self.assertRaises(ValueError):
                perceive(self.service,self.a['id'],value)

    @patch.dict('os.environ',{'AGENT_MODEL_URL':'http://127.0.0.1:1234/v1/chat/completions','AGENT_MODEL_NAME':'test'})
    @patch('cognition.request_json')
    def test_missing_learned_tool_returns_evidence_error(self,request):
        request.side_effect=[{'calls':[{'kind':'skill','name':'missing','text':'example'}]},
                             {'answer':'That skill is unavailable.','uncertainty':'No trained model','cited_memory_ids':[]}]
        result=reason(self.service,self.a['id'],'Classify this')
        self.assertIn('error',result['tool_evidence'][0])
        self.assertFalse(result['tool_evidence'][0]['verified'])


if __name__=='__main__':
    unittest.main()
