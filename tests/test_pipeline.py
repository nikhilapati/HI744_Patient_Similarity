import unittest
from src.preprocessing import preprocess
from src.utils import parse_relevance
from src.evaluation import evaluate

class PipelineTests(unittest.TestCase):
    def test_relevance(self):
        self.assertEqual(parse_relevance("{'x': 1, 'y': 2, 'z': 0}"), {'x','y'})
    def test_clean(self):
        self.assertIn('patient', preprocess('The patients, PATIENTS!'))
    def test_metrics(self):
        p=[{'patient_uid':'a','patient_id':1,'patient':'x','similar_patients':"{'b': 2}"},
           {'patient_uid':'b','patient_id':2,'patient':'y','similar_patients':'{}'}]
        r={'a':{'tfidf':['b'],'bm25':['b']},'b':{'tfidf':['a'],'bm25':['a']}}
        metrics,summary=evaluate(p,r)
        self.assertEqual(metrics['a']['tfidf']['precision_5'],.2)
        self.assertEqual(metrics['a']['tfidf']['recall_5'],1)
        self.assertIsNone(metrics['b']['tfidf']['recall_5'])
        self.assertEqual(summary['tfidf']['recall_excluded'],1)
if __name__=='__main__': unittest.main()
