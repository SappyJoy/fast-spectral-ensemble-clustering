import unittest
import numpy as np
from fsec.clustering import FSEC

class TestFSEC(unittest.TestCase):
    def test_fsec_fit_predict(self):
        X = np.random.rand(100, 5)
        fsec = FSEC(final_n_clusters=3)
        labels = fsec.fit_predict(X)
        self.assertEqual(len(labels), X.shape[0])
        self.assertTrue(np.unique(labels).size <= 3)

if __name__ == '__main__':
    unittest.main()

