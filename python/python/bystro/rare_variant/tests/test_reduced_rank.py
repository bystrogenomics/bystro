import unittest
import numpy as np
from bystro.rare_variant.reduced_rank import ReducedRankML  

class TestReducedRankML(unittest.TestCase):

    def setUp(self):
        # Initialize data and model parameters for testing
        np.random.seed(42)
        self.X = np.random.rand(100, 10)
        self.Y = np.random.randint(2, size=(100, 5))
        self.model = ReducedRankML(lamb_sparsity=1.0, lamb_rank=1.0)

    def test_fit(self):
        # Test the fit method
        self.model.fit(self.X, self.Y)
        
        # Check if the model has been trained
        self.assertTrue(hasattr(self.model, 'B_'))
        self.assertTrue(hasattr(self.model, 'alpha_'))

    def test_predict(self):
        # Test the predict method
        self.model.fit(self.X, self.Y)
        predictions = self.model.predict(self.X)

        # Check if the predictions have the correct shape
        self.assertEqual(predictions.shape, (self.X.shape[0], self.Y.shape[1]))

    def test_decision_function(self):
        # Test the decision_function method
        self.model.fit(self.X, self.Y)
        scores = self.model.decision_function(self.X)

        # Check if the scores have the correct shape
        self.assertEqual(scores.shape, (self.X.shape[0], self.Y.shape[1]))

if __name__ == '__main__':
    unittest.main()

