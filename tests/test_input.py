import unittest

import numpy as np
import pandas as pd

from xgb_trainer.input import process_features

from sklearn.preprocessing import MinMaxScaler

class TestInput(unittest.TestCase):

    feat = pd.DataFrame(np.arange(10).reshape(5, 2))

    def test_process_features(self):

        # Build test dataset
        test_feat = pd.DataFrame(process_features(self.feat))

        # Build validation dataset
        scaler = MinMaxScaler()
        scaler.fit(self.feat)
        valid_feat = scaler.transform(self.feat)

        # Test equality
        try:
            np.testing.assert_array_equal(test_feat, valid_feat)
            res = True
        except AssertionError as err:
            res = False
            print (err)
        self.assertTrue(res)


if __name__ == "__main__":
    unittest.main()