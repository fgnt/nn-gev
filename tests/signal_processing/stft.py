import unittest

import numpy as np

from fgnt.signal_processing import stft


class TestStft(unittest.TestCase):
    def testSyntheticNoise_StftShouldNotFail(self):
        noise = np.random.randn(6, 61538)
        spectrogram = stft(noise)


if __name__ == '__main__':
    unittest.main()
