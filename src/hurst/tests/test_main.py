# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from eva_hurst import compute_Hc

def test_hurst():
    np.random.seed(42)
    random_increments = np.random.randn(99999)
    series = np.cumsum(random_increments)

    H, c, data = compute_Hc(series)
    assert H<0.6 and H>0.4
