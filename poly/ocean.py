"""Ocean floor properties from its age: the isochron bands carry the age, this turns it into depth."""
from __future__ import annotations

import numpy as np

# Stein & Stein (1992), GDH1: the floor subsides as the lithosphere cools, as the square root of
# its age at first, then levelling out as the plate reaches its full thickness.
RIDGE_DEPTH_KM = 2.6
YOUNG_KM_PER_SQRT_MYR = .365
OLD_DEPTH_KM = 5.651
OLD_DROP_KM = 2.473
OLD_RATE_PER_MYR = .0278
YOUNG_MYR = 20.


def depth_km(age_myr):
    """Depth of ocean floor below sea level, km, for crust `age_myr` old."""
    age = np.maximum(np.asarray(age_myr, float), 0.)
    young = RIDGE_DEPTH_KM + YOUNG_KM_PER_SQRT_MYR * np.sqrt(age)
    old = OLD_DEPTH_KM - OLD_DROP_KM * np.exp(-OLD_RATE_PER_MYR * age)
    return np.where(age < YOUNG_MYR, young, old)
