from enum import IntEnum


class Pollution(IntEnum):
    SO2 = 0
    CO = 1
    OZ = 2
    NO2 = 3
    PM10 = 4
    PM25 = 5


class Category(IntEnum):
    Good = 0
    Moderate = 1
    Unhealthy = 2
    Hazardous = 3


category_score = [(0, 50), (51, 100), (101, 250), (251, 500)]


def _get_index_lo_hi(category):
    return category_score[category]


bp_table = [
    [(0, 0.02), (0.021, 0.05), (0.051, 0.15), (0.151, 1)],  # SO2
    [(0, 2), (2.1, 9), (9.1, 15), (15.1, 50)],  # CO
    [(0, 0.03), (0.031, 0.09), (0.091, 0.15), (0.151, 0.6)],  # OZ
    [(0, 0.03), (0.031, 0.06), (0.061, 0.2), (0.201, 2)],  # NO2
    [(0, 30), (31, 80), (81, 150), (151, 600)],  # PM100
    [(0, 15), (16, 35), (36, 75), (76, 500)]  # PM25
]


def _get_category(pollution, value):
    for index, tup in enumerate(bp_table[pollution]):
        hi = tup[1]
        if value <= hi:
            return index
    return Category.Hazardous


def _aqi_equation(i_lo, i_hi, bp_lo, bp_hi, c_p):
    return (i_hi - i_lo) * (c_p - bp_lo) / (bp_hi - bp_lo) + i_lo


def _aqi_single(pollution, c_p):
    category = _get_category(pollution, c_p)
    category = int(category)
    i_lo, i_hi = _get_index_lo_hi(category)
    bp_lo, bp_hi = bp_table[pollution][category]
    return category, _aqi_equation(i_lo, i_hi, bp_lo, bp_hi, c_p)


def aqi(cons):
    # concentrations: [SO4, CO, O3, NO2, PM10, PM25]

    # 1. Get (category, AQI value) for each
    results = list()
    for index, con in enumerate(cons):
        results.append(_aqi_single(index, con))  # index == Pollution type

    # 2. Find max value & Count Unhealthy or Hazardous
    count_bad = 0
    max_aqi = 0
    for index, tup in enumerate(results):
        cat, val = tup
        if cat == Category.Unhealthy or cat == Category.Hazardous:
            count_bad += 1
        if max_aqi < val:
            max_aqi = val

    # round
    max_aqi = round(max_aqi)

    # 3. Final AQI
    if count_bad == 0 or count_bad == 1:
        # Usual case: return max value
        return max_aqi
    elif count_bad == 2:
        # Bad case: return max + 50
        return max_aqi + 50
    else:
        # Very bad case: return max + 75
        return max_aqi + 75


