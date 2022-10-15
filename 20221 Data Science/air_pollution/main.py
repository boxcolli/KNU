import pandas as pd
import numpy as np
from tqdm import tqdm
from aqi import aqi


"""
# ------------- #
# Get DataFrame #
# ------------- #
"""
# Read xlsx file
filename = 'airkorea2021/seoul_2021_daily_crop.xlsx'
air_xlsx = pd.read_excel(filename)
print(air_xlsx.dtypes)
print(air_xlsx[air_xlsx.isnull().any(axis=1)])

# Interpolate missing values
air = air_xlsx
air['아황산가스'] = air['아황산가스'].interpolate()
print(air[air.isnull().any(axis=1)])

"""
# --------------- #
# Split by 7 days #
# --------------- #
"""

days = list()
weeks = list()
for i in range(len(air)):
    if i % 7 == 0:
        aqi_values = list()
        for j in range(7):
            sub = air.iloc[i+j, 1:].tolist()
            # arrange order
            # sub : PM10 PM2.5 OZ NO2 CO SO2
            # arr : SO2 CO OZ NO2 PM10 PM25
            arrange = [0] * 6
            arrange[0] = sub[5]
            arrange[1] = sub[4]
            arrange[2] = sub[2]
            arrange[3] = sub[3]
            arrange[4] = sub[0]
            arrange[5] = sub[1]
            value = aqi(arrange)
            aqi_values.append(value)
            days.append(value)
        weeks.append(round(np.mean(aqi_values)))

# for day in days:
#     print(day)
for week in weeks:
    print(week)