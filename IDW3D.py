import numpy as np
import math


def __Interpolation(Array):
    Content = []
    Rows, Columns = Array.shape
    for i in range(Rows):
        for j in range(Columns):
            if(Array[i, j] != np.nan):
                Content.append([i, j, Array[i, j]])
            else:
                continue
    return Content


def __distanceMap(Array, Content, exp=2):
    Rows, Columns = Array.shape
    Map = {}
    for item in Content:
        Distance = np.ones((Rows, Columns))
        itemRows, itemColumns, itemValue = item
        for i in range(Rows):
            for j in range(Columns):
                Distance[i, j] = math.sqrt((i - itemRows)**2 +(j - itemColumns)**2)
        mask = Distance == 0
        mx = np.ma.masked_array(Distance, mask=mask)
        inv_distance = np.power(1/mx, exp)
        inv_distance = inv_distance.data - inv_distance.mask
        Map[itemValue] = inv_distance
    return Map


def __ComputingValue(Array, Content, Map):
    Rows, Columns = Array.shape
    idwMap = np.zeros((Rows, Columns))
    for i in range(Rows):
        for j in range(Columns):
            Denominator = 0
            WeightedSum = 0
            for key in Map:
                Data = Map[key]
                Denominator = Denominator + Data[i, j]
                WeightedSum = WeightedSum + Data[i, j] * key
            idwMap[i, j] = WeightedSum / Denominator
    return idwMap


def Interpolation_3D(Array):
    Content = __Interpolation(Array)
    Map = __distanceMap(Array, Content)
    idwMap = __ComputingValue(Array, Content, Map)
    return idwMap
