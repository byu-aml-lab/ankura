from __future__ import division
import ctypes

def convertToList(ctypesArray, count):
    result = []
    for i in range(count):
        result.append(ctypesArray[i])
    return result

def convertToListOfLists(ctypesArrayOfArrays, sizeList):
    result = []
    for i in range(len(sizeList)):
        result.append(convertToList(ctypesArrayOfArrays[i], sizeList[i]))
    return result

def convertFromDoubleList(pyList):
    result = (ctypes.c_double * len(pyList))()
    for i in range(len(pyList)):
        result[i] = ctypes.c_double(pyList[i])
    return result

def convertFromIntList(pyList):
    result = (ctypes.c_int * len(pyList))()
    for i in range(len(pyList)):
        result[i] = ctypes.c_int(pyList[i])
    return result
