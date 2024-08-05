import VectorAnalyzer as va
import VectorPlotter as vp
import matplotlib.pyplot as plt
import numpy as np
import Constants as const
#from mpl_toolkits import mplot3d

NTHLINE = const.NTHLINE

# This file reads an stl file and produces a list of all
# vertexes contained in the mesh. Excludes duplicate vertexes
# (x, y, z) : # times counted

def readVectorsSTL(fileName, file = None):
    if file == None:
        file = open(fileName)   # default on "rt" (read, text)
    map = {}                # empty dictionary
    # Header
    line = file.readline()

    ### Continue reading shapes
    while line != '':
        # normal and outer loop
        file.readline()
        # If we passed endsolid marker
        if file.readline() == '':
            break
        point1 = readVertex(file)
        point2 = readVertex(file)
        point3 = readVertex(file)
        triangle = (point1, point2, point3)
        ## print(triangle)
        # Create a vertex to hold and insert into map
        # Tuples could work for x,y,z!
        map[point1] = map.get(point1, 0) + 1
        map[point2] = map.get(point2, 0) + 1
        map[point3] = map.get(point3, 0) + 1

        # Endloop and Endfacet
        file.readline()
        line = file.readline()
    file.close()
    return map

def readVectorsWRL(fileName):
    file = open(fileName)
    map = {}

    line = file.readline()
    i = 0
    while  i < 200:
        # Checking for correct line
        if (line == "point"):
            line = file.readline().strip()
            # Check if next line is bracket
            if (line == "["):
                break
            continue
        if (line == "point	["):
            break
        line = file.readline().strip()
        # print(line)
        i += 1
    if (i == 200):
        print("Points not found")
        quit()
    
    print("\nPoints")
    i = 0
    line = file.readline().strip().rstrip(',')
    while line != ']':
        # handling any comments
        if "#" in line:
            line = file.readline().strip().rstrip(',')
            continue

        points = readLineWRL(line)
        for point in points:
            map[point] = map.get(point, 0) + 1
            i += 1
        line = file.readline().strip().rstrip(',')
        #if (i % NTHLINE == 0):
        #    print(f"Point: {point}")
    file.close()
    return map

def sciToFloat(coord):
    idx = coord.find("e")
    ## Apparently e not necessary
    if idx == -1:
        # print("Error parsing vectors. Could not find exponent marker 'e+/-'")
        # Just return the number without sci conversion
        return float(coord)
    numVal = float(coord[:idx]) #Convert to a float
    exp = int(coord[idx + 1:])  #Convert to an int
    numVal *= 10 ** exp
    # print("Exponent: ", exp)
    return numVal

# Returns a 3-tuple using a 3-array of 
# numbers in scientific notation
def readVertex(file):
    # Read in points, strip excess
    point = file.readline().strip()[6:].split()
    #print(point)
    # convert to numerical value
    x = sciToFloat(point[0])
    y = sciToFloat(point[1])
    z = sciToFloat(point[2])
    return (x, y, z)

# Takes in a line of comma separated points
# and returns a list of points
def readLineWRL(line):
    pointStrings = line.split(", ")
    points = []
    for pointStr in pointStrings:
        try:
            points.append(readVertexWRL(pointStr))
        except:
            print(f"Couldn't read: {pointStr}")
    return points

# Returns a 3-tuple of coords
def readVertexWRL(pointString):
    point = pointString.split()
    x = float(point[0])
    y = float(point[1])
    z = float(point[2])
    return (x, y, z)

# def storeVector
# def open # loads the file in
# def run
# def save
