# This file will find the maximum span along the shape,
# slice along the nearest axis and find the center of the
# collection of points, then use linear regression to 
# create an approximate fit for a centerline along the
# channel. This process will occur repeatedly until
# the change falls below the specified threshold
## axis (x, y, z)

# from timeit import default_timer as timer
#from mpl_toolkits import mplot3d

import matplotlib.pyplot as plt
import numpy as np
import Constants as const
import Slice as slc
import math as mth

# Number of slices for computing centroids
#SLICES = 5
# Will print every nth point if num points
# greater than COMPRESSLEN
#NTHTERM = 500
# Limit before the prints will skip
#SKIPLEN = 1000

SLICES = const.SLICES
NTHTERM = const.NTHTERM
SKIPLEN = const.SKIPLEN

MINISLICES = 2

def dprint(string, debug = False):
    if (debug):
        print(string)

# Finds the axis with the greatest span
# and returns the letter of that axis
def greatestSpan(mesh):
    sampleVector = mesh.vectorList[0]                    #Get some vector
    xMin = sampleVector[0]
    xMax = sampleVector[0]
    yMin = sampleVector[1]
    yMax = sampleVector[1]
    zMin = sampleVector[2]
    zMax = sampleVector[2]

    #Find maxes and mins
    for point in mesh.vectorList:
        if point[0] < xMin:
            xMin = point[0]
        elif point[0] > xMax:
            xMax = point[0]
        if point[1] < yMin:
            yMin = point[1]
        elif point[1] > yMax:
            yMax = point[1]
        if point[2] < zMin:
            zMin = point[2]
        elif point[2] > zMax:
            zMax = point[2]
    
    xSpan = xMax - xMin
    ySpan = yMax - yMin
    zSpan = zMax - zMin

    if xSpan >= ySpan and xSpan >= zSpan:
        maxSpan = xSpan
        mesh.greatestSpan = ("x", maxSpan)
        mesh.axisIdx = 0
        return ("x", maxSpan)
    if ySpan > xSpan and ySpan > zSpan:
        maxSpan = ySpan
        mesh.greatestSpan = ("y", maxSpan)
        mesh.axisIdx = 1
        return ("y", maxSpan)
    if zSpan > xSpan and zSpan > ySpan:
        maxSpan = zSpan
        mesh.greatestSpan = ("z", maxSpan)
        mesh.axisIdx = 2
        return ("z", maxSpan)
    else:
        print("Error finding span")
        quit()


def findCentroids(mesh, debug = False):
    #Sort vectors by x, y, or z parameter
    centroids = []
    axis, span = greatestSpan(mesh) 

    # Get index of the axis
    match axis:
        case "x":
            axisIdx = 0
        case "y":
            axisIdx = 1
        case "z":
            axisIdx = 2
        case _:
            print("Invalid axis; Quitting")
            quit()

    # Sort points
    mesh.vectorList = sortPoints(axis, mesh.vectorList)
    vectorList = mesh.vectorList
    print(f"Axis idx: {axisIdx}")
    # print(vectorList)
    sliceSize = span / SLICES
    max = vectorList[0][axisIdx]
    min = vectorList[-1][axisIdx]
    print(f"Finding centroids- max= {vectorList[0]} min= {vectorList[-1]}")

    floor = max - sliceSize
    ceiling = max
    
    # Iteratively get sums of individual x, y, and z
    # then divide by # vertices to get the avg x, y, and z per slice
    it = iter(vectorList)
    v = next(it)
    dprint(f"Span: {mesh.greatestSpan}  SliceSize: {sliceSize}", debug)
    dprint(f"Floor: {floor}  Ceiling: {ceiling}", debug)
    for i in range(SLICES):
        numVertices = 0
        sums = [0, 0, 0]
        # Move the floor all the way down if on last slice
        # to account for slight inaccuracy from floats
        if (i == SLICES - 1):
            floor = min - 1
        #print(f"First point: {v[axisIdx]} <= {ceiling} = {v[axisIdx] <= ceiling}")
        #print(f"Within constraints?  {v[axisIdx] > floor and v[axisIdx] <= ceiling}")
        while v[axisIdx] > floor and v[axisIdx] <= ceiling:
            #print(f"{floor} < {v[axisIdx]} < {ceiling}")
            sums[0] += v[0]
            sums[1] += v[1]
            sums[2] += v[2]
            numVertices += 1
            try:
                v = next(it)
            except StopIteration: # No next error
                #print("No vectors left")
                break
        # Calculate average by dividing each by numVertices
        centroid = [0, 0, 0]
        foundVertices = False
        for j in range(len(sums)):
            if numVertices == 0:
                pass
            else:
                centroid[j] = sums[j] / numVertices
                foundVertices = True
        if foundVertices:
            centroids.append(tuple(centroid))
        dprint(f"Slice: {i}, numVertices: {numVertices}, centroid: {tuple(centroid)}", debug)
        #print("Iteration", i, "Floor:", floor)
        floor -= sliceSize
    # print("Min= ", min)
    return centroids, vectorList

def sortX(v):
    return v[0]
def sortY(v):
    return v[1]
def sortZ(v):
    return v[2]

# Sorts the points based on the letter axis
# highest to lowest
def sortPoints(axis, vectorCollection):
    if (type(vectorCollection) == dict):
        vectorList = list(vectorCollection.keys())
    else:
        vectorList = list(vectorCollection)
    match axis:
        case "x":
            vectorList.sort(reverse = True, key = sortX)
        case 0:
            vectorList.sort(reverse = True, key = sortX)
        case "y":
            print(type(vectorList))
            vectorList.sort(reverse = True, key = sortY)
        case 1:
            vectorList.sort(reverse = True, key = sortY)
        case "z":
            vectorList.sort(reverse = True, key = sortZ)
        case 2:
            vectorList.sort(reverse = True, key = sortZ)
        case _:
            # 7/1/24 12am bug fix: You idiot why did you specifically put a pass for ints
            print("Invalid axis; Quitting")
            quit()
    #-for v in vectorList:
    #-    print(v)
    return vectorList

# Takes the vectorList and counts the number of points
# at each length value on the axis. Used in plotting
def slicePoints(axisIdx, vectorList):
    vectorList = sortPoints(axisIdx, vectorList)
    it = iter(vectorList)
    v = next(it)
    slices = {}
    i = 0
    while v is not None:
        length = v[axisIdx]
        slices[length] = slices.get(length, 0) + 1
        try:
            v = next(it)
        except StopIteration:
            print("Finished value slicing")
            v = None
    return slices

# This method takes in a vector list and creates a list
# of Slices, finding the centroids of minislices, along with
# their dirVector
def sliceMiniSlices(vectorList, axisIdx, greatestSpanSize) -> list[slc.Slice]:
    # Expects a sorted list
    print(f"\nComputing {MINISLICES} minislices per slice ({SLICES}) across axis[{axisIdx}]")
    if (MINISLICES < 2):
        print(f"Not enough centroids to find a line! MINISLICES = {MINISLICES} < 2")
        quit()
    # [Slice, Slice, ...]
    slices = []
    print(f"  Top Point: {vectorList[0]}\n  Bottom point: {vectorList[-1]}")
    sliceSize = greatestSpanSize / SLICES
    # List must be sorted
    it = iter(vectorList)
    v = next(it)
    miniSliceSize = sliceSize / MINISLICES
    boundary = v[axisIdx] - sliceSize
    
    miniBoundary = v[axisIdx] - miniSliceSize
    
    # For each slice
    for i in range(SLICES):
        print(f"Computing slice {i}")
        miniCentroids = []
        slice = slc.Slice()
        slice.OGVECTORLIST = []
        slice.axisIdx = axisIdx
        # Compute each minislice until its boundary
        for j in range(MINISLICES):
            print(f"  Computing minislice [{j}] of slice [{i}]")
            numVerts = 0
            centroid = [0, 0, 0]
            # For each point in a minislice
            print(f"  Vector: {v[axisIdx]}  Miniboundary: {miniBoundary}  Boundary: {boundary}")
            while v[axisIdx] > miniBoundary:
                slice.OGVECTORLIST.append(v)
                centroid[0] += v[0]
                centroid[1] += v[1]
                centroid[2] += v[2]
                numVerts += 1
                try:
                    v = next(it)
                except:
                    #print("  Out of vertices")
                    break
            # Get average within the minislice
            centroid[0] = centroid[0] / numVerts
            centroid[1] = centroid[1] / numVerts
            centroid[2] = centroid[2] / numVerts
            print(f"Centroid found for slice[{i}][{j}]")
            # Add this minicentroid into the current slice
            miniCentroids.append(centroid)
            # Shift minislice boundary down
            if j != MINISLICES - 2:
                miniBoundary -= miniSliceSize
            # If at the end of a slice
            else:
                # Set to the boundary
                miniBoundary = boundary
        # If at end of mesh, set boundary below
        if i != SLICES - 2:
            boundary -= sliceSize
        else:
            boundary -= (sliceSize + 1)
        # Add minicentroids to list
        slice.centroids = miniCentroids
        #centroids.append(miniCentroids)
        slices.append(slice)

    # Now compute the dirV and datamean for each
    for s in slices:
        s.vectorList = s.OGVECTORLIST.copy()
        s.findCenterline()
        s.VECTORCOUNT = len(s.vectorList)
    return slices
    
# Sets a point on the centerline at the
# same level as the lowest point on the mesh
# to the origin
def setToOrigin(sortedVL, dirVector, axisIdx):
    
    pass

# Rotates the mesh by given number of degrees
# about the given axis
def rotate(vectorList, theta, axisIdx, radians= False):
    if axisIdx == 1:
        theta = -theta
    if radians == False:
        # Convert to radians since sin,cos expect that
        thetaRadians = theta * np.pi / 180
    else:
        thetaRadians = theta
    Rx = np.array([[1, 0, 0],
                  [0, np.cos(thetaRadians), -np.sin(thetaRadians)],
                  [0, np.sin(thetaRadians), np.cos(thetaRadians)]]
                  )
    Ry = np.array([[np.cos(thetaRadians), 0, np.sin(thetaRadians)],
                  [0, 1, 0],
                  [-np.sin(thetaRadians), 0, np.cos(thetaRadians)]]
                  )
    Rz = np.array([[np.cos(thetaRadians), -np.sin(thetaRadians), 0],
                  [np.sin(thetaRadians), np.cos(thetaRadians), 0],
                  [0, 0, 1]]
                  )
    Rotation = (Rx, Ry, Rz)
    for i in range(len(vectorList)):
        vectorList[i] = Rotation[axisIdx] @ vectorList[i]

#Find a fit for the centroids
def findCenterline(mesh):
    centroids = np.array(mesh.centroids)
    # Finding line of best fit
    datamean = centroids.mean(axis= 0)
    #print(datamean)

    # Use SVD to find the direction of the 
    # vector of best fit
    uu, dd, vv = np.linalg.svd(centroids - datamean)
    mesh.dirVector = vv[0]
    mesh.datamean = datamean
    # Use this direction vector elsewhere + datamean

#Fit the line of the centroids to be parallel with the axis
def fitGeometry(mesh, debug = False):
    # Takes the original vectorList and rotates by
    # the accumulated rotation to limit float error
    axisIdx = mesh.axisIdx
    centroids = mesh.centroids
    dirVector = mesh.dirVector
    datamean = mesh.datamean

    vectorList = mesh.OGVECTORLIST.copy()
    #vectorList = mesh.vectorList
    # Centroids sorted by height due to slicing
    pt = centroids[0] 
    # We need the longest axis paired with the other 2

    # True means it's been used; ie not to be used
    axes = [False, False, False]
    axes[axisIdx] = True
    origin = findPointAlongLine(dirVector, axisIdx, datamean)
    dprint(f"Point: {pt}", debug)
    dprint("Line origin: {origin}", debug)
    
    # Rotate about the other 2 axes
    for count, axis in enumerate(axes):
        if axis == False:
            # Get angle of rotation
            dprint(f"Comparing axis {axisIdx} and {count}", True)
            dprint(f"Difference ({pt[axisIdx] - origin[axisIdx]}, {pt[count] - origin[count]})", debug)
            theta = np.arctan((pt[axisIdx] - origin[axisIdx]) / (pt[count] - origin[count]))
            rotationTheta = None
            # Rotate to vertical depending the closer rotation
            # range of arctan is pi/2 to -pi/2
            if (theta >= 0):
                rotationTheta = (np.pi / 2) - theta
            else:
                rotationTheta = -(np.pi / 2) - theta
            # Mark axis
            axes[count] = True
            rotationAxis = None
            # Get other axis to rotate about
            for idx, status in enumerate(axes):
                if (status == False):
                    rotationAxis = idx
                    dprint(f"Rotated about axis {idx}", True)
            dprint(f"Rotating by {np.pi/2} - {theta} = {rotationTheta} radians about ax[{rotationAxis}] to fit", True)
            dprint(f"{theta} radians from horizontal", True)
            # Add to the total rotation the mesh has experienced
            mesh.totalRotation[rotationAxis] += rotationTheta       
            mesh.curRotation[rotationAxis] = rotationTheta
            rotate(vectorList, mesh.totalRotation[rotationAxis], rotationAxis, True)
            # Unmark so the next rotation can use this axis
            axes[count] = False
    mesh.vectorList = vectorList


# Finds a point along some line given the values of
# one coordinate and an anchor point
def findPointAlongLine(dirVector, axisIdx, datamean, axisVal = 0, debug = False):
    # (x, y, z) = dirVector * t + (datamean)

    # t is the scalar at which that axis hits 0
    t = (axisVal -datamean[axisIdx]) / dirVector[axisIdx]
    dprint(f"t at coord[{axisIdx}] = 0 is {t}", debug)
    foundPoint = (dirVector * t) + datamean
    dprint(f"Point found: {foundPoint}", debug)

    return foundPoint

def findTAxis(dirVector, axisIdx, givenPoint, axisVal) -> float:
    # (x, y, z) = dirVector * t + (datamean)
    # t is the scalar at which that axis hits the given axisIdx's coord
    t = (axisVal -givenPoint[axisIdx]) / dirVector[axisIdx]
    return t

def findPerpendicularOnLine(a, b, c):
    # Trying to find point D on a line AC that
    # makes a perpendicular line to B
    # t from point of reference of A
    # https://math.stackexchange.com/questions/4347497/find-a-point-on-a-line-that-creates-a-perpendicular-in-3d-space
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    t = np.dot((b - a), (c - a)) / np.square(np.linalg.norm(c - a))
    d = a + t(c - a)
    return t, d

#FIXME
def findRSquared(dirVector, axisIdx, vectorList, datamean) -> float:
    sst = np.array([0, 0, 0]) # Sum of squares total
    sse = np.array([0, 0, 0]) # Sum of squares error/residual
    pass
    for v in vectorList:
        # Find t according to axisIdx
        sst += (v - datamean)**2
        #sse += (v - )

# TODO: WIP
def calculateDiameterBetweenCentroids(axisIdx, vectorList, centroids : list, avgDiameter: dict = None) -> tuple[dict, list]:
    ''' 
    Calculates the diameter using lines generated between points
    '''
    dirVector = []
    # Add a slice at the ends for final (1/4 size?)

    # Calculate lines by subtracting centroids
    avgDiameter = {}
    for i in range(len(centroids) - 1):
        dirVector[i] = centroids[i] - centroids[i + 1]

    for line in dirVector:
        # Vector list needs to be split
        # datamean anchor
        calculateDiameterAlongLine(axisIdx, vectorList, line, __, avgDiameter)


# Finds the distance between a point and a line defined
# by two points
def findDistFromLine(x1, x2, x0):
    # Finds dist btwn two points on a line (x1, x2)
    # and a point x0
    # https://mathworld.wolfram.com/Point-LineDistance3-Dimensional.html
    x1 = np.array(x1)
    x2 = np.array(x2)
    x0 = np.array(x0)
    crossprod = np.cross(x0 - x1, x0 - x2)
    length = np.linalg.norm(crossprod) / np.linalg.norm(x2 - x1)
    return length

# distance using projection of x0 onto x1x2
# BUG
def findDistFromLineProjection(x1, x2, x0):
    x1 = np.array(x1)
    x2 = np.array(x2)
    x0 = np.array(x0)
    d = (x1 - x2) / np.linalg.norm(x1 - x2)
    print(f"norm = {np.linalg.norm(x1 - x2)}")
    print(f"d = {d}")
    v = x0 - x2
    print(f"v = {v}")
    t = np.dot(v, d)
    print(f"t = {t}")
    # the point on the line closest to x0
    p = x2 + np.dot(t, d)
    print(f"p = {p}")
    length = np.linalg.norm(p - x1)
    return length

def calculateDiameterAlongLineHelper():
    pass
def someFunc():
    # input: avgDiameter dict along t vals
    # process: use line to add to 
    # output: avgDiameter dict along t vals
    pass

def calculateDiameterAlongLine(axisIdx, vectorList, dirVector, datamean, avgDiameter: dict = None) -> tuple[dict, list]:
    ''' Calculates the distance of a point from a line '''
    
    dirVector = np.array(dirVector)
    # Initialize a dictionary
    if (not isinstance(avgDiameter, dict)):
        avgDiameter = {}

    # dict(t, [totallength, count])
    
    centers = []
    # 10 and 20 are arbitrary
    x1 = dirVector * 10 + datamean
    x2 = dirVector * 20 + datamean
    i = 0
    # for each point in the mesh
    for x0 in vectorList:
        # To find the 
        # |(x0 - x1) x (x0 - x2)| / 
        # |x2 - x1|
        # where x0 is the point youre trying to find
        length = findDistFromLine(x1, x2, x0)
        # EXAMINE: is this t calculation right? 7/17/24
        # Yeah, it finds t based on an existing dirV and point 7/17/24
        t = (x0[axisIdx] - datamean[axisIdx]) / dirVector[axisIdx]
        entry = avgDiameter.get(t, [0, 0])
        entry[0] += length
        entry[1] += 1
        avgDiameter[t] = entry

        # Center tracking
        i += 1
        center = dirVector * t + datamean
        if (len(vectorList) < SKIPLEN):
            #print(v, "point vs center", center)
            #print(f"dirV= {dirVector} datamean= {datamean} scalar= {scalar}")
            centers.append(center)
        elif i % (NTHTERM * 5) == 0:
            centers.append(center)
            #print("Shortened", v, "vs", center)
    return avgDiameter, centers

# Calculates average diameter along some centerline from each "x" value
def calculateDiameterByValue(axisIdx, sortedVL, dirVector, datamean, avgDiameter: dict = None) -> tuple[dict, list]:
    dirVector = np.array(dirVector)
    # Use the given dict if ones given
    if (not isinstance(avgDiameter, dict)):
        avgDiameter = {}
    # Axis length to avg diameter at that length
    centers = []
    slices = slicePoints(axisIdx, sortedVL)
    it = iter(sortedVL)
    v = next(it)
    i = 0
    for k in slices.items():
        totalDiameter = 0
        # print(f"k = {k}  v = {v}")
        while k[0] == v[axisIdx]:
            # scalar = v[axis] - datamean / dirVector
            # point = scalar * dirV + datamean
            ## Wrong! -> scalar = v[axisIdx] / (dirVector[axisIdx] + datamean[axisIdx]) 
            if (dirVector[axisIdx] == 0):
                print("EXCEPTION CAUGHT")
                print(f"(scalar = {v[axisIdx] - datamean}) / {dirVector[axisIdx]}")
                print(f"= {(v[axisIdx] - datamean) / dirVector[axisIdx]}")
            try:
                scalar = (v[axisIdx] - datamean) / dirVector[axisIdx]  
            except:
                print("ERROR CALCULATING DIAMETER")
                print(f"(scalar = {v[axisIdx]} - {datamean}) / {dirVector[axisIdx]}")
                quit()  
            # This is how much we need to multiply
            # the direction vector by to get to the same plane as v[axisIdx]
            center = dirVector * scalar + datamean
            i += 1
            if (len(sortedVL) < SKIPLEN):
               #print(v, "point vs center", center)
               #print(f"dirV= {dirVector} datamean= {datamean} scalar= {scalar}")
               centers.append(center)
            elif i % (NTHTERM * 5) == 0:
               centers.append(center)
               #print("Shortened", v, "vs", center)
            # Euclidean distance using numpy
            dist = np.linalg.norm(v - center)
            totalDiameter += dist * 2
            try:
                v = next(it)
            except StopIteration:
                print("Finished calculating diameter")
                break
        avgDiameter[k[0]] = totalDiameter / k[1]    # divide the diameter total by num of points
    # Returns dict of lengths along longest axis to avg diameters
    # print("Centers")
    # print(centers)
    return avgDiameter, centers
        
# Calculates diameter along each value, then averages it into
# the specified chunk width. If dirvector has been calculated
# per slice, goes along with slice dirvector
#
# THIS FUNCTION SERVES MULTIPLE PURPOSES
# 1. Allows for getting the diameter as an average across values
# 2. Supports the analyzation of diameter using straightened slices
#    which it will do if the channel was sliced
# 3. Supports the getting of diameter on a value-by-value basis
# 4. Supports getting diameter along an unsliced channel
#
# In short, use this function for most diameter getting since it's
# more powerful. calculateDiameter() is only able to be used in the
# case of getting value-by-value avg diameters across an unsliced
# computed mesh.
def condenseDiameterByChunk(axisIdx, specifiedWidth, channel) -> tuple[dict, list]:
    # Calculate diameter across some specified width
    # Axis length to avg diameter at that length
    avgDiameter = {}    # Length : diameter in that length
    centers = []

    if channel.slices != None:
        # Start at the top
        # Get all diameter values per point into avgDiameter
        print("Channel slices were computed, using individual centerlines")
        for slice in channel.slices:
            d, c = slice.calculateDiameter(avgDiameter)
            centers.extend(c)

        # Then merge them across the specified width
        topPoint = channel.slices[0].vectorList[0]
        bottomPoint = channel.slices[-1].vectorList[-1]
    else:
        print("Channel slices not computed, computing across centerline")
        avgDiameter, centers = calculateDiameterByValue(axisIdx, channel.vectorList, channel.dirVector, channel.datamean, avgDiameter)
        topPoint = channel.vectorList[0]
        bottomPoint = channel.vectorList[-1]    # 6/30/24 Oops, left it as 0 before
    floor = topPoint[axisIdx] - specifiedWidth

    # If the specified width is 0, just return what we have. No need to condense
    if specifiedWidth == 0:
        return avgDiameter, centers

    numChunks = mth.ceil((topPoint[axisIdx] - bottomPoint[axisIdx]) / specifiedWidth)
    chunkAvgDiameter = {}
    lengthSum = 0
    diameterSum = 0
    numSlicesInChunk = 0
    # Go through each entry and combine them within the specifiedWidth
    # (assumes dictionary is sorted greatest to least)
    for height, diameter in avgDiameter.items():
        if height > floor:
            lengthSum += height
            diameterSum += diameter
            numSlicesInChunk += 1
        else:
            chunkAvgDiameter[lengthSum / numSlicesInChunk] = diameterSum / numSlicesInChunk
            lengthSum = 0
            diameterSum = 0
            numSlicesInChunk = 0
            numChunks -= 1
            if numChunks == 1:
                floor = bottomPoint[axisIdx] - 1
            else:
                floor -= specifiedWidth
    return chunkAvgDiameter, centers

print("Vector Analyzer imported")

