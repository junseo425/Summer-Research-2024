# This file contains the representation of a mesh from a .STL file
# and contains functions related to analyzing and plotting it
import VectorReader as vr
import VectorAnalyzer as va
import VectorPlotter as vp
import Constants as const
import numpy as np
import matplotlib.pyplot as plt

THRESHOLD = const.THRESHOLD
ITERATIONNUM = const.ITERATIONNUM

class Straightenable:
    def __init__(self):
        # Unchanging
        self.VECTORCOUNT  = None
        self.OGVECTORLIST = None
        self.ITERATIONNUM = ITERATIONNUM

        # Updates as the object rotates
        self.axisIdx        = None
        self.greatestSpan   = None  # tracks the axis of greatest span and the units it spans
        self.vectorList     = None  # required for sorting and ordering; current position
        self.centroids      = None  # represents the centroids of the slices of the channel
        self.datamean       = None  # represents the center of the channel
        self.dirVector      = None  # represents the slope of the centerline
        self.totalRotation     = [0, 0, 0]  # tracks the rotation applied to the original mesh (in radians)
        self.curRotation       = [0, 0, 0]  # the amount the object was rotated by the current iteration
        self.change         = None  # the amount of change the mesh went through in the last iteration

        # Data
        self.diameter : dict = None
        self.condensedDiameter : dict = None    # stores the condensed version of diameter
        self.centers : list  = None

        # Flags
        self.isSorted         = False # tracks if the vectorList has been sorted
        self.centroidsUpdated = False # tracks if the centroids have been computed
        self.greatestSpanUpdated    = False # tracks if the greatest span has been kept updated
        self.diameterUpdated     = False # tracks if avgDiameter is up to date

        # Methods
        self.diameterMethod = va.calculateDiameterByValue

    def getGreatestSpan(self):
        if self.greatestSpanUpdated == False:
            # Sets axisIdx as well
            self.greatestSpan = va.greatestSpan(self)
            self.greatestSpanUpdated = True
    
    def sortPoints(self):
        # Don't sort if already sorted
        if (self.isSorted):
            return
        self.vectorList = va.sortPoints(self.axisIdx, self.vectorList)
        self.isSorted = True
        print("Sorted")

    def findCentroids(self):
        if self.centroidsUpdated == True:
            return
        self.centroids, self.vectorList = va.findCentroids(self)
        va.findCenterline(self)
        self.isSorted = True  # Finding the centroids sorts the points
        self.centroidsUpdated = True

    def fitGeometry(self):
        self.findCentroids()
        va.fitGeometry(self)
        # Rotation mixes points
        # centroids must be recomputed 
        self.resetState()

    # Straightens the channel iteratively until threshold in Straightenable.py
    # Then calculates the diameters across the axis
    def straighten(self, resolution = .01, show = True):
        if show:
            print(f"Resolution {resolution}")
            ax1 = plt.axes(projection = '3d')
            plt.title("Original")
            vp.plotMesh(self.vectorList, ax1, resolution)
            plt.show()
        iteration = 1
        print(f"\nIteration: {iteration}    threshold: {THRESHOLD}")
        self.fitGeometry()
        if show == True and iteration % self.ITERATIONNUM == 0:
                plt.title(f"Iteration {iteration}")
                self.show(resolution)
        rotationSum = (self.curRotation[0] ** 2) + (self.curRotation[1] ** 2) + (self.curRotation[2] ** 2)
        print(f"Rotation: {self.curRotation} =  {rotationSum}")
        while not self.atThreshold():
            iteration += 1
            print(f"\nIteration: {iteration}")
            self.fitGeometry()
            rotationSum = (self.curRotation[0] ** 2) + (self.curRotation[1] ** 2) + (self.curRotation[2] ** 2)
            print(f"Rotation: {self.curRotation} =  {rotationSum}")
            print(f"Total rotation = {self.totalRotation}")
            if show == True and iteration % self.ITERATIONNUM == 0:
                plt.title(f"Iteration {iteration}")
                self.show(resolution)
        print(f"\nThreshold reached ({THRESHOLD})")
        print(f"  Last rotation (radians): {self.curRotation} = {rotationSum}")
        print(f"  Total rotation (radians): {self.totalRotation}")
        if show:
            plt.title("Straightened")
            self.show(resolution)

    
    # Make a diameter function for different slices

    def atThreshold(self):
        pt = self.centroids[0] # Take a point from somewhere
        radius = np.linalg.norm(pt - self.datamean)  # Get the dist
        c = 0
        for axisIdx, radians in enumerate(pt):
            c += (radius * self.curRotation[axisIdx]) ** 2
        c = np.sqrt(c)
        self.change = c
        print(f"Threshold: {c} < {THRESHOLD} = {c < THRESHOLD}")

        if c < THRESHOLD:
            return True
        else:
            return False
        
    def rotate(self, rotation, radians = True):
        if not radians:
            for axis, theta in rotation:
                rotation[axis] = theta * np.pi / 180
        for axis, rads in enumerate(rotation):
            va.rotate(self.vectorList, rads, axis, True)
        self.resetState()   # set flags to false

    def computeDiameter(self, width = 0):
        if self.diameterUpdated:
            return
        self.diameter, self.centers = self.diameterMethod(self.axisIdx, self.vectorList, self.dirVector, self.datamean)
        self.diameterUpdated = True
        #calculateDiameterByValue(axisIdx, sortedVL, dirVector, datamean, avgDiameter: dict = None) -> tuple[dict, list]
        #calculateDiameterAlongLine(axisIdx, vectorList, dirVector, datamean, avgDiameter: dict = None) -> tuple[dict, list]:
        #calculateDiameterBetweenCentroids(axisIdx, vectorList, points, avgDiameter: dict = None) -> tuple[dict, list]:

    ### Setters and Getters ###
    def getDiameter(self):
        if self.diameterUpdated:
            return self.diameter
        self.computeDiameter()
        return self.diameter
    def getCenters(self):
        if self.diameterUpdated:
            return self.centers
        self.computeDiameter()
        self.centers

    def getSortedVL(self):
        if self.isSorted:
            return self.vectorList
        self.sortPoints()
        return self.vectorList
    # For data purposes, gets the single avg diameter across
    # entire channel
    def getEntireDiameter(self, width):
        self.findCentroids()
        avgDiameter, centers = va.condenseDiameterByChunk(self.axisIdx, width, self)
        avgDiameterNum = 0
        for d in avgDiameter.values():
            avgDiameterNum += d
        avgDiameterNum /= len(avgDiameter)
        print(f"Average diameter across object: {avgDiameterNum}")
        return avgDiameterNum

    # Sets all flags to false so everything will be recomputed
    def resetState(self) -> None:
        self.isSorted = False
        self.centroidsUpdated = False
        self.greatestSpanUpdated = False
        self.diameterUpdated = False

    def setDiameterMethod(self, method):
        self.diameterMethod = method

    ## Files
    # UNFINISHED
    def save(self, filename):
        file = open(filename, "w")
        # Info
        file.write(f"VECTORCOUNT,{self.VECTORCOUNT}")
        file.write(f"axisIdx,{self.axisIdx}")
        file.write(f"greatestSpan,{self.greatestSpan[0]},{self.greatestSpan[1]}")
        file.write(f"change,{self.change}")
        # Flags
        file.write(f"sorted,{self.isSorted}")
        file.write(f"centroidsFound,{self.centroidsUpdated}")
        file.write(f"greatestSpanUpdated,{self.greatestSpanUpdated}")
        file.write(f"\n")

        # Any point lists
        file.write(f"OGVECTORLIST")
        for v in self.OGVECTORLIST:
            file.write(f"{v[0]},{v[1]},{v[2]}")
        file.write(f"\n")
        file.write("vectorList")
        for v in self.vectorList:
            file.write(f"{v[0]},{v[1]},{v[2]}")

    ### Plotting
    def dprint(self, string, debug):
        if debug:
            print(string)

    def plotCentroids(self, ax):
        self.findCentroids()
        vp.plotCentroids(self.centroids, ax)

    def plotMesh(self, ax, resolution = 1):
        vp.plotMesh(self.vectorList, ax, resolution)

    def plotCenterLine(self, ax, debug = False):
        min = self.vectorList[-1][self.axisIdx]
        max = self.vectorList[0][self.axisIdx]
        self.dprint(f"Min: {self.vectorList[-1]}, Max: {self.vectorList[0]}", debug)
        vp.plotCenterLine(self.centroids, ax, min, max)

    # Shows centroids, centerline, mesh, and axes
    def show(self, resolution = 1, debug = False):
        ax1 = plt.axes(projection = '3d')
        self.plotCentroids(ax1)
        self.dprint(f"\nPlotting", debug)
        self.dprint(f"datamean: {self.datamean}", debug)
        self.plotCenterLine(ax1)
        self.plotMesh(ax1, resolution)
        self.plotTargetAxis(ax1, self.axisIdx)
        plt.show()
    
    def plotTargetAxis(self, ax, axisIdx):
        axes = [[0, 0], [0, 0], [0, 0]]
        axes[axisIdx][1] = self.datamean[axisIdx]
        vp.plotAxes(ax, self.datamean[axisIdx] / 4)
        ax.plot3D(axes[0], axes[1], axes[2])

    # Mostly deprecated, use chunkdiameter to show diameter plotted
    # along slice centerlines
    def plotDiameter(self, ax):
        self.findCentroids()
        self.computeDiameter()
        vp.plotDiameter(ax, self.getDiameter())
        
        axis = None
        match self.axisIdx:
            case 0:
                axis = "x"
            case 1:
                axis = "y"
            case 2:
                axis = "z"
            case _:
                print("Invalid axis")
                quit()
        plt.xlabel(f"{axis} axis")
        plt.ylabel("Diameter")
        ax.set_ylim([0, None])
        plt.title("Diameter across the longest axis")

    def showDiameter(self, ax: plt.Axes = plt.axes()):
        self.plotDiameter()
        plt.show()
    
    # Shows plot for diameter and centers
    def plotDiameterInfo(self):
        self.plotDiameter()
        self.showCenters(self.getCenters())

    def saveDiameterPlot(self, filepath : str):
        ax = plt.axes()
        self.plotDiameter(ax)
        plt.savefig(filepath)

    def showCenters(self, centers):
        # Showing centers in relation
        x = []
        y = []
        z = []
        for v in centers:
            x.append(v[0])
            y.append(v[1])
            z.append(v[2])
        ax1 = plt.axes(projection= '3d')
        self.plotMesh(ax1, .01)
        ax1.scatter(x, y, z)
        print(f"Centers: {len(centers)}")
        plt.title("Centers")
        plt.show()

    # TODO: Make chunkdiameter called by getDiameter
    def plotChunkDiameter(self, specifiedWidth):
        ax = plt.axes()
        self.findCentroids()
        avgDiameter, centers = va.condenseDiameterByChunk(self.axisIdx, specifiedWidth, self)
        avgDiameterNum = 0
        for d in avgDiameter.values():
            avgDiameterNum += d
        avgDiameterNum /= len(avgDiameter)
        print(f"Average diameter across object: {avgDiameterNum}")
        vp.plotDiameter(ax, avgDiameter)
        
        axis = None
        match self.axisIdx:
            case 0:
                axis = "x"
            case 1:
                axis = "y"
            case 2:
                axis = "z"
            case _:
                print("Invalid axis")
                quit()
        plt.xlabel(f"{axis} axis")
        plt.ylabel("Diameter")
        ax.set_ylim([0, None])
        plt.title("Diameter across the longest axis")
        plt.show()
        self.showCenters(centers)
