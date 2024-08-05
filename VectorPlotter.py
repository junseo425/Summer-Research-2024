import numpy as np
import matplotlib.pyplot as plt
import Constants as const

NTHTERM = const.NTHTERM
SKIPLEN = 300

def dprint(string, debug = False):
    if (debug):
        print(string)

def plotCentroids(centroids, ax):
    #centroids should be numpy array
    ptsX = []
    ptsY = []
    ptsZ = []
    if centroids is None:
        print("do mesh.findCentroids first!")
        quit()
    for v in centroids:
        ptsX.append(v[0])
        ptsY.append(v[1])
        ptsZ.append(v[2])
    ptsX = np.array(ptsX)
    ptsY = np.array(ptsY)
    ptsZ = np.array(ptsZ)
    ax.scatter(ptsX, ptsY, ptsZ)

def plotMesh(vectorList, ax, resolution):
    #print(f"Resolution {resolution}")
    # VectorList should be numpy array
    ptsX = []
    ptsY = []
    ptsZ = []
    if (resolution == 1):
        for v in vectorList:
            ptsX.append(v[0])
            ptsY.append(v[1])
            ptsZ.append(v[2])
    else:
        i = 0
        n = 1 // resolution
        if n > len(vectorList):
            print(f"Attempted to plot every {n}th point but mesh contains {len(vectorList)} points")
            n = 1
            print(f"Resolution too low; plotting every {n}th point")
        print(f"Plotting every {n}th point")
        for v in vectorList:
            if (i % n == 0):
                ptsX.append(v[0])
                ptsY.append(v[1])
                ptsZ.append(v[2])
            i += 1

    ptsX = np.array(ptsX)
    ptsY = np.array(ptsY)
    ptsZ = np.array(ptsZ)
    ax.set_box_aspect((np.ptp(ptsX), np.ptp(ptsY), np.ptp(ptsZ)))  # aspect ratio is 1:1:1 in data space
    ax.scatter(ptsX, ptsY, ptsZ)


def plotCenterLine(centroids, ax, min, max, debug = False):
    centroids = np.array(centroids)
    dprint(f"Plot centerline centroids: {centroids}", debug)
    if centroids is None or centroids.size == 0:
        print("Error: Call findCentroids() first! The direction vector, datamean, and centroids haven't been found yet!")
        quit()
    # Finding line of best fit
    datamean = centroids.mean(axis= 0)
    #print(datamean)

    # Use SVD to find the direction of the 
    # vector of best fit
    uu, dd, vv = np.linalg.svd(centroids - datamean)

    # Now generate points for plotting
    #point1 = va.findPointAlongLine(vv[0], axisIdx, datamean, min)
    #point2 = va.findPointAlongLine(vv[0], axisIdx, datamean, max)
    #print(f"POINTS: {point1} {point2}")
    #linepts = vv[0] * np.array([point1, point2]) + datamean
    linepts = vv[0] * np.mgrid[min:max:2j][:, np.newaxis] + min
    dprint(f"VV[0]= {vv[0]}", debug)
    dprint(f"mgrid= {np.mgrid[min:max:2j][:, np.newaxis]}", debug)
    dprint(f"linepts= {linepts}", debug)
    #print("input to plot3d:", *linepts.T)
    ax.plot3D(*linepts.T)

# Takes in a dictionary of x value to avg diameter
def plotDiameter(ax, avgDiameter):
    axisVals = list(avgDiameter.keys())
    diameters = list(avgDiameter.values())
    # print(f"axisVals = {axisVals}\ndiameter = {diameters}")
    plt.plot(axisVals, diameters)
    
    # i = 0
    # for v in avgDiameter.items():
    #     if (len(avgDiameter) > SKIPLEN):
    #         if (i % NTHTERM == 0):
    #             print(f"Skipped {NTHTERM} [{v[0]}, {v[1]}]")
    #     else:
    #         print(f"[{v[0]}, {v[1]}]")
    #     i += 1
        
def plotAxes(ax, length):
    ax.plot3D([0, length],  [0, 0], [0, 0], c = 'r')
    ax.plot3D([0, 0], [0, length], [0, 0], c = 'gold')
    ax.plot3D([0, 0], [0, 0], [0, length], c = 'b')

# Takes in an axes object and clears it
def clearPlot(ax):
    ax.cla()
