import numpy as np
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy


maxShift = 2


# assignment stage
def assignment(df, centroid, weights):
    for i in centroid.keys():
        df['distance_from_{}'.format(i)] = (
                weights[i][0] * ((df['x'] - centroid[i][0]) ** 2)
                + weights[i][1] * ((df['y'] - centroid[i][1]) ** 2)
                + weights[i][2] * ((df['i'] - centroid[i][2]) ** 2)
        )
    centroid_distance_cols = ['distance_from_{}'.format(i) for i in centroid.keys()]
    df['closest'] = df.loc[:, centroid_distance_cols].idxmin(axis=1)
    df['closest'] = df['closest'].map(lambda x: int(x.lstrip('distance_from_')))
    return df


def update(centroid, df):
    for i in centroid.keys():
        centroid[i][0] = int(np.mean(df[df['closest'] == i]['x'])) + 1
        centroid[i][1] = int(np.mean(df[df['closest'] == i]['y'])) + 1
        centroid[i][2] = int(np.mean(df[df['closest'] == i]['i'])) + 1
        centroid[i][3] = len(df[df['closest'] == i]['i'])

    return centroid


''' finding the weights '''
def find_weights(centroid):

    wx = []
    wy = []
    wz = []

    for i in centroid.keys():
        temp1 = 1/(centroid[i][2] - 1)
        temp2 = 0
        for t in centroid[i][4]:
            temp2 = temp2 + ((t[0] - centroid[i][0])**2)

        temp3 = np.sqrt(temp1 * temp2)

        temp4 = 1 / (centroid[i][2] - 1)
        temp5 = 0
        for t in centroid[i][4]:
            temp5 = temp5 + ((t[1] - centroid[i][1]) ** 2)

        temp6 = np.sqrt(temp4 * temp5)

        temp7 = 1 / (centroid[i][2] - 1)
        temp8 = 0
        for t in centroid[i][4]:
            temp8 = temp8 + ((t[2] - centroid[i][2]) ** 2)

        temp9 = np.sqrt(temp7 * temp8)

        cj = (temp3 * temp6 * temp9) ** (1/3)

        wx.append(cj/temp3)
        wy.append(cj/temp6)
        wz.append(cj/temp9)

    weights = {
        i + 1: [wx[i], wy[i], wz[i]]
        for i in range(len(wz))
    }

    return  weights


img1 = cv2.imread('f1.bmp', 0)
img2 = cv2.imread('f2.bmp', 0)

k = int(input("Enter the value of k: "))


def kmeans(img):
    px = []
    py = []
    pi = []

    rows1 = img.shape[0]
    cols1 = img.shape[1]

    for i in range(rows1):
        for j in range(cols1):
            px.append(i)
            py.append(j)
            pi.append(img[i,j])

    dfPixels = pd.DataFrame({
        'x': px,
        'y': py,
        'i': pi
    })

    cx = []
    cy = []

    c1 = int(rows1/k)
    c2 = int(cols1/k)
    print(c1,c2)

    for i in range(0, rows1, k):
        for j in range(0, cols1, k):
            cx.append(int(k/2) + i)
            cy.append(int(k/2) + j)


    mx = []
    my = []
    mz = []
    mn = []
    mp = []

    tempX = []
    tempY = []
    tempZ = []
    tempP = []

    for i in range(0, rows1, k):
        for j in range(0, cols1, k):
            for ti in range(k):
                if ti+i <= rows1 and ti+j <= cols1:
                    tempX.append(i+ti)
                    tempY.append(j+ti)
                    tempZ.append(img[i+ti, j+ti])
            for ti in range(k):
                for tj in range(k):
                    if ti + i <= rows1 and tj + j <= cols1:
                        tempP.append([i+ti, j+tj, img[i+ti, j+tj]])

            mx.append( int(np.mean(tempX)) + 1)
            my.append( int(np.mean(tempY)) + 1)
            mz.append( int(np.mean(tempZ)) + 1)
            mn.append(len(tempX)*len(tempY))
            mp.append( tempP)
            tempZ = []
            tempX = []
            tempY = []
            tempP = []


    centroidMeans = {
        i+1:[mx[i], my[i], mz[i], mn[i], mp[i]]
        for i in range(len(mz))
    }

    weights = find_weights(centroidMeans)

    dfPixels = assignment(dfPixels, centroidMeans, weights)
    #print(dfPixels)
    # print(len(dfPixels[dfPixels['closest'] == 40]['i']))


    countIteration = 0
    reCalCount = 0


    while True:
        countIteration = countIteration + 1
        reCalCount = reCalCount + 1
        print("iteration : ", countIteration)

        closest_centroids = dfPixels['closest'].copy(deep=True)
        centroidMeans = update(centroidMeans, dfPixels)
        dfPixels = assignment(dfPixels, centroidMeans, weights)
        if closest_centroids.equals(dfPixels['closest']):
            break


    for i in centroidMeans.keys():
        tempP = []
        tempP.append([dfPixels[dfPixels['closest'] == i]['x'], dfPixels[dfPixels['closest'] == i]['y'], dfPixels[dfPixels['closest'] == i]['i']])
        centroidMeans[i][4] = tempP
    return [dfPixels, centroidMeans]


r1 = kmeans(img1)
dfPixels1 = r1[0]
centroidMeans1 = r1[1]

r2 = kmeans(img2)
dfPixels2 = r2[0]
centroidMeans2 = r2[1]




''' cluster matching '''
def cluster_5d_matrix_genrate(centroid, image1, image2):
    msj = []
    mmj = []
    msj = []
    temp1 = 0
    for i in centroid.keys():
        temp1 = 1/centroid[i][3]
        sum = 0
        for j in range(centroid[i][3]):
            sum = sum + image1[centroid[i][4][0][0].iloc[j], centroid[i][4][0][1].iloc[j]] - image2[centroid[i][4][0][0].iloc[j], centroid[i][4][0][1].iloc[j]]

        msj.append(int(temp1*sum))


    ''' momentum genreation'''

    meanXList = []
    meanYList = []
    for i in centroid.keys():
        meanX = 0
        meanY = 0
        meanI = 0
        for j in range(centroid[i][3]):
            meanX = meanX + (centroid[i][4][0][0].iloc[j])*image1[centroid[i][4][0][0].iloc[j], centroid[i][4][0][1].iloc[j]]
            meanY = meanY + (centroid[i][4][0][1].iloc[j]) * image1[centroid[i][4][0][0].iloc[j], centroid[i][4][0][1].iloc[j]]
            meanI = meanI + image1[centroid[i][4][0][0].iloc[j], centroid[i][4][0][1].iloc[j]]

        meanXList.append(meanX/meanI)
        meanYList.append(meanY/meanI)

    for i in centroid.keys():
        temp2 = 0
        for j in range(centroid[i][3]):
            temp2 = temp2 + ((centroid[i][4][0][0].iloc[j] - meanXList[i-1])**2 + (centroid[i][4][0][1].iloc[j] - meanYList[i-1]**2) )*image1[centroid[i][4][0][0].iloc[j], centroid[i][4][0][1].iloc[j]]

        mmj.append(int(temp2))


    matrix_5d = {
        i : [centroid[i][0], centroid[i][1], centroid[i][2], mmj[i-1], msj[i-1]]
        for i in centroid.keys()
    }

    return matrix_5d


matrix1 = cluster_5d_matrix_genrate(centroidMeans1, img1, img2)

matrix2 = cluster_5d_matrix_genrate(centroidMeans2, img2, img1)



def find_u_weights(m1,m2):
    dmx = 0
    dmy = 0
    dmz = 0
    dms = 0
    dmm = 0

    for i in m1.keys():
        dmx = dmx + (m1[i][0] - m2[i][0])**2
        dmy = dmy + (m1[i][1] - m2[i][1])**2
        dmz = dmz + (m1[i][2] - m2[i][2])**2
        dmm = dmm + (m1[i][3] - m2[i][3])**2
        dms = dms + (m1[i][4] - m2[i][4])**2

    ux = (dmx**(-4/5))*(dmy**(1/5))*(dmz**(1/5))*(dmm**(1/5))*(dms**(1/5))
    uy = ux * dmx / dmy
    uz = ux * dmx / dmz
    um = ux * dmx / dmm
    us = ux * dmx / dms

    return [ux, uy, uz, um, us]


uWeights = find_u_weights(matrix1, matrix2)


def cluster_matching(df, centroid1, centroid2, weights):
    for i in centroid1.keys():
        df['matching_from_{}'.format(i)] = (
                weights[0] * (( df['x'] - centroid2[i][0]) ** 2)
                + weights[1] * ((df['y'] - centroid2[i][1]) ** 2)
                + weights[2] * ((df['i'] - centroid2[i][2]) ** 2)
                + weights[3] * ((df['m'] - centroid2[i][3]) ** 2)
                + weights[4] * ((df['s'] - centroid2[i][4]) ** 2)
            )
    centroid_distance_cols = ['matching_from_{}'.format(i) for i in centroid1.keys()]
    df['closestMatch'] = df.loc[:, centroid_distance_cols].idxmin(axis=1)
    df['closestMatch'] = df['closestMatch'].map(lambda x: int(x.lstrip('matching_from_')))
    return df




def clustering(df, centroids1, centroids2, weights):
    countMatchingIteration = 0
    while True:
        countMatchingIteration = countMatchingIteration + 1
        print("matching iteration : ", countMatchingIteration)
        closest_Match = df['closestMatch'].copy(deep=True)
        df = cluster_matching(df,centroids1, centroids2, weights)
        if closest_Match.equals(df['closestMatch']):
            break

    return df


cx = []
cy = []
ci = []
cm = []
cs = []

for i in matrix1.keys():
    cx.append(matrix1[i][0])
    cy.append(matrix1[i][1])
    ci.append(matrix1[i][2])
    cm.append(matrix1[i][3])
    cs.append(matrix1[i][4])

dfClusterMatch = pd.DataFrame({
        'x': cx,
        'y': cy,
        'i': ci,
        'm': cm,
        's': cs
    })
dfClusterMatch = cluster_matching(dfClusterMatch,matrix1, matrix2, uWeights)
final_match = clustering(dfClusterMatch, matrix1, matrix2, uWeights)


ax = plt.axes()
plt.xlim(0, img1.shape[0])
plt.ylim(0, img1.shape[1])
for i in centroidMeans1.keys():
    k = final_match.closestMatch[i-1]
    old_x = centroidMeans2[k][0]
    old_y = centroidMeans2[k][1]
    dx = (centroidMeans1[i][0] - centroidMeans2[k][0])
    dy = (centroidMeans1[i][1] - centroidMeans2[k][1])
    cv2.arrowedLine(img2,  (old_x, old_y), (centroidMeans1[i][0], centroidMeans1[i][1]), (0,0,255), 1)
    ax.arrow(old_x, old_y, dx, dy, head_width=2, head_length=3)
plt.show()


for i in centroidMeans1.keys():
    plt.scatter(centroidMeans1[i][0],centroidMeans1[i][1])
plt.show()

for i in centroidMeans2.keys():
    plt.scatter(centroidMeans2[i][0],centroidMeans2[i][1])
plt.show()

cv2.imshow('image', img2)
cv2.waitKey(0)
cv2.destroyAllWindows()
