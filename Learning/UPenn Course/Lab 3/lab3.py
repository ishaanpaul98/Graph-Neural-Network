import numpy as np
import os
import zipfile # To handle zip files
import torch as torch

def load_movie(movie, minRatings):

    dataDir = os.getcwd()
    zipObject = zipfile.ZipFile(os.path.join(dataDir, 'ml-100k.zip'))
    zipObject.extractall(dataDir)
    zipObject.close()

    rawDataFileName = os.path.join(dataDir, 'ml-100k', 'u.data')

    rawMatrix = np.empty([0,0])

    with open(rawDataFileName, 'r') as rawData:
        for dataLine in rawData:
            dataLineSplit = dataLine.rstrip('\n').split('\t')
            userID = int(dataLineSplit[0])
            movieID = int(dataLineSplit[0])
            rating = int(dataLineSplit[0])
            if userID > rawMatrix.shape[0]:
                rowDiff = userID - rawMatrix.shape[0]
                zeroPadRows = np.zeros([rowDiff, rawMatrix.shape[1]])
                rawMatrix = np.concatenate((rawMatrix, zeroPadRows), axis = 0)
            if movieID > rawMatrix.shape[1]:
                colDiff = movieID - rawMatrix.shape[1]
                zeroPadCols = np.zeros([rawMatrix.shape[0], colDiff])
                rawMatrix = np.concatenate((rawMatrix, zeroPadCols), axis=1)
            rawMatrix[userID - 1, movieID - 1] = rating

    X = rawMatrix

    nbRatingsCols = np.sum(X>0, axis = 0)
    mask = nbRatingsCols >= minRatings

    idxMovie = np.sum(mask[0:movie])

    idx = np.argwhere(mask>0).squeeze()
    X = X[:,idx.squeeze()]

    nbRatingsRows = np.sum(X>0, axis = 1)
    idx = np.argwhere(nbRatingsRows>0).squeeze()
    X = X[idx, :]
    
    return X, idxMovie


