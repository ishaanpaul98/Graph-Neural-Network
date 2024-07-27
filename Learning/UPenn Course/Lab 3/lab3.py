import numpy as np
import os
import zipfile # To handle zip files
import torch as torch

def load_data(movie, minRatings):

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

def create_graph(X, idxTrain, knn):

    zeroTolerance = 1e-9

    N = X.shape[1]

    XTrain = np.transpose(X[idxTrain, :])

    binaryTemplate = (XTrain > 0).astype(XTrain.dtype)
    sumMatrix = XTrain.dot(binaryTemplate.T)
    countMatrix = binaryTemplate.dot(binaryTemplate.T)
    countMatrix[countMatrix == 0] = 1
    avgMatrix = sumMatrix/countMatrix
    sqSumMatrix = XTrain.dot(XTrain.T)
    correlationMatrix = sqSumMatrix / countMatrix - avgMatrix * avgMatrix.T

    sqrtDiagonal = np.sqrt(np.diag(correlationMatrix))
    nonzeroSqrtDiagonalIndex = (sqrtDiagonal > zeroTolerance).astype(sqrtDiagonal.dtype)
    sqrtDiagonal[sqrtDiagonal < zeroTolerance] = 1
    invSqrtDiagonal = 1/sqrtDiagonal
    invSqrtDiagonal = invSqrtDiagonal * nonzeroSqrtDiagonalIndex
    normalizationMatrix = np.diag(invSqrtDiagonal)

    normalizedMatrix = normalizationMatrix.dot(correlationMatrix.dot(normalizationMatrix)) - np.eye(correlationMatrix.shape[0])

    normalizationMatrix[np.abs(normalizedMatrix) < zeroTolerance] = 0
    W = normalizedMatrix

    WSorted = np.sort(W, axis=1)
    threshold = WSorted[:, -knn].squeeze()
    thresholdMatrix = (np.tile(threshold, (N, 1))).transpose()
    W[W<thresholdMatrix] = 0

    E, V = np.linalg.eig(W)
    W = W/np.max(np.abs(E))

    return W



def split_data(X, idxTrain, idxTest, idxMovie):  
    
    N = X.shape[1]
    
    xTrain = X[idxTrain,:]
    idx = np.argwhere(xTrain[:,idxMovie]>0).squeeze()
    xTrain = xTrain[idx,:]
    yTrain = np.zeros(xTrain.shape)
    yTrain[:,idxMovie] = xTrain[:,idxMovie]
    xTrain[:,idxMovie] = 0
    
    xTrain = torch.tensor(xTrain)
    xTrain = xTrain.reshape([-1,1,N])
    yTrain = torch.tensor(yTrain)
    yTrain = yTrain.reshape([-1,1,N])
    
    xTest = X[idxTest,:]
    idx = np.argwhere(xTest[:,idxMovie]>0).squeeze()
    xTest = xTest[idx,:]
    yTest = np.zeros(xTest.shape)
    yTest[:,idxMovie] = xTest[:,idxMovie]
    xTest[:,idxMovie] = 0
    
    xTest = torch.tensor(xTest)
    xTest = xTest.reshape([-1,1,N])
    yTest = torch.tensor(yTest)
    yTest = yTest.reshape([-1,1,N])
    
    return xTrain, yTrain, xTest, yTest

def movieMSELoss(yHat,y,idxMovie):
    mse = torch.nn.MSELoss()
    return mse(yHat[:,:,idxMovie].reshape([-1,1]),y[:,:,idxMovie].reshape([-1,1]))

def main():

    X, idxContact = load_data(movie=257, min_ratings=150)

    # Creating and sparsifying the graph

    nTotal = X.shape[0] # total number of users (samples)
    permutation = np.random.permutation(nTotal)
    nTrain = int(np.ceil(0.9*nTotal)) # number of training samples
    idxTrain = permutation[0:nTrain] # indices of training samples
    nTest = nTotal-nTrain # number of test samples
    idxTest=permutation[nTrain:nTotal] # indices of test samples

    W = create_graph(X=X, idxTrain=idxTrain, knn=40)
    xTrain, yTrain, xTest, yTest = split_data(X, idxTrain, idxTest, idxContact)
    nTrain = xTrain.shape[0]
    nTest = xTest.shape[0]