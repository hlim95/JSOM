import numpy as np
import pandas as pd
import random

def corr2_coeff(A,B):
    A_mA = A - A.mean(1)[:,None]
    B_mB = B - B.mean(1)[:,None]

    ssA = (A_mA**2).sum(1);
    ssB = (B_mB**2).sum(1);

    return np.dot(A_mA,B_mB.T)/np.sqrt(np.dot(ssA[:,None],ssB[None]))

def correlation(Dataset1, Dataset2):
    CORR = corr2_coeff(Dataset1, Dataset2)

    matching1to2 = []
    matching1to2_dist = []
    for i in range(len(Dataset1)):
        max_idxs = np.argsort(CORR[i,:])
        max_idx = max_idxs[len(max_idxs)-1]
        matching1to2.append(max_idx)
        matching1to2_dist.append(np.sort(CORR[i,:])[-1])
        
    matching2to1 = []
    matching2to1_dist = []
    for i in range(len(Dataset2)):
        max_idxs = np.argsort(CORR[:,i])
        max_idx = max_idxs[len(max_idxs)-1]
        matching2to1.append(max_idx)
        matching2to1_dist.append(np.sort(CORR[:,i])[-1])
        

    Matching1to2 = [matching1to2, matching1to2_dist]
    Matching2to1 = [matching2to1, matching2to1_dist]
    return Matching1to2, Matching2to1

def stepMiner(l):  # l is a list
    l=sorted(l)
    m = np.mean(l)
    sstot = sum([m - i for i in l])
    L = []
    R = l.copy() 
    nL = len(L)
    nR = len(R)
    mL = np.mean(L)
    mR = np.mean(R)
    min_sse = sstot
    for element in l:
        
        L.append(element)
        mL = np.mean(L)
        nL = len(L)
        R.remove(element)
        mR = np.mean(R)
        nR = len(R)
        ssr = (nL*((mL-m)**2)) + (nR*((mR-m)**2))
        sse = sstot - ssr
        if min_sse > sse:
            min_sse = sse
            min_ml = mL
            min_mr = mR
        
    return (min_ml + min_mr)/2

def stepMiner_transform(Dataset1):
    m,n = Dataset1.shape
    transData = [] 
    for i in range(n):
        threshold =  stepMiner(Dataset1[:,i])
        transData.append(1*(Dataset1[:,i] > threshold))
    return np.array(transData).T


def binary(Dataset1, Dataset2):
    dataset1_dd = Dataset1
    dataset2_dd = Dataset2
    tmp_dataset1 = dataset1_dd
    tmp_dataset2 = dataset2_dd

    tDataset1 = stepMiner_transform(tmp_dataset1)
    tDataset2 = stepMiner_transform(tmp_dataset2)

    matching1to2 = []
    matching1to2_dist = []
    for i in range(len(tDataset1)):
        d1 = np.reshape(tDataset1[i,:], (1,len(tDataset1[i,:])))
        distances = pairwise_distances(d1, tDataset2, metric='manhattan')[0]
        matching1to2.append(np.argsort(distances)[0])
        matching1to2_dist.append(np.sort(distances)[0])
        
    matching2to1 = []
    matching2to1_dist = []
    for i in range(len(tDataset2)):
        d2 = np.reshape(tDataset2[i,:], (1,len(tDataset2[i,:])))
        distances = pairwise_distances(d2, tDataset1, metric='manhattan')[0]
        matching2to1.append(np.argsort(distances)[0])
        matching2to1_dist.append(np.sort(distances)[0])
        

    Matching1to2 = [matching1to2, 1 - matching1to2_dist / np.max(matching1to2_dist)]
    Matching2to1 = [matching2to1, 1 - matching2to1_dist / np.max(matching2to1_dist)]
    return Matching1to2, Matching2to1

def density_downsampling(Data):
    from scipy.spatial import distance
    from scipy.spatial.distance import pdist, squareform
    if len(Data) > 10000:
        tmp = np.random.randint(0, len(Data), size=10000)
        Data = Data[tmp,:]
    dist = distance.pdist(Data, metric='euclidean')    
    dist_m = distance.squareform(dist)
    sorted_dist_m = np.sort(dist_m)
    median_min_dist = np.median(sorted_dist_m[:,1])
    dist_thres = 5 * median_min_dist

    local_densities = np.sum(1*(dist_m < dist_thres),0)

    OD = np.quantile(local_densities, 0.02)
    TD = np.quantile(local_densities, 0.1)

    IDX_TO_KEEP = []
    for i in range(len(local_densities)):
        if local_densities[i] < OD:
            continue
        elif OD < local_densities[i] and local_densities[i] < TD:
            IDX_TO_KEEP.append(i)
        else:  
            prob = TD / local_densities[i]
            if random.random() < prob:
                IDX_TO_KEEP.append(i)
    downsampled_data = Data[IDX_TO_KEEP,:]
    return IDX_TO_KEEP