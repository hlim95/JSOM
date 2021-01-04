import numpy as np
import random
from sklearn.metrics.pairwise import euclidean_distances
import pandas as pd
from tqdm import tqdm
import sys
import argparse
from sklearn.metrics import pairwise_distances
from utils import *
import os
from os import path
    #trying to make shift_graphs directory if it does not already exist:

    


class JSOM(object):
    def __init__(self, data1, data2, matching1, matching2, m, n, n_iterations, ratio=None, alpha=None, sigma=None, random_seed=None):
        # data1 should be a np array of size [# of samples , # of features ]
        # data2 should be a np array of size [# of samples , # of features ]
        # matching1 is a np array of size [# of samples for data1, ]. Each element, Xi, 
                            #is the index of its(i-th data from data1) closest data in data2 
        # matching2 is a np array of size [# of samples for data2 ]. Each element, Xi, 
                            #is the index of its(i-th data from data2) closest data in data1
        # m = # of SOM nodes in x-axis 
        # n = # of SOM nodes in y-axis 
        # n_iteration = # of times it fits 
        
        #Assign required variables first
        self.Data1 = data1
        self.Data2 = data2
        self.Matching1to2 = matching1[0]
        self.Matching2to1 = matching2[0]
        self.Matching1to2_weight = matching1[1]
        self.Matching2to1_weight = matching2[1]

        self.m = m
        self.n = n
        self.num_runs = n_iterations
        if alpha is None:
            self.alpha = 0.9
        else:
            self.alpha = float(alpha)
        if sigma is None:
            self.sigma = np.sqrt(2)*(np.maximum(m,n)-1)/3
        else:
            self.sigma = float(sigma)
        
        if ratio is None:
            self.ratio = 4
        else:
            self.ratio = int(ratio)

        if random_seed is None:
            self.random_seed = 78
        else:
            self.random_seed = args.randomseed  
        
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        
        self.NN = m*n
        self.min_dist1 = np.median(np.sort(euclidean_distances(data1))[:,1])
        self.min_dist2 = np.median(np.sort(euclidean_distances(data2))[:,1])
    def train(self):    
        self.nodes_coord = []
        ct = 0
        for i in range(self.m):
            for j in range(self.n):
                ct = ct + 1
                self.nodes_coord.append([i,j])
        self.nodes_coord = np.array(self.nodes_coord)
        
        random_index_1 = np.random.randint(0, len(self.Data1)-1, size=self.NN)
        random_index_2 = np.random.randint(0, len(self.Data2)-1, size=self.NN)

        ##random initiazliation 
        permData1 = np.random.permutation(np.random.permutation(self.Data1).T).T
        permData2 = np.random.permutation(np.random.permutation(self.Data2).T).T

        ct = 0
        self.nodes_vectors_array_1 = []
        self.nodes_vectors_array_2 = []
        for i in range(self.NN):
            ct = ct + 1
            self.nodes_vectors_array_1.append(permData1[random_index_1[i],:])
            self.nodes_vectors_array_2.append(permData2[random_index_2[i],:])
        self.nodes_vectors_array_1 = np.array(self.nodes_vectors_array_1, dtype=float)
        self.nodes_vectors_array_2 = np.array(self.nodes_vectors_array_2, dtype=float)
        self.initial_nodes_vectors_array_1 = self.nodes_vectors_array_1[:]
        self.initial_nodes_vectors_array_2 = self.nodes_vectors_array_2[:]
        input_order1 = np.random.permutation(len(self.Data1))
        input_order2 = np.random.permutation(len(self.Data2))
        input_order3 = np.random.permutation(len(self.Data1))
        input_order4 = np.random.permutation(len(self.Data2))
        
        ct_d1 = 0 
        ct_d2 = 0
        ct_d3 = 0
        ct_d4 = 0
            
        for run in tqdm(range(self.num_runs)):
            # if run % 3000 == 0:
            #     print('Training...', run, 'out of', self.num_runs)
            e = self.sigma*(1-run/self.num_runs)
            if e < np.sqrt(2):
                # e = np.sqrt(2) + 0.000001 
                e = np.sqrt(2) - 0.000001 
            if run < np.ceil(0.2 * self.num_runs):
                alpha = self.alpha
            else:
                alpha = self.alpha*(run-(np.ceil(0.2 * self.num_runs)-1))*(-1/(self.num_runs-(np.ceil(0.2 * self.num_runs)-1)))+1
            #coupled 
            if run % self.ratio == 0:
                ct_d1 = ct_d1 + 1
                input_from_1 = np.array(self.Data1[input_order1[ct_d1-1],:], dtype = float)
                input_from_1 = np.reshape(input_from_1, (1,len(input_from_1)))
                distance_to_nodes = euclidean_distances(input_from_1, self.nodes_vectors_array_1)
                nn = np.argmin(distance_to_nodes)
                df1 = self.Matching1to2_weight[input_order1[ct_d1-1]]
                df2 = self.min_dist1 / (np.min(distance_to_nodes) + 0.0001)
                # df = df1 * df2
                df = df1
                if df > 1:
                    df = 1
                # print(df)
                
                mapping_index = self.Matching1to2[input_order1[ct_d1-1]]
                im_input_for2 = self.Data2[mapping_index,:]
                for node in range(self.NN):
                    a = np.reshape(self.nodes_coord[node], (1,len(self.nodes_coord[node])))
                    b = np.reshape(self.nodes_coord[nn], (1,len(self.nodes_coord[nn])))
                    if euclidean_distances(a,b)[0] < e:
                        # self.nodes_vectors_array_1[node,:] = self.nodes_vectors_array_1[node,:] + (alpha* df) * (input_from_1 - self.nodes_vectors_array_1[node,:])
                        self.nodes_vectors_array_1[node,:] = self.nodes_vectors_array_1[node,:] + (df*alpha) * (input_from_1 - self.nodes_vectors_array_1[node,:])
                        
                        self.nodes_vectors_array_2[node,:] = self.nodes_vectors_array_2[node,:] + (df*alpha) * (im_input_for2 - self.nodes_vectors_array_2[node,:])
                        # self.nodes_vectors_array_1[node,:] = self.nodes_vectors_array_1[node,:] + (alpha) * (input_from_1 - self.nodes_vectors_array_1[node,:])
                        # self.nodes_vectors_array_2[node,:] = self.nodes_vectors_array_2[node,:] + (alpha) * (im_input_for2 - self.nodes_vectors_array_2[node,:])
                        
                        # print(run, alpha* df, e)


                ct_d2 = ct_d2 + 1
                input_from_2 = np.array(self.Data2[input_order2[ct_d2-1],:], dtype = float)
                input_from_2 = np.reshape(input_from_2, (1,len(input_from_2)))
                distance_to_nodes = euclidean_distances(input_from_2, self.nodes_vectors_array_2)
                nn = np.argmin(distance_to_nodes)
                #print(ct_d2-1,input_order2[ct_d2-1] )
                df1 = self.Matching2to1_weight[input_order2[ct_d2-1]]
                df2 = self.min_dist2 / (np.min(distance_to_nodes) + 0.0001)
                #df = df1 * df2
                df = df1
                if df > 1:
                    df = 1
                # print(df)

                mapping_index = self.Matching2to1[input_order2[ct_d2-1]]
                im_input_for1 = self.Data1[mapping_index,:]
                for node in range(self.NN):
                    a = np.reshape(self.nodes_coord[node], (1,len(self.nodes_coord[node])))
                    b = np.reshape(self.nodes_coord[nn], (1,len(self.nodes_coord[nn])))
                    if euclidean_distances(a,b)[0] < e:
                        # self.nodes_vectors_array_1[node,:] = self.nodes_vectors_array_1[node,:] + (alpha*df) * (im_input_for1 - self.nodes_vectors_array_1[node,:])
                        self.nodes_vectors_array_2[node,:] = self.nodes_vectors_array_2[node,:] + (df*alpha) * (input_from_2 - self.nodes_vectors_array_2[node,:])
                        
                        # self.nodes_vectors_array_2[node,:] = self.nodes_vectors_array_2[node,:] + (alpha*df) * (input_from_2 - self.nodes_vectors_array_2[node,:])
                        self.nodes_vectors_array_1[node,:] = self.nodes_vectors_array_1[node,:] + (df*alpha) * (im_input_for1 - self.nodes_vectors_array_1[node,:])
                        # self.nodes_vectors_array_2[node,:] = self.nodes_vectors_array_2[node,:] + (alpha) * (input_from_2 - self.nodes_vectors_array_2[node,:])
                            
                        # print(run, alpha* df)

            else:
                ct_d3 = ct_d3 + 1
                input_from_1 = np.array(self.Data1[input_order3[ct_d3-1],:], dtype = float)
                input_from_1 = np.reshape(input_from_1, (1,len(input_from_1)))
                distance_to_nodes = euclidean_distances(input_from_1, self.nodes_vectors_array_1)
                nn = np.argmin(distance_to_nodes)
                for node in range(self.NN):
                    a = np.reshape(self.nodes_coord[node], (1,len(self.nodes_coord[node])))
                    b = np.reshape(self.nodes_coord[nn], (1,len(self.nodes_coord[nn])))
                    if euclidean_distances(a,b)[0] < e:
                        self.nodes_vectors_array_1[node,:] = self.nodes_vectors_array_1[node,:] + alpha * (input_from_1 - self.nodes_vectors_array_1[node,:])

                ct_d4 = ct_d4 + 1
                input_from_2 = np.array(self.Data2[input_order4[ct_d4-1],:], dtype = float)
                input_from_2 = np.reshape(input_from_2, (1,len(input_from_2)))
                distance_to_nodes = euclidean_distances(input_from_2, self.nodes_vectors_array_2)
                nn = np.argmin(distance_to_nodes)
                for node in range(self.NN):
                    a = np.reshape(self.nodes_coord[node], (1,len(self.nodes_coord[node])))
                    b = np.reshape(self.nodes_coord[nn], (1,len(self.nodes_coord[nn])))
                    if euclidean_distances(a,b)[0] < e:
                        self.nodes_vectors_array_2[node,:] = self.nodes_vectors_array_2[node,:] + alpha * (input_from_2 - self.nodes_vectors_array_2[node,:])
            if ct_d1 > len(self.Data1)-1:
                ct_d1 = 0
            if ct_d2 > len(self.Data2)-1:
                ct_d2 = 0
            if ct_d3 > len(self.Data1)-1:
                ct_d3 = 0
            if ct_d4 > len(self.Data2)-1:
                ct_d4 = 0
        print('training_done')
        
    def initial_vectors(self):
        return self.initial_nodes_vectors_array_1, self.initial_nodes_vectors_array_2

    def nodes_weights(self):
        return self.nodes_vectors_array_1, self.nodes_vectors_array_2

    def mapping_main(self, Data1, Data2):
        NODES_1 = []
        for i in range(len(Data1)):
            input_from_1 = np.array(Data1[i,:], dtype = float)
            input_from_1 = np.reshape(input_from_1, (1,len(input_from_1)))
            distance_to_nodes = euclidean_distances(input_from_1, self.nodes_vectors_array_1)
            nn = np.argmin(distance_to_nodes)
            NODES_1.append(nn)

        NODES_2 = []
        for i in range(len(Data2)):
            input_from_2 = np.array(Data2[i,:], dtype = float)
            input_from_2 = np.reshape(input_from_2, (1,len(input_from_2)))
            distance_to_nodes = euclidean_distances(input_from_2, self.nodes_vectors_array_2)
            nn = np.argmin(distance_to_nodes)
            NODES_2.append(nn)
        NODES_1 = np.array(NODES_1)
        NODES_2 = np.array(NODES_2)        
        SUMMARY = np.zeros((1,4))
        
        self.Data1_nodes = []
        self.Data2_nodes = []
        for i in range(self.NN):
            summary = np.zeros((1,4))
            index1 = np.where(NODES_1 == i)[0];
            index2 = np.where(NODES_2 == i)[0];
            summary[0,0]=len(index1)
            summary[0,1]=len(index2)
            if len(index1)+len(index2) == 0:
                summary[0,2] = 0
                summary[0,3] = 0
            else:
                summary[0,2] = len(index1)/(len(index1) + len(index2))
                summary[0,3] = len(index2)/(len(index1) + len(index2))
            #print(summary)
            SUMMARY = np.vstack((SUMMARY,summary))
            self.Data1_nodes.append(index1)
            self.Data2_nodes.append(index2)
        SUMMARY = SUMMARY[1:,:]
        ENTROPY = []
        for i in range(self.NN):
            if SUMMARY[i,2] == 0 or SUMMARY[i,3] == 0:
                entropy = 0
            else:
                entropy=-SUMMARY[i,2]*np.log2(SUMMARY[i,2])-SUMMARY[i,3]*np.log2(SUMMARY[i,3])
            ENTROPY.append(entropy)

        self.High_E_Index = np.where(np.array(ENTROPY) > 0.2)[0]        
        self.NODE_LOC_1to2 = []
        IDX = []
        for i in range(len(self.Data1)):
            for j in range(len(self.Data1_nodes)):
                node_cell = self.Data1_nodes[j]
                if i in node_cell:
                    node_loc = self.Data2_nodes[j]
                    distance = 1.1
                    while len(node_loc) == 0:
                        neighbors = []
                        for node in range(self.NN):
                            a = np.reshape(self.nodes_coord[node], (1,len(self.nodes_coord[node])))
                            b = np.reshape(self.nodes_coord[j], (1,len(self.nodes_coord[j])))
                            if euclidean_distances(a,b)[0] < distance and len(self.Data2_nodes[node]) > 0:
                                for element in range(len(self.Data2_nodes[node])):
                                    neighbors.append(self.Data2_nodes[node][element])                  
                        node_loc = np.array(neighbors)
                        distance = distance + 1
                    self.NODE_LOC_1to2.append(node_loc)                   
        
        self.NODE_LOC_2to1 = []
        IDX = []
        for i in range(len(self.Data2)):
            for j in range(len(self.Data2_nodes)):
                node_loc = self.Data2_nodes[j]
                if i in node_loc:
                    node_cell = self.Data1_nodes[j]
                    distance = 1.1
                    while len(node_cell) == 0:
                        neighbors = []
                        for node in range(self.NN):
                            a = np.reshape(self.nodes_coord[node], (1,len(self.nodes_coord[node])))
                            b = np.reshape(self.nodes_coord[j], (1,len(self.nodes_coord[j])))
                            if euclidean_distances(a,b)[0] < distance and len(self.Data1_nodes[node]) > 0:
                                for element in range(len(self.Data1_nodes[node])):
                                    neighbors.append(self.Data1_nodes[node][element])                  
                        node_cell = np.array(neighbors)
                        distance = distance + 1
                    self.NODE_LOC_2to1.append(node_cell)
        print('Mapping Done')
        return self.NODE_LOC_1to2, self.NODE_LOC_2to1, self.High_E_Index 



    def mapping(self):
        NODES_1 = []
        for i in range(len(self.Data1)):
            input_from_1 = np.array(self.Data1[i,:], dtype = float)
            input_from_1 = np.reshape(input_from_1, (1,len(input_from_1)))
            distance_to_nodes = euclidean_distances(input_from_1, self.nodes_vectors_array_1)
            nn = np.argmin(distance_to_nodes)
            NODES_1.append(nn)

        NODES_2 = []
        for i in range(len(self.Data2)):
            input_from_2 = np.array(self.Data2[i,:], dtype = float)
            input_from_2 = np.reshape(input_from_2, (1,len(input_from_2)))
            distance_to_nodes = euclidean_distances(input_from_2, self.nodes_vectors_array_2)
            nn = np.argmin(distance_to_nodes)
            NODES_2.append(nn)
        NODES_1 = np.array(NODES_1)
        NODES_2 = np.array(NODES_2)        
        SUMMARY = np.zeros((1,4))
        
        self.Data1_nodes = []
        self.Data2_nodes = []
        for i in range(self.NN):
            summary = np.zeros((1,4))
            index1 = np.where(NODES_1 == i)[0];
            index2 = np.where(NODES_2 == i)[0];
            summary[0,0]=len(index1)
            summary[0,1]=len(index2)
            if len(index1)+len(index2) == 0:
                summary[0,2] = 0
                summary[0,3] = 0
            else:
                summary[0,2] = len(index1)/(len(index1) + len(index2))
                summary[0,3] = len(index2)/(len(index1) + len(index2))
            #print(summary)
            SUMMARY = np.vstack((SUMMARY,summary))
            self.Data1_nodes.append(index1)
            self.Data2_nodes.append(index2)
        SUMMARY = SUMMARY[1:,:]
        ENTROPY = []
        for i in range(self.NN):
            if SUMMARY[i,2] == 0 or SUMMARY[i,3] == 0:
                entropy = 0
            else:
                entropy=-SUMMARY[i,2]*np.log2(SUMMARY[i,2])-SUMMARY[i,3]*np.log2(SUMMARY[i,3])
            ENTROPY.append(entropy)

        self.High_E_Index = np.where(np.array(ENTROPY) > 0.2)[0]        
        self.NODE_LOC_1to2 = []
        IDX = []
        for i in range(len(self.Data1)):
            for j in range(len(self.Data1_nodes)):
                node_cell = self.Data1_nodes[j]
                if i in node_cell:
                    node_loc = self.Data2_nodes[j]
                    distance = 1.1
                    while len(node_loc) == 0:
                        neighbors = []
                        for node in range(self.NN):
                            a = np.reshape(self.nodes_coord[node], (1,len(self.nodes_coord[node])))
                            b = np.reshape(self.nodes_coord[j], (1,len(self.nodes_coord[j])))
                            if euclidean_distances(a,b)[0] < distance and len(self.Data2_nodes[node]) > 0:
                                for element in range(len(self.Data2_nodes[node])):
                                    neighbors.append(self.Data2_nodes[node][element])                  
                        node_loc = np.array(neighbors)
                        distance = distance + 1
                    self.NODE_LOC_1to2.append(node_loc)                   
        
        self.NODE_LOC_2to1 = []
        IDX = []
        for i in range(len(self.Data2)):
            for j in range(len(self.Data2_nodes)):
                node_loc = self.Data2_nodes[j]
                if i in node_loc:
                    node_cell = self.Data1_nodes[j]
                    distance = 1.1
                    while len(node_cell) == 0:
                        neighbors = []
                        for node in range(self.NN):
                            a = np.reshape(self.nodes_coord[node], (1,len(self.nodes_coord[node])))
                            b = np.reshape(self.nodes_coord[j], (1,len(self.nodes_coord[j])))
                            if euclidean_distances(a,b)[0] < distance and len(self.Data1_nodes[node]) > 0:
                                for element in range(len(self.Data1_nodes[node])):
                                    neighbors.append(self.Data1_nodes[node][element])                  
                        node_cell = np.array(neighbors)
                        distance = distance + 1
                    self.NODE_LOC_2to1.append(node_cell)
        print('Mapping Done')
        return self.NODE_LOC_1to2, self.NODE_LOC_2to1, self.High_E_Index 
   
    def data_per_nodes(self):
        return self.Data1_nodes, self.Data2_nodes



def generate_results(jsom, Dataset1, Dataset2):
    from scipy import stats
    from scipy.cluster import hierarchy
    if not os.path.exists('results'):
        os.makedirs('results')
    nodes1, nodes2 = jsom.nodes_weights()
    Results = []
    ###
    print('further clustering into {} clusters'.format(args.numClusters))
    NODES_1 = []
    for i in range(len(Dataset1)):
        input_from_1 = np.array(Dataset1[i,:], dtype = float)
        input_from_1 = np.reshape(input_from_1, (1,len(input_from_1)))
        distance_to_nodes = euclidean_distances(input_from_1, nodes1)
        nn = np.argmin(distance_to_nodes)
        NODES_1.append(nn)
    NODES_2 = []
    for i in range(len(Dataset2)):
        input_from_2 = np.array(Dataset2[i,:], dtype = float)
        input_from_2 = np.reshape(input_from_2, (1,len(input_from_2)))
        distance_to_nodes = euclidean_distances(input_from_2, nodes2)
        nn = np.argmin(distance_to_nodes)
        NODES_2.append(nn)

    NODES_1 = np.array(NODES_1)
    NODES_2 = np.array(NODES_2)
      
    if args.numClusters != 0:
        nodes_comb = np.hstack((nodes1,nodes2))
        Z = hierarchy.linkage(nodes_comb, 'ward')
        Nodes_Clustered = hierarchy.fcluster(Z, t=args.numClusters, criterion= 'maxclust')

        data1_nodes, data2_nodes = jsom.data_per_nodes()

        d1_z = []
        for i in NODES_1:
            d1_z.append(Nodes_Clustered[i])
        d2_z = []
        for i in NODES_2:
            d2_z.append(Nodes_Clustered[i])
        d1_z = np.array(d1_z)
        d2_z = np.array(d2_z)

        tmp1 = pd.DataFrame(d1_z, dtype = int).T#.to_csv('test.csv')
        tmp1.to_csv('./results/data1_cluster_assignment.csv',index=False)
        tmp2 = pd.DataFrame(d2_z, dtype = int).T#.to_csv('test.csv')
        tmp2.to_csv('./results/data2_cluster_assignment.csv',index=False)

        Results_cl = []
        Names_cl = []
        for cluster in np.unique(Nodes_Clustered):
            cluster_idx = np.where(Nodes_Clustered == cluster)[0]
            temp1 = [list(data1_nodes[i]) for i in cluster_idx]
            temp1 = [item for sublist in temp1 for item in sublist]
            temp2 = [list(data2_nodes[i]) for i in cluster_idx]
            temp2 = [item for sublist in temp2 for item in sublist]
            Results_cl.append(np.array(temp1).astype(int))
            Results_cl.append(np.array(temp2).astype(int))
            Names_cl.append('Cluster{}-d1'.format(cluster+1))
            Names_cl.append('Cluster{}-d2'.format(cluster+1))

        tmp = pd.DataFrame(Results_cl, dtype = int).T#.to_csv('test.csv')
        tmp.columns = Names_cl
        tmp.to_csv('./results/data_by_clusters.csv',index=False)

        Results = []
        Names = []
        for i in range(num_x * num_y):
            idx1 = np.where(NODES_1==i+1)[0]
            idx2 = np.where(NODES_2==i+1)[0]
            if args.fillNearest == 0:
                Results.append(np.array(idx1).astype(int))
                Results.append(np.array(idx2).astype(int))
                Names.append('Node{}-d1'.format(i+1))
                Names.append('Node{}-d2'.format(i+1))
            
            elif args.fillNearest == 1:
                if len(idx1) == 0 and len(idx2) != 0:
                    node_vector = nodes1[i,:]
                    node_vector = np.reshape(node_vector, (len(node_vector),1))
                    distance_to_nodes = euclidean_distances(node_vector.T, nodes1)
                    b=np.argsort(distance_to_nodes)[0]
                    ct = 0
                    while len(idx1) == 0:
                        nearest_node = b[ct+1]
                        idx1 = np.where(NODES_1==nearest_node)[0]
                        
                if len(idx2) == 0 and len(idx1) != 0:
                    node_vector = nodes2[i,:]
                    node_vector = np.reshape(node_vector, (len(node_vector),1))
                    distance_to_nodes = euclidean_distances(node_vector.T, nodes2)
                    b=np.argsort(distance_to_nodes)[0]
                    ct = 0
                    while len(idx2) == 0:
                        nearest_node = b[ct+1]
                        idx2 = np.where(NODES_2==nearest_node)[0]

                Results.append(np.array(idx1).astype(int))
                Results.append(np.array(idx2).astype(int))
                Names.append('Node{}-d1'.format(i+1))
                Names.append('Node{}-d2'.format(i+1))

            else:
                continue

        tmp = pd.DataFrame(Results, dtype = int).T#.to_csv('test.csv')
        tmp.columns = Names
        tmp.to_csv('./results/data_by_nodes.csv',index=False)

    else:
        Results = []
        Names = []
        for i in range(num_x * num_y):
            idx1 = np.where(NODES_1==i+1)[0]
            idx2 = np.where(NODES_2==i+1)[0]
            if args.fillNearest == 0:
                Results.append(np.array(idx1).astype(int))
                Results.append(np.array(idx2).astype(int))
                Names.append('Node{}-d1'.format(i+1))
                Names.append('Node{}-d2'.format(i+1))
            
            elif args.fillNearest == 1:
                if len(idx1) == 0 and len(idx2) != 0:
                    node_vector = nodes1[i,:]
                    node_vector = np.reshape(node_vector, (len(node_vector),1))
                    distance_to_nodes = euclidean_distances(node_vector.T, nodes1)
                    b=np.argsort(distance_to_nodes)[0]
                    ct = 0
                    while len(idx1) == 0:
                        nearest_node = b[ct+1]
                        idx1 = np.where(NODES_1==nearest_node)[0]
                        
                if len(idx2) == 0 and len(idx1) != 0:
                    node_vector = nodes2[i,:]
                    node_vector = np.reshape(node_vector, (len(node_vector),1))
                    distance_to_nodes = euclidean_distances(node_vector.T, nodes2)
                    b=np.argsort(distance_to_nodes)[0]
                    ct = 0
                    while len(idx2) == 0:
                        nearest_node = b[ct+1]
                        idx2 = np.where(NODES_2==nearest_node)[0]

                Results.append(np.array(idx1).astype(int))
                Results.append(np.array(idx2).astype(int))
                Names.append('Node{}-d1'.format(i+1))
                Names.append('Node{}-d2'.format(i+1))

            else:
                continue

        tmp = pd.DataFrame(Results, dtype = int).T#.to_csv('test.csv')
        tmp.columns = Names
        tmp.to_csv('./results/data_by_nodes.csv',index=False)
                
        # Results.append(np.array(idx1).astype(int))
        # Results.append(np.array(idx2).astype(int))
        # Names.append('Node{}-d1'.format(i+1))
        # Names.append('Node{}-d2'.format(i+1))

    tmp1 = pd.DataFrame(NODES_1, dtype = int).T
    tmp1.to_csv('./results/data1_node_assignment.csv',index=False)
    tmp2 = pd.DataFrame(NODES_2, dtype = int).T
    tmp2.to_csv('./results/data2_node_assignment.csv',index=False)

    print('Results Exported')


if __name__ == "__main__":
    #### PARSER ####
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--file1', 
                        type=str, 
                        default="./data/sample1.csv",
                        help='Path to file 1')
    parser.add_argument('--file2', 
                        type=str, 
                        default="./data/sample2.csv",
                        help='Path to file 2')
    
    parser.add_argument('--matching1', 
                        type=str, 
                        default="None",
                        help='Path to matching file 1')
    
    parser.add_argument('--matching2', 
                        type=str, 
                        default="None",
                        help='Path to matching file 2')

    parser.add_argument('--method', 
                        type=str, 
                        default="correlation",
                        help='correlation or binary')

    parser.add_argument('--fillNearest', 
                        type=int, 
                        default=0,
                        help='If no data were assigned from one dataset, fill with nearest node information')

    parser.add_argument('--density_downsampling', 
                        type=bool, 
                        default=False,
                        help='Perform density downsampling')

    parser.add_argument("--node_x",
                        type=int,
                        default=10,
                        help="Number of Node along X-axis")
    
    parser.add_argument("--node_y",
                        type=int,
                        default=10,
                        help="Number of Node along y-axis")

    parser.add_argument("--epochs",
                        type=int,
                        default=3,
                        help="Number of epochs")

    parser.add_argument("--ratio",
                        type=int,
                        default=1,
                        help="Number of ratio used for the joint training")

    parser.add_argument("--numClusters",
                        type=int,
                        default=0,
                        help="Number of Clusters")

    parser.add_argument("--randomseed",
                        type=int,
                        default=78,
                        help="Random Seed")



    args = parser.parse_args()
    print(args)

    data_dir1 = args.file1
    if data_dir1[0] != '.':
        data_dir1 = '.' + data_dir1
    data_dir2 = args.file2
    if data_dir2[0] != '.':
        data_dir2 = '.' + data_dir2
    original_Dataset1 = np.array(pd.read_csv(data_dir1, header = None))
    original_Dataset2 = np.array(pd.read_csv(data_dir2, header = None))

    if args.matching1 == 'None':
        matching_Dataset1 = original_Dataset1
    else:
        if args.matching1[0] != '.':
            args.matching1 = '.' + args.matching1
        matching_Dataset1 = np.array(pd.read_csv(args.matching1, header = None))
    if args.matching2 == 'None':
        matching_Dataset2 = original_Dataset2
    else:
        if args.matching2[0] != '.':
            args.matching2 = '.' + args.matching2
        matching_Dataset2 = np.array(pd.read_csv(args.matching2, header = None))


    if len(original_Dataset1) > 10000:
        tmp= np.random.randint(0, len(self.Data1)-1, 10000)
        Dataset1 = original_Dataset1[tmp,:]
        matching_Dataset1 = matching_Dataset1[tmp,:]
    else: 
        Dataset1 = original_Dataset1

    if len(original_Dataset2) > 10000:
        tmp= np.random.randint(0, len(self.Data2)-1, 10000)
        Dataset2 = original_Dataset2[tmp,:]
        matching_Dataset2 = matching_Dataset2[tmp,:]
    else:
        Dataset2 = original_Dataset2


    if args.density_downsampling == True:
        dd_idx1 = density_downsampling(Dataset1)
        dd_idx2 = density_downsampling(Dataset2)
        Dataset1 = Dataset1[dd_idx1,:]
        Dataset2 = Dataset2[dd_idx2,:]
        matching_Dataset1 = matching_Dataset1[dd_idx1,:] 
        matching_Dataset2 = matching_Dataset2[dd_idx2,:]



    if args.method == 'correlation':
        Matching1to2, Matching2to1 = correlation(matching_Dataset1, matching_Dataset2)
    elif args.method == 'binary':
        Matching1to2, Matching2to1 = binary(matching_Dataset1, matching_Dataset2)
    else:
        print('Erorr : No matching provided')

    num_x = args.node_x 
    num_y = args.node_y
    num_run = args.epochs * np.max([len(Dataset1),len(Dataset2)])
    jsom = JSOM(Dataset1, Dataset2, Matching1to2, Matching2to1, num_x, num_y, num_run, args.ratio)
    jsom.train()
    jsom.mapping_main(original_Dataset1,original_Dataset2)
    generate_results(jsom, Dataset1, Dataset2)

    





