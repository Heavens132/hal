# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.



    # After the Kmeans stops, the function chooses one cluster, and then chooses one group




#def kmeans_choose(groups_list):
    #groups = torch.Tensor(groups_list)
    #centers, cluster_prob = Kmeans(groups, k=1000);
    #group = choose_group(groups, centers, cluster_prob)
    #return group.tolist()


import torch.nn as nn
import torch.nn.functional as F
import torch
import sklearn
import sklearn.cluster

import numpy as np
import scipy.sparse as sp
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.utils.extmath import row_norms, squared_norm, stable_cumsum






################ MLP ###########################################

import os
from pathlib import Path

import torch
import torch.utils.data
import torchvision
import torchvision.transforms

import torch.optim as optim

class MLP(torch.nn.Module):
    NLS = {'relu': torch.nn.ReLU, 'tanh': nn.Tanh, 'sigmoid': nn.Sigmoid, 'softmax': nn.Softmax, 'logsoftmax': nn.LogSoftmax}

    def __init__(self, D_in, hidden_dims, D_out, nonlin='relu'):
        super().__init__()
        
        all_dims = [D_in, *hidden_dims, D_out]
        non_linearity = MLP.NLS[nonlin]
        layers = []
        
        for in_dim, out_dim in zip(all_dims[:-1], all_dims[1:]):
            layers += [
                nn.Linear(in_dim, out_dim, bias=True),
                non_linearity()
            ]
        
        # Sequential is a container for layers
        self.fc_layers = nn.Sequential(*layers[:-1])
        
        # Output non-linearity
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = torch.reshape(x, (x.shape[0], -1))
        z = self.fc_layers(x)
        y_pred = self.log_softmax(z)
        # Output is always log-probability
        return y_pred


################################################################





def _k_init(X, n_clusters, x_squared_norms, random_state, n_local_trials=None):
    """Init n_clusters seeds according to k-means++

    Parameters
    -----------
    X : array or sparse matrix, shape (n_samples, n_features)
        The data to pick seeds for. To avoid memory copy, the input data
        should be double precision (dtype=np.float64).

    n_clusters : integer
        The number of seeds to choose

    x_squared_norms : array, shape (n_samples,)
        Squared Euclidean norm of each data point.

    random_state : numpy.RandomState
        The generator used to initialize the centers.

    n_local_trials : integer, optional
        The number of seeding trials for each center (except the first),
        of which the one reducing inertia the most is greedily chosen.
        Set to None to make the number of trials depend logarithmically
        on the number of seeds (2+log(k)); this is the default.

    Notes
    -----
    Selects initial cluster centers for k-mean clustering in a smart way
    to speed up convergence. see: Arthur, D. and Vassilvitskii, S.
    "k-means++: the advantages of careful seeding". ACM-SIAM symposium
    on Discrete algorithms. 2007

    Version ported from http://www.stanford.edu/~darthur/kMeansppTest.zip,
    which is the implementation used in the aforementioned paper.
    """
    n_samples, n_features = X.shape

    centers = np.empty((n_clusters, n_features), dtype=X.dtype)

    assert x_squared_norms is not None, 'x_squared_norms None in _k_init'

    # Set the number of local seeding trials if none is given
    if n_local_trials is None:
        # This is what Arthur/Vassilvitskii tried, but did not report
        # specific results for other than mentioning in the conclusion
        # that it helped.
        n_local_trials = 2 + int(np.log(n_clusters))

    # Pick first center randomly
    center_id = random_state.randint(n_samples)
    if sp.issparse(X):
        centers[0] = X[center_id].toarray()
    else:
        centers[0] = X[center_id]

    # Initialize list of closest distances and calculate current potential
    closest_dist_sq = euclidean_distances(
        centers[0, np.newaxis], X, Y_norm_squared=x_squared_norms,
        squared=True)
    current_pot = closest_dist_sq.sum()

    # Pick the remaining n_clusters-1 points
    for c in range(1, n_clusters):
        # Choose center candidates by sampling with probability proportional
        # to the squared distance to the closest existing center
        rand_vals = random_state.random_sample(n_local_trials) * current_pot
        candidate_ids = np.searchsorted(stable_cumsum(closest_dist_sq),
                                        rand_vals)

        # Compute distances to center candidates
        distance_to_candidates = euclidean_distances(
            X[candidate_ids], X, Y_norm_squared=x_squared_norms, squared=True)

        # Decide which candidate is the best
        best_candidate = None
        best_pot = None
        best_dist_sq = None
        for trial in range(n_local_trials):
            # Compute potential when including center candidate
            new_dist_sq = np.minimum(closest_dist_sq,
                                     distance_to_candidates[trial])
            new_pot = new_dist_sq.sum()

            # Store result if it is the best local trial so far
            if (best_candidate is None) or (new_pot < best_pot):
                best_candidate = candidate_ids[trial]
                best_pot = new_pot
                best_dist_sq = new_dist_sq

        # Permanently add best center candidate found in local tries
        if sp.issparse(X):
            centers[c] = X[best_candidate].toarray()
        else:
            centers[c] = X[best_candidate]
        current_pot = best_pot
        closest_dist_sq = best_dist_sq

    return centers


def main(_args):
    
    
    def cluster(data, k, num_iter, init = None, cluster_temp=5):
    
        #normalize x so it lies on the unit sphere
        data = torch.diag(1./torch.norm(data, p=2, dim=1)) @ data
        #use kmeans++ initialization if nothing is provided
        if init is None:
            data_np = data.detach().numpy()
            norm = (data_np**2).sum(axis=1)
            init = _k_init(data_np, k, norm, sklearn.utils.check_random_state(None))
            init = torch.tensor(init, requires_grad=True)
            if num_iter == 0: return init
        mu = init
        n = data.shape[0]
        d = data.shape[1]
        #    data = torch.diag(1./torch.norm(data, dim=1, p=2))@data
        for t in range(num_iter):
            #get distances between all data points and cluster centers
            #        dist = torch.cosine_similarity(data[:, None].expand(n, k, d).reshape((-1, d)), mu[None].expand(n, k, d).reshape((-1, d))).reshape((n, k))
            dist = data @ mu.t()
            #cluster responsibilities via softmax
            r = torch.softmax(cluster_temp*dist, 1)
            #total responsibility of each cluster
            cluster_r = r.sum(dim=0)
            #mean of points in each cluster weighted by responsibility
            cluster_mean = (r.t().unsqueeze(1) @ data.expand(k, *data.shape)).squeeze(1)
            #update cluster means
            new_mu = torch.diag(1/cluster_r) @ cluster_mean
            mu = new_mu
        dist = data @ mu.t()
        r = torch.softmax(cluster_temp*dist, 1)
        return mu, r, dist
    
    
    def kmeans(groups_with_passes_num, k, gates_num):
        diff = 1
        # random initialization for centers (there are k centers)
        groups = groups_with_passes_num[:,0:gates_num]
        passes_num = torch.sum(groups_with_passes_num[:,gates_num:], 1)
        torch.reshape(passes_num, (passes_num.shape[0], 1))
        cluster_prob = torch.zeros(groups.shape[0], k)
        cen_indices = torch.randint(0, groups.shape[0] , (k,))
        centers = torch.index_select(groups, 0, cen_indices)
        iterate = 0
        while diff and iterate <1000:
            iterate +=1
            # for each group, calculate the probability that it belongs to each cluster
            for i, row in enumerate(groups):
                dist = float('inf')
                # dist of the group from all centers
                reshaped_row = torch.reshape(row, (1, groups.shape[1]))
                m = torch.nn.Softmin(dim = 0)
                cluster_distr_i = m(torch.norm(reshaped_row - centers, dim=1, p=2))
                cluster_prob[i] = cluster_distr_i

            # calculate the new centers
            new_centers = torch.Tensor()
            for i in range(k):
                weighted_groups = torch.sum(torch.reshape(torch.mul(cluster_prob[:, i], passes_num), (groups.shape[0], 1)) * groups, 0)
                weights_sum = torch.sum(torch.reshape(torch.mul(cluster_prob[:, i], passes_num), (groups.shape[0], 1)), 0)
                temp_center = torch.reshape(weighted_groups / weights_sum, (1,-1))
                new_centers = torch.cat((new_centers, temp_center), 0)
            # if centers did not change then leave
            if torch.count_nonzero(centers - new_centers) == 0:
                diff = 0
            else:
                centers = new_centers
        return centers, cluster_prob
  
    def not_bad_groups_in_cluster(groups, group_cluster, passes_num, i):
    	groups_in_chosen_clu = groups[group_cluster == i]
    	passes_num_in_chosen_clu = passes_num[group_cluster == i]
    	not_bad_groups_ind = [i for i in range(groups_in_chosen_clu.shape[0]) if (torch.count_nonzero(groups_in_chosen_clu[i])>7)]
    	if len(not_bad_groups_ind)==0:
    		return 0
    	return sum([passes_num_in_chosen_clu[i] for i in not_bad_groups_ind])
    	
    def groups_close_to_sizes_in_cluster(groups, group_cluster, passes_num, i, Expected_Reg_Sizes):
    	groups_in_chosen_clu = groups[group_cluster == i]
    	passes_num_in_chosen_clu = passes_num[group_cluster == i]
    	close_size_groups_ind = [i for i in range(groups_in_chosen_clu.shape[0]) if group_close_to_a_size(groups_in_chosen_clu[i],Expected_Reg_Sizes)]
    	if len(close_size_groups_ind)==0:
    		return 0
    	return sum([passes_num_in_chosen_clu[i] for i in close_size_groups_ind])

    def group_close_to_a_size(group, Expected_Reg_Sizes):
    	for size in Expected_Reg_Sizes:
    		if torch.count_nonzero(group) >= (size*0.5) and torch.count_nonzero(group) <= (size*2):
    			return True
    	return False
    
    def group_in_sizes(group, Expected_Reg_Sizes):
    	for size in Expected_Reg_Sizes:
    		if torch.count_nonzero(group) == size:
    			return True
    	return False
    	
    def get_cluster_size(groups, group_cluster, passes_num, i):
        groups_in_chosen_clu = groups[group_cluster == i]
        passes_num_in_chosen_clu = passes_num[group_cluster == i]
        cluster_groups_ind = [i for i in range(groups_in_chosen_clu.shape[0])]
        if len(cluster_groups_ind) == 0:
       	    return 0
        return sum([passes_num_in_chosen_clu[i] for i in cluster_groups_ind])

	
    def choose_group(groups_with_passes_num, centers, cluster_prob ,gates_num, Expected_Reg_Sizes, k):
        groups = groups_with_passes_num[:,0:gates_num]
        passes_num = torch.sum(groups_with_passes_num[:,gates_num:], 1)
        #group_cluster = torch.argmax(cluster_prob, dim=1)
        group_cluster = (torch.multinomial(cluster_prob, num_samples=1)).view(-1)
        clusters_to_choose_from = torch.Tensor([get_cluster_size(groups, group_cluster, passes_num, i) for i in range(k)])
        if(len(Expected_Reg_Sizes)>0):
        	clusters_with_good_sizes = [groups_close_to_sizes_in_cluster(groups, group_cluster, passes_num, i, Expected_Reg_Sizes) for i in range(k)]
        	if max(clusters_with_good_sizes)!=0:
        		clusters_to_choose_from = torch.Tensor(clusters_with_good_sizes)
        	else:
        		clusters_not_all_bad_groups = [not_bad_groups_in_cluster(groups, group_cluster, passes_num, i) for i in range(k)]
        		if max(clusters_not_all_bad_groups)!=0:
        			clusters_to_choose_from = torch.Tensor(clusters_not_all_bad_groups)
        else:
        	clusters_not_all_bad_groups = [not_bad_groups_in_cluster(groups, group_cluster, passes_num, i) for i in range(k)]
        	if max(clusters_not_all_bad_groups)!=0:
        		clusters_to_choose_from = torch.Tensor(clusters_not_all_bad_groups)	
        chosen_cluster = torch.argmax(torch.Tensor(clusters_to_choose_from)).item()
        reshaped_center = torch.reshape(centers[chosen_cluster], (1, -1))
        groups_in_chosen_clu = groups[group_cluster == chosen_cluster]
        passes_num_in_chosen_clu = passes_num [group_cluster == chosen_cluster]
        groups_to_choose_from = groups_in_chosen_clu
        passes_to_choose_from = passes_num_in_chosen_clu
        if(len(Expected_Reg_Sizes)>0):
        	close_to_priority_ind = [i for i in range(groups_in_chosen_clu.shape[0]) if (group_in_sizes(groups_in_chosen_clu[i], Expected_Reg_Sizes))]
        	if len(close_to_priority_ind)>0:
        		groups_to_choose_from=groups_in_chosen_clu[close_to_priority_ind]
        		passes_to_choose_from=passes_num_in_chosen_clu[close_to_priority_ind]
        	else :
        		less_close_to_priority_ind = [i for i in range(groups_in_chosen_clu.shape[0]) if group_close_to_a_size(groups_in_chosen_clu[i],Expected_Reg_Sizes)]
        		if len(less_close_to_priority_ind)>0:
        			groups_to_choose_from=groups_in_chosen_clu[less_close_to_priority_ind]
        			passes_to_choose_from=passes_num_in_chosen_clu[less_close_to_priority_ind]
        		else:
        			not_bad_groups_ind = [i for i in range(groups_in_chosen_clu.shape[0]) if (torch.count_nonzero(groups_in_chosen_clu[i])>7)]
        			if len(not_bad_groups_ind)>0:
        				groups_to_choose_from=groups_in_chosen_clu[not_bad_groups_ind]
        				passes_to_choose_from=passes_num_in_chosen_clu[not_bad_groups_ind]
        else:
        	not_bad_groups_ind = [i for i in range(groups_in_chosen_clu.shape[0]) if (torch.count_nonzero(groups_in_chosen_clu[i])>7)]
        	if len(not_bad_groups_ind)>0:
        		groups_to_choose_from=groups_in_chosen_clu[not_bad_groups_ind]
        		passes_to_choose_from=passes_num_in_chosen_clu[not_bad_groups_ind]
        #normed_dis = torch.norm(groups_to_choose_from - reshaped_center, dim=1,keepdim=True, p=2)
        #best_group_ind = torch.argmin(normed_dis, dim =0).item()
        #best_group_val = normed_dis[best_group_ind].item()
        best_group_ind = torch.argmax(passes_to_choose_from, dim =0).item()
        best_group_val = passes_to_choose_from[best_group_ind].item()
        chosen_groups_ind = [i for i in range(groups_to_choose_from.shape[0]) if passes_to_choose_from[i]>=(best_group_val*0.8)]
        chosen_groups = groups_to_choose_from[chosen_groups_ind]
        return chosen_groups, group_cluster
            
    def loss_centers(dist, cluster_prob):
        dot_res = dist*cluster_prob
        # print('dot_res shape:')
        # print(dot_res.shape)
        sum = torch.sum(dot_res)
        print('sum shape:')
        print(sum.shape)
        return sum
    	
    # def loss_cluster_size(group_cluster, k):
    # 	loss = 0
    # 	avarage = groups.shape[0]/k
    # 	for i in range(k):
    # 		loss+= abs((sum(group_cluster==i)).item()-avarage)
    # 	return loss
    	
        
    groups_file = _args['groups_embedding']
    gates_num = int(_args['gates_num']) 
    Expected_Reg_Sizes = [int(num) for num in (_args['sizes']).split()]
    with open(groups_file) as textFile:
        groups_list = [[int(num) for num in line.split()] for line in textFile.readlines()]
    groups_num = len(groups_list)
    groups = torch.Tensor(groups_list)
    k = max(5, groups_num//100)
    data_embedding = groups[:,0:gates_num]
    
    print('data_embedding shape =')
    print(data_embedding.shape)
    saved_model = Path('../plugins/dataflow_analysis/src/evaluation/the_saved_model.pth')
    MLPmodel = MLP(gates_num, hidden_dims=[32, 32, 32], D_out=64, nonlin='relu')
    optimizer = optim.Adam(MLPmodel.parameters(),lr=5e-2, weight_decay=0.01)
    if saved_model.exists():
        checkpoint = torch.load('../plugins/dataflow_analysis/src/evaluation/the_saved_model.pth')
        MLPmodel.load_state_dict(checkpoint['MLPmodel_state_dict'])
        MLPmodel.eval()
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    optimizer.zero_grad()
    data = MLPmodel(data_embedding)
    centers, cluster_prob, dist_from = cluster(data, k,1000, init = None, cluster_temp=5)
    loss = loss_centers(dist_from, cluster_prob)
    loss.backward()
    optimizer.step()
    ###
    # torch.save(MLPmodel,'../plugins/dataflow_analysis/src/evaluation/the_saved_model.pth')
    torch.save({
            'MLPmodel_state_dict': MLPmodel.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, '../plugins/dataflow_analysis/src/evaluation/the_saved_model.pth')
    
    print('model params:')
    print(list(MLPmodel.parameters()))
    
    print('dist_from shape =')
    print(dist_from.shape)
    print('centers shape =')
    print(centers.shape)
    print('cluster_prob shape =')
    print(cluster_prob.shape)
    
    
    chosen_groups, group_cluster = choose_group(groups, centers, cluster_prob, gates_num, Expected_Reg_Sizes, k)
    #loss = loss_centers(dist_from, group_cluster) + loss_cluster_size(group_cluster, k)
    chosen_groups = chosen_groups.tolist()
    f= open("chosen_group.txt","w+")
    for i in range(len(chosen_groups)):
    	for j in  range(len(chosen_groups[i])):
    		if(chosen_groups[i][j]):
    			f.write(str(j))
    			if(j<len(chosen_groups[i])-1):
    				f.write(" ")
    	f.write("\n")
    f.close()
    	




if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='args')
    parser.add_argument('--groups_embedding', help='file with the embedding of all the groups')
    parser.add_argument('--gates_num', help='the number of all the gates')
    parser.add_argument('--sizes', help='groups with regester size close to Expected_Reg_Sizes will be prioritized')

    _args =  vars(parser.parse_args())
    #_args = {'groups_embedding': './group_embedding.txt' , 'gates_num': '1976', 'sizes': "56 64 "} 
    main(_args)



