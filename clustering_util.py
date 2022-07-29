import numpy as np
import networkx as nx
from sklearn.mixture import GaussianMixture
from scipy.linalg import eigh


def ASE(g, dims=2, dim_offset=0):
    """
    Embed undirected graph `g` using Adjacency Spectal Embedding.
    Returns a dims x n numpy array, where n is the number of nodes
    """
    A = nx.adjacency_matrix(g).todense()
    _, eigvec = eigh(A)
    p = eigvec[:, (0 + dim_offset):(dims + dim_offset)]
    return np.asarray(p.T)


def LSE(g, dims=2, eigval_tolerance=1e-10, dim_offset=0):
    """
    Embed undirected graph `g` using Laplacian Spectal Embedding.
    Returns a `dims` x `n` numpy array, where `n` is the number of nodes.
    This uses the Laplacian Eigenmaps algorithm:
    https://web.cse.ohio-state.edu/~belkin.8/papers/LEM_NC_03.pdf

    Parameters:
        g    :   Networkx graph
        dims :   embedding dimension
        eigval_tolerance : exclude eigenvalues smaller than this (see note 1)
        dim_offset : Skip embedding dimensions (see note 2)

    Returns:
        p : a dims x n array, where n is the number of nodes in g

    Notes:
    1.  This uses scipy's eigh function, which isn't guaranteed to compute
        eigenvalues of 0 exactly. So, eigval_tolerance is used to
        exclude the 0 eigenvalues (useful when you don't know how many
        connected components g has
    2.  dim_offset typically doesn't need to be used; if you want to see how
        it embeds in higher dimensions, just use a larger value of dim and
        slice the resulting array.
    """
    A = np.asarray(nx.adjacency_matrix(g).todense())
    D = np.diag(A.sum(axis=0))
    L = D - A
    _, eigvec = eigh(L, D, subset_by_value=[eigval_tolerance, A.shape[0]])
    p = eigvec[:, (0 + dim_offset):(dims + dim_offset)]
    return np.asarray(p.T)


def cluster_member(vector):
    """Return the co-clustering matrix C of a cluster label vector.
    If `vector[i]` is the label of node `i`, then `C[i,j]` is 1 if
    `vector[i] == vector[j]` and 0 otherwise."""
    return np.array([a == vector for a in vector], dtype=int)


def GMM_cluster(p, n_comp=2, init_params="random", n_init=1, **kwargs):
    """Apply Gaussian Mixture Model clustering to data p and return the predicted labels.
    Note that the output of `ASE` and `LSE` will need to be transposed to work properly."""
    model = GaussianMixture(n_components=n_comp, init_params=init_params, n_init=n_init, **kwargs)
    labels = model.fit(p).predict(p)
    return labels

def averaging_cluster(g, times= 3, embeding= LSE, return_labels=None, coclusters=None):
    """calcuating the average clustering for several time of EM algorishm for different initialization.
    the function will return an array of averaging result from EM algorishm at different initializaion"""
    p= embeding(g)
    cocluster = []
    all_labels = []
    for i in range(times):
        labels = GMM_cluster(p.T)
        C_trial = cluster_member(labels)
        cocluster.append(C_trial)
        all_labels.append(labels)
        if coclusters is not None:
            coclusters[i,:,:] = labels
    if return_labels is not None:
        return_labels[:,:] = np.array(all_labels)
    return sum(cocluster)/times

def coclustering_trials(g, n_trials, embedding=LSE):
    """Run GMM clsutering on g for n_trials, and return
    the array of labels (which will be n_trials x n_nodes)
    the array of coclusterings (which will be n_trials x n_nodes x n_nodes
    """
    p = embedding(g)
    k_all = [GMM_cluster(p.T) for _ in range(n_trials)]
    C_all = [cluster_member(k) for k in k_all]
    return np.array(k_all), np.array(C_all)


def quality_score(vector, dim=1,n_init=100, n_comp=2):
    """Given a histogram h, compute its "quality score", meaning..."""
    clustering= GMM_cluster(vector.reshape(-1,dim), n_init=n_init, n_comp=n_comp)
    score= (np.mean(vector[clustering==0])-np.mean(vector[clustering==1]))/(np.std(vector[clustering==0])+np.std(vector[clustering==1]))
    return round(abs(score),0)
