#!/usr/bin/env python
import sys
import math
import numpy as np

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import sklearn
from sklearn.cluster import DBSCAN

from root_numpy import root2array, array2tree

from sklearn.decomposition import PCA
from wpca import WPCA, EMPCA

from cluster_common import *

#ROOT.gROOT.SetBatch(1)

max_events = 100
max_parts = 1

max_layers = 28

def store_hits(fname = "../ntuples/hgcalNtuple_ele15_n100_testhelper.root"):

    #hit_type = 'tc'
    hit_type = 'rechit'
    array = get_event_array(fname, hit_type = hit_type, max_events = max_events)

    i_ev = -1

    n_parts = 0

    pca_widths = []
    '''
    fig = plt.figure(figsize=(10, 8))
    ax = plt.subplot(projection = '3d')
    '''

    part_clust_data = []

    hits = np.zeros((1,3))
    enes = np.zeros((1))

    print 80*"#"
    print("Reading data from tree")
    for event in array:

        if n_parts >= max_parts: break

        i_ev += 1

        particles = get_genparticles(event)

        if len(particles) < 1: continue

        n_parts += len(particles)

        ### HITS
        hits_xyz, hits_energies = get_hits(event, hit_type = hit_type, max_layer = max_layers)

        #hits = hits_xyz
        #enes = hits_energies

        #hits.append(hits_xyz)
        #enes.append(hits_energies)
        hits = np.concatenate((hits, hits_xyz))
        enes = np.concatenate((enes, hits_energies))

    # fixing hack
    hits = np.delete(hits, (0), axis = 0)
    enes = np.delete(enes, (0), axis = 0)

    print('Plotting')
    print n_parts

    #ax.scatter(hits[:, 1], hits[:, 0], hits[:,2], s = 0.01, c = 'black')#enes*100)

    if True:

        ## CLUSTER
        #clusters = my_cluster(hits_xyz, hits_energies)
        clusters = my_cluster(hits, enes)

        if clusters:
            part_clust_data.append((particles,clusters))
        #else: continue

        fig = plt.figure(figsize=(10, 8))
        ax = plt.subplot(projection = '3d')

        ax.scatter(hits[:, 1], hits[:, 0], hits[:,2], s = 0.01, c = 'black')#enes*100)

        colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(clusters))]

        for cluster, col in zip(clusters, colors):
            if cluster.pca:
                #print cluster.energy
                pcaw = cluster.pca.explained_variance_
                pcaw /= np.sum(cluster.pca.explained_variance_)
                pca_widths.append(pcaw)
            ## print hits of cluster
            ax.scatter(cluster.hits[:, 1], cluster.hits[:, 0], cluster.hits[:,2], s = cluster.energies*50, c = col)

            # plot outliers
            ax.scatter(cluster.outliers[:, 1], cluster.outliers[:, 0], cluster.outliers[:,2], s = 5, c = col, marker = 'h')

        plt.title('Event %i' %i_ev)

        fig.tight_layout()
        plt.show()
    #analyzer(part_clust_data, figtitle = 'Rechit')
    '''
    #n,_ ,_ = ax.hist(z_arr, np.arange(28), weights = sample_weights)
    pca_widths = np.array(pca_widths)
    #print pca_widths[:,0]
    #print pca_widths[:,1]
    ax.hist(1-pca_widths[:,0], np.linspace(0,0.1,50))
    ax.hist(pca_widths[:,1], np.linspace(0,0.1,50))
    ax.hist(pca_widths[:,2], np.linspace(0,0.1,50))

    fig.tight_layout()

    #if not ax.empty:
    print i_ev, n_parts
    plt.title('RH')

    plt.show()
    '''

def main(fname):
    global mip_calibs

    #mip_calibs = read_mip_calib()
    store_hits(fname)

if __name__ == "__main__":

    #if '-b' in sys.argv: sys.argv = [sys.argv[0]]

    if len(sys.argv) > 1:
        if '-b' in sys.argv:
            fname = sys.argv[1]
        else:
            fname = sys.argv[1]
        #print '# Input file is', fname
        main(fname)
    else:
        #print("No input files given!")
        main()

    # load tree
    #load_tree()
