#!/usr/bin/env python
import sys
import math
import numpy as np
#import cPickle as pickle

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import sklearn
from sklearn.cluster import DBSCAN

from root_numpy import root2array, array2tree

from sklearn.decomposition import PCA
from wpca import WPCA, EMPCA

from cluster_common import *

max_events = 1000
max_parts = 1000

max_layers = 28

def store_hits(fname = "../ntuples/hgcalNtuple_ele15_n100_testhelper.root"):

    branches = ["genpart_gen","genpart_reachedEE","genpart_pt","genpart_energy","genpart_pid", "genpart_eta", "genpart_phi","genpart_posx","genpart_posy","genpart_posz"]
    #branches = ["genpart_gen","genpart_reachedEE","genpart_energy","genpart_eta", "genpart_pid","genpart_posx","genpart_posy","genpart_posz"]
    branches += ["tc_x", "tc_y", "tc_z", "tc_energy","tc_layer"]
    array = root2array(fname,
                       treename='hgcalTriggerNtuplizer/HGCalTriggerNtuple',
                       branches = branches,
                       stop = max_events
    )

    i_ev = -1

    n_parts = 0

    pca_widths = []
    part_clust_data = []
    #fig = plt.figure(figsize=(10, 8))
    #ax = plt.subplot()

    for event in array:

        if n_parts > max_parts: break

        i_ev += 1

        selected_genparts = (event['genpart_gen'] > 0)
        selected_genparts &= (event['genpart_reachedEE'] > 1)#
        selected_genparts &= (event['genpart_energy'] > 0.1)
        selected_genparts &= (event['genpart_eta'] > 0)
        #selected_genparts &= (event['genpart_eta'] > 2.)
        #selected_genparts &= (abs(event['genpart_pid']) == 22)
        #selected_genparts &= (abs(event['genpart_eta']) > 2.3)

        x_arr = event['genpart_posx'][selected_genparts]
        y_arr = event['genpart_posy'][selected_genparts]
        z_arr = event['genpart_posz'][selected_genparts]

        if len(x_arr) < 1: continue
        #print "Found %i gen particles" % len(x_arr)
        n_parts += len(x_arr)

        eta_arr = event['genpart_eta'][selected_genparts]
        phi_arr = event['genpart_phi'][selected_genparts]
        ene_arr = event['genpart_energy'][selected_genparts]

        #fig = plt.figure(figsize=(10, 8))
        #ax = plt.subplot(projection='3d')

        particles = []
        for i_part,xa in enumerate(x_arr):
            particles.append(Particle( ene_arr[i_part], eta_arr[i_part], phi_arr[i_part] ))

            ##print x_arr[i_part].shape
            if len(x_arr[i_part]) < 1: continue

            ## Plot particle trajectory
            max_lay = min(40,max_layers)
            #ax.plot(x_arr[i_part][:max_layers],z_arr[i_part][:max_layers],y_arr[i_part][:max_layers], '--b')
            layers = np.array(range(1,max_lay+1))
            #ax.plot(x_arr[i_part][:max_lay],layers,y_arr[i_part][:max_lay], '--b')

        if len(x_arr[i_part]) < 1: continue

        ### HITS
        sel_hit_indices = (event['tc_energy'] > 0.01)
        sel_hit_indices &= (event['tc_layer'] < max_layers)
        #sel_hit_indices &= (event['tc_layer'] %4 == 0 )
        sel_hit_indices &= (event['tc_z'] > 0.)

        x_arr = event['tc_x'][sel_hit_indices]
        y_arr = event['tc_y'][sel_hit_indices]
        z_arr = event['tc_z'][sel_hit_indices]
        #z_arr = event['tc_layer'][sel_hit_indices]

        sample_weights = event['tc_energy'][sel_hit_indices]

        ## rescale Z
        #z_arr -= 320
        #z_arr /= 10.

        if len(sample_weights) < 20: continue
        #print("Found %i hits" % len(sample_weights))

        ##cluster
        X = np.column_stack((z_arr,x_arr,y_arr))
        clusters = my_cluster(X, sample_weights)

        '''
        for cluster in clusters:
            if cluster.pca:
                #print cluster.energy
                pcaw = cluster.pca.explained_variance_
                pcaw /= np.sum(cluster.pca.explained_variance_)
                pca_widths.append(pcaw)
            ## print hits of cluster
            ax.scatter(cluster.hits[:, 1], cluster.hits[:, 0], cluster.hits[:,2], s = cluster.energies*100)

            #if pcaw1 > 0.5:
        #    pca_widths.append(pcaw1)

        plt.title('Event %i' %i_ev)
        plt.show()
        '''
        part_clust_data.append((particles,clusters))


    analyzer(part_clust_data, figtitle = 'TC')
    '''
    pca_widths = np.array(pca_widths)
    ax.hist(1-pca_widths[:,0], np.linspace(0,0.1,50))
    ax.hist(pca_widths[:,1], np.linspace(0,0.1,50))
    ax.hist(pca_widths[:,2], np.linspace(0,0.1,50))

    fig.tight_layout()

    plt.title('TC')

    print i_ev, n_parts
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
