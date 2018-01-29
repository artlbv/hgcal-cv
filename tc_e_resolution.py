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

max_events = 1000
max_parts = 1000

max_layers = 28

def calc_ene_sigma(fname = "../ntuples/hgcalNtuple_ele15_n100_testhelper.root", n_remove_layer = 4):

    branches = ["genpart_gen","genpart_reachedEE","genpart_energy","genpart_eta","genpart_phi", "genpart_pid","genpart_posx","genpart_posy","genpart_posz"]
    #branches += ["rechit_x", "rechit_y", "rechit_z", "rechit_energy","rechit_layer",'rechit_flags','rechit_eta']
    branches += ["tc_x", "tc_y", "tc_z", "tc_energy","tc_layer"]
    print("Reading data from tree")
    array = root2array(fname,
                       treename='hgcalTriggerNtuplizer/HGCalTriggerNtuple',
                       branches = branches,
                       stop = max_events
    )

    i_ev = -1

    n_parts = 0

    pca_widths = []
    #fig = plt.figure(figsize=(10, 8))
    #ax = plt.subplot()

    part_clust_data = []

    print 80*"#"
    print("Processing events")
    for event in array:

        if n_parts > max_parts: break

        i_ev += 1

        selected_genparts = (event['genpart_gen'] > 0)
        selected_genparts &= (event['genpart_reachedEE'] > 1)#
        selected_genparts &= (event['genpart_energy'] > 5)
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
        ## Plot particle trajectory
        for i_part,xa in enumerate(x_arr):
            particles.append(Particle( ene_arr[i_part], eta_arr[i_part], phi_arr[i_part] ))
            ##print x_arr[i_part].shape
            if len(x_arr[i_part]) < 1: continue

            #print "Particle ene/eta/phi", ene_arr[i_part], eta_arr[i_part], phi_arr[i_part]

            max_lay = min(40,max_layers)
            #ax.plot(x_arr[i_part][:max_layers],z_arr[i_part][:max_layers],y_arr[i_part][:max_layers], '--b')
            layers = np.array(range(1,max_lay+1))
            #ax.plot(x_arr[i_part][:max_lay],layers,y_arr[i_part][:max_lay], '--b')

        if len(x_arr[i_part]) < 1: continue

        ### HITS
        sel_hit_indices = (event['tc_energy'] > -0.01)
        sel_hit_indices &= (event['tc_layer'] < max_layers+1)
        sel_hit_indices &= (event['tc_z'] > 0.)
        sel_hit_indices &= (event['tc_layer'] %n_remove_layer != 0 )
        #sel_hit_indices &= ((event['tc_layer']+1) %n_remove_layer != 0 )

        x_arr = event['tc_x'][sel_hit_indices]
        y_arr = event['tc_y'][sel_hit_indices]
        #x_arr = event['tc_eta'][sel_hit_indices]
        #y_arr = event['tc_phi'][sel_hit_indices]
        z_arr = event['tc_z'][sel_hit_indices]
        #z_arr = event['tc_layer'][sel_hit_indices]

        sample_weights = event['tc_energy'][sel_hit_indices]
        #sample_weights *= 1/(1-0.25)
        sample_weights *= 1/(1-1/float(n_remove_layer))

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

    #analyzer(part_clust_data, figtitle = 'Rechit')
    sigma = analyzeEnergy(part_clust_data, int(28/ n_remove_layer))

    return sigma
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

    fig = plt.figure(figsize=(6, 8))

    sigmas = []
    #frac_removed_layers = [10000] + range(10,1,-1)
    frac_removed_layers = [10000,14,7,4,3,2]
    #frac_removed_layers = [10000,8,4]#,3,2]
    #frac_removed_layers = [10000,2]
    #frac_removed_layers = [30,10,9,8,7,6,5]

    for i in frac_removed_layers:
        print i
        sigma = calc_ene_sigma(fname, i)
        #sigma = 1/float(i)
        #print i, sigma
        #sigmas.append([i, sigma])
        sigmas.append(sigma)

    plt.subplot(211)
    plt.legend()
    plt.xlabel('Ecluster/Eele')

    sigmas = np.array(sigmas)
    print sigmas
    sigmas /= sigmas[0]

    frac_removed_layers = np.array(frac_removed_layers)
    n_layers_removed = 28/frac_removed_layers
    print n_layers_removed
    #print np.array(sigmas)

    #ax = plt.subplot()
    plt.subplot(212)
    plt.xlabel('N of removed layers')
    plt.ylabel('Sigma/Sigma(0)')
    plt.scatter(n_layers_removed, sigmas)

    plt.savefig('sigma_plot.png')

    plt.show()

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
