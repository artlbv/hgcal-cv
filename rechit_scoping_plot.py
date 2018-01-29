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

max_events = 10
max_parts = 1

max_layers = 28

def calc_ene_sigma(fname = "../ntuples/hgcalNtuple_ele15_n100_testhelper.root", n_remove_layer = 4, ax = None):

    #branches = ["genpart_gen","genpart_reachedEE","genpart_pt","genpart_energy","genpart_pid", "genpart_eta", "genpart_phi","genpart_posx","genpart_posy","genpart_posz"]
    branches = ["genpart_gen","genpart_reachedEE","genpart_energy","genpart_eta","genpart_phi", "genpart_pid","genpart_posx","genpart_posy","genpart_posz"]
    branches += ["rechit_x", "rechit_y", "rechit_z", "rechit_energy","rechit_layer",'rechit_flags','rechit_eta']
    array = root2array(fname,
                       treename='ana/hgc',
                       branches = branches,
                       #selection = 'rechit_z > 0',
                       stop = max_events
    )

    i_ev = -1

    n_parts = 0

    pca_widths = []
    #fig = plt.figure(figsize=(10, 8))
    #ax = plt.subplot()
    #ax = plt.subplot(projection='3d')

    part_clust_data = []

    print 80*"#"
    print("Reading data from tree")
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
        sel_hit_indices = (event['rechit_energy'] > -0.01)
        sel_hit_indices &= (event['rechit_layer'] < max_layers)
        sel_hit_indices &= (event['rechit_flags'] < 3)
        sel_hit_indices &= (event['rechit_z'] > 0.)
        sel_hit_indices &= (event['rechit_layer'] %n_remove_layer != 0 )

        x_arr = event['rechit_x'][sel_hit_indices]
        y_arr = event['rechit_y'][sel_hit_indices]
        #x_arr = event['rechit_eta'][sel_hit_indices]
        #y_arr = event['rechit_phi'][sel_hit_indices]
        z_arr = event['rechit_z'][sel_hit_indices]
        #z_arr = event['rechit_layer'][sel_hit_indices]

        sample_weights = event['rechit_energy'][sel_hit_indices]
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

        for cluster in clusters:
            if cluster.pca:
                #print cluster.energy
                pcaw = cluster.pca.explained_variance_
                pcaw /= np.sum(cluster.pca.explained_variance_)
                pca_widths.append(pcaw)
            ## print hits of cluster
            ax.scatter(cluster.hits[:, 1], cluster.hits[:, 0], cluster.hits[:,2], s = cluster.energies*100, label = str(n_remove_layer))

            return 1

            #if pcaw1 > 0.5:
        #    pca_widths.append(pcaw1)

        #plt.title('Event %i' %i_ev)
        '''
        part_clust_data.append((particles,clusters))
        '''
    #plt.show()
    #analyzer(part_clust_data, figtitle = 'Rechit')
    #sigma = analyzeEnergy(part_clust_data, int(28/ n_remove_layer))
    #return sigma
    return 0
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

    #fig = plt.figure(figsize=(6, 8))
    fig = plt.figure(figsize=(10, 8))
    ax = plt.subplot(projection='3d')

    sigmas = []
    #frac_removed_layers = [10000] + range(10,1,-1)
    #frac_removed_layers = [10000,4,3,2]
    #frac_removed_layers = [10000,14,7,4,3,2]
    #frac_removed_layers = [10000,2]
    frac_removed_layers = [2,10000]

    for i in frac_removed_layers:
        print i
        sigma = calc_ene_sigma(fname, i, ax)
        #sigma = 1/float(i)
        #print i, sigma
        #sigmas.append([i, sigma])
        sigmas.append(sigma)

    '''
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

    '''
    plt.legend()
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
