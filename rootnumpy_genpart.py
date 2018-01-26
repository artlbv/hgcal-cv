#!/usr/bin/env python
import sys
import ROOT, math
import numpy as np
#import cPickle as pickle

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import sklearn
from sklearn.cluster import DBSCAN

from root_numpy import root2array, array2tree

from sklearn.decomposition import PCA
from wpca import WPCA, EMPCA

#ROOT.gROOT.SetBatch(1)

max_events = 10
max_parts = 10

max_layers = 28

def store_hits(fname = "../ntuples/hgcalNtuple_ele15_n100_testhelper.root"):

    #branches = ["genpart_gen","genpart_reachedEE","genpart_pt","genpart_energy","genpart_pid", "genpart_eta", "genpart_phi","genpart_posx","genpart_posy","genpart_posz"]
    branches = ["genpart_gen","genpart_reachedEE","genpart_energy","genpart_eta", "genpart_pid","genpart_posx","genpart_posy","genpart_posz"]
    #branches += ["rechit_x", "rechit_y", "rechit_z", "rechit_energy","rechit_layer",'rechit_flags','rechit_eta']
    branches += ["rechit_x", "rechit_y", "rechit_z", "rechit_energy","rechit_layer",'rechit_flags','rechit_cluster2d','cluster2d_multicluster',]
    array = root2array(fname,
                       treename='ana/hgc',
                       branches = branches,
                       #selection = 'rechit_z > 0',
                       stop = max_events
    )

    i_ev = -1

    n_parts = 0

    for event in array:

        if n_parts > max_parts: break

        i_ev += 1

        selected_genparts = (event['genpart_gen'] > 0.)
        selected_genparts &= (event['genpart_reachedEE'] > 1)#
        selected_genparts &= (event['genpart_energy'] > 5)
        selected_genparts &= (event['genpart_eta'] > 0)
        #selected_genparts &= (event['genpart_eta'] > 2.)
        selected_genparts &= (abs(event['genpart_pid']) == 22)
        #selected_genparts &= (abs(event['genpart_eta']) > 2.3)

        x_arr = event['genpart_posx'][selected_genparts]
        y_arr = event['genpart_posy'][selected_genparts]
        z_arr = event['genpart_posz'][selected_genparts]

        if len(x_arr) < 1: continue
        print "Found %i gen particles" % len(x_arr)
        n_parts += len(x_arr)


        fig = plt.figure(figsize=(10, 8))
        ax = plt.subplot(projection='3d')

        ## Plot particle trajectory
        for i_part,xa in enumerate(x_arr):
            #print x_arr[i_part].shape
            if len(x_arr[i_part]) < 1: continue
            #layers = np.arange(1,40,1)
            layers = np.array(range(1,max_layers+1))
            #ax.scatter(x_arr[i_part],layers,y_arr[i_part])
            #ax.plot(x_arr[i_part][:max_layers],z_arr[i_part][:max_layers],y_arr[i_part][:max_layers], '--r')
            ax.plot(x_arr[i_part][:max_layers],layers,y_arr[i_part][:max_layers], '--r')

        if len(x_arr[i_part]) < 1: continue

        ### HITS
        sel_hit_indices = (event['rechit_energy'] > -0.01)
        sel_hit_indices &= (event['rechit_layer'] < max_layers)
        sel_hit_indices &= (event['rechit_flags'] < 3)
        sel_hit_indices &= (event['rechit_z'] > 0.)

        x_arr = event['rechit_x'][sel_hit_indices]
        y_arr = event['rechit_y'][sel_hit_indices]
        #z_arr = event['rechit_z'][sel_hit_indices]
        z_arr = event['rechit_layer'][sel_hit_indices]

        sample_weights = event['rechit_energy'][sel_hit_indices]

        ## rescale Z
        #z_arr -= 320
        #z_arr /= 10.


        if len(sample_weights) < 20: continue
        print("Found %i hits" % len(sample_weights))

        rh_cl2d = event['rechit_cluster2d'][sel_hit_indices]
        rh_mcl = event['cluster2d_multicluster'][rh_cl2d]

        n_multicl = set(rh_mcl)
        colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(n_multicl))]

        ## Plot long energy profile
        #n,_ ,_ = ax.hist(z_arr, np.arange(28), weights = sample_weights)

        ax.scatter(x_arr, z_arr, y_arr, c = rh_mcl, s = sample_weights*100)

        fig.tight_layout()
        plt.title('Event %i' %i_ev)
        #if not ax.empty:
        plt.show()

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
        print '# Input file is', fname
        main(fname)
    else:
        print("No input files given!")
        main()

    # load tree
    #load_tree()
