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

max_events = 50
max_parts = 3

def store_hits(fname = "../ntuples/hgcalNtuple_ele15_n100_testhelper.root"):

    #branches = ["genpart_gen","genpart_reachedEE","genpart_pt","genpart_energy","genpart_pid", "genpart_eta", "genpart_phi","genpart_posx","genpart_posy","genpart_posz"]
    branches = ["genpart_gen","genpart_reachedEE","genpart_energy","genpart_eta", "genpart_phi","genpart_posx","genpart_posy","genpart_posz"]
    #branches = ["tc_x", "tc_y", "tc_z", "tc_energy","tc_layer"]
    branches += ["tc_x", "tc_y", "tc_z", "tc_energy","tc_layer",'tc_multicluster_id','tc_eta']

    array = root2array(fname,
                       treename='hgcalTriggerNtuplizer/HGCalTriggerNtuple',
                       branches = branches,
                       stop = max_events
    )

    i_ev = -1

    n_parts = 0

    for event in array:

        if n_parts > max_parts: break

        i_ev += 1

        selected_genparts = (event['genpart_gen'] > 0.) & (event['genpart_reachedEE'] > 1) & (event['genpart_eta'] > 0)

        x_arr = event['genpart_posx'][selected_genparts]
        y_arr = event['genpart_posy'][selected_genparts]
        z_arr = event['genpart_posz'][selected_genparts]

        if len(x_arr) < 1: continue
        print "Found %i gen particles" % len(x_arr)

        n_parts += len(x_arr)

        fig = plt.figure(figsize=(10, 5))
        ax = plt.subplot(projection='3d')

        ## Plot particle trajectory
        for i_part,xa in enumerate(x_arr):
            print len(x_arr[i_part])
            if len(x_arr[i_part]) < 1: continue
            #layers = np.arange(1,40,1)
            #ax.scatter(x_arr[i_part],layers,y_arr[i_part])
            ax.plot(x_arr[i_part][:28],z_arr[i_part][:28],y_arr[i_part][:28], '--or')

        if len(x_arr[i_part]) < 1: continue

        ### HITS
        sel_hit_indices = (event['tc_energy'] > 0.01) & (event['tc_z'] > 0.) & (event['tc_layer'] < 29.)
        x_arr = event['tc_x'][sel_hit_indices]
        y_arr = event['tc_y'][sel_hit_indices]
        z_arr = event['tc_z'][sel_hit_indices]

        sample_weights = event['tc_energy'][sel_hit_indices]

        ## rescale Z
        #z_arr -= 320
        #z_arr /= 10.


        if len(sample_weights) < 20: continue

        rh_mcl = event['tc_multicluster_id'][sel_hit_indices]

        n_multicl = set(rh_mcl)
        #colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(n_multicl))]

        ax.scatter(x_arr, z_arr, y_arr, c = rh_mcl, s = sample_weights*100)

        #fig.tight_layout()
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
