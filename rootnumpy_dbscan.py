#!/usr/bin/env python
import sys
import ROOT, math
import numpy as np
#import cPickle as pickle

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN

from root_numpy import root2array, array2tree

from sklearn.decomposition import PCA
from wpca import WPCA, EMPCA

#ROOT.gROOT.SetBatch(1)

max_events = 10
#max_hits
max_parts = 100

mip_calibs = {}

def read_mip_calib(fname = "rechit_calib.txt"):

    mip_calib = {} # GeV to MIP caliv, per layer, per wafer thickness

    fcalib = open(fname)
    calib_lines = fcalib.readlines()
    fcalib.close()

    max_lay = 52

    for i, line in enumerate(calib_lines):

        if i < 3 or i > 60: continue

        items = line.split()
        #print i, len(items)

        if len(items) == 17: # Si layers with 3 thicknesses
            layer = int(items[0])
            gev2mip_300 = float(items[6])
            gev2mip_200 = float(items[7])
            gev2mip_100 = float(items[8])

            mip_calib[layer] = (gev2mip_100, gev2mip_200, gev2mip_300)

        if len(items) == 9: # Sc layers
            layer = int(items[0])
            gev2mip = float(items[4])

            mip_calib[layer] = gev2mip

    return mip_calib

def get_mip_calib(hit):
    global mip_calibs

    # return MIP per GeV factor
    if hit.layer() < 41:
        return mip_calibs[hit.layer()][0]
        '''
        if hit.thickness() < 130: # 100um
            return mip_calibs[hit.layer()][0]
        elif hit.thickness() < 230: # 100um
            return mip_calibs[hit.layer()][1]
        elif hit.thickness() > 230: # 100um
            return mip_calibs[hit.layer()][2]
        '''
    else:
        return mip_calibs[hit.layer()]


def store_hits(fname = "../ntuples/hgcalNtuple_ele15_n100_testhelper.root"):

    #branches = ["genpart_gen","rechit_x"]
    branches = ["rechit_x", "rechit_y", "rechit_z", "rechit_energy","rechit_layer",'rechit_flags','rechit_eta']
    array = root2array(fname,
                       treename='ana/hgc',
                       branches = branches,
                       #selection = 'rechit_z > 0',
                       stop = max_events
    )

    #print array
    #print len(array[0]['rechit_x'])

    tot_nevents = 0
    tot_genpart = 0
    tot_rechit = 0

    hist_data = {}

    part_hit_data = []

    #canv = ROOT.TCanvas("hits","Hits",1000,600)
    #hitmap = ROOT.TH2F("hitmap_xy","hitmap",41,-200,200,41,-200,200)
    #hitmap = ROOT.TH1F("hitmap_layer","hitmap_layer",28,1,29)

    #for event in ntuple:
    #for X in array['rechit_z','rechit_x','rechit_y']:
    #for X in array[['rechit_x','rechit_y']]:

    x_arr = None

    i_ev = -1
    for event in array:
        i_ev += 1

        '''
        x_arr = event['rechit_x']
        y_arr = event['rechit_y']
        z_arr = event['rechit_z']
        '''
        # with selection
        sel_hit_indices = (event['rechit_energy'] > 0.01) & (event['rechit_z'] > 0.) & (event['rechit_layer'] < 29.) & (event['rechit_flags'] < 3)# & (event['rechit_eta'] < 2.2)
        #sel_hit_indices = (event['rechit_energy'] > -0.01) & (event['rechit_z'] > 0.) & (event['rechit_layer'] < 39.) & (event['rechit_flags'] < 3)
        #sel_hit_indices = (event['rechit_energy'] > 0.01) & (event['rechit_layer'] < 39.) & (event['rechit_flags'] < 3)
        x_arr = event['rechit_x'][sel_hit_indices]
        y_arr = event['rechit_y'][sel_hit_indices]
        z_arr = event['rechit_z'][sel_hit_indices]
        #z_arr = event['rechit_layer'][sel_hit_indices]

        sample_weights = event['rechit_energy'][sel_hit_indices]

        print "Found %i hits" % len(sample_weights)
        if len(sample_weights) < 20: continue

        ## rescale Z
        #z_arr -= 320
        #z_arr /= 10.

        '''
        if x_arr == None:
            x_arr = event['rechit_x'][sel_hit_indices]
            y_arr = event['rechit_y'][sel_hit_indices]
            z_arr = event['rechit_z'][sel_hit_indices]
        else:
            print len(x_arr)
            np.insert(x_arr, 0, event['rechit_x'][sel_hit_indices])
            np.insert(x_arr, 0, event['rechit_x'][sel_hit_indices])
            np.insert(x_arr, 0, event['rechit_x'][sel_hit_indices])
            print len(x_arr)
        '''

        X = np.column_stack((z_arr,x_arr,y_arr))
        print X.shape, X.ndim

        #for i, lay in enumerate(event['rechit_layer']): hitmap.Fill(lay)
        #for lay, ene in event[['rechit_layer','rechit_energy']]:
        #    hitmap.Fill(lay,ene)

        #print hitmap.GetEntries()
        print "start clusterig"
        db = DBSCAN(eps=3, min_samples=5).fit(X)
        #db = DBSCAN(eps=2, min_samples = 2, algorithm = "auto").fit(X, sample_weight = sample_weights)

        print "done clusterig"
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_

        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        print('Estimated number of clusters: %d' % n_clusters_)

        if n_clusters_ < 1: continue

        # Plot result
        #fig = plt.figure(figsize=(15, 8))
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(111, projection='3d')

        # Black removed and is used for noise instead.
        unique_labels = set(labels)
        colors = [plt.cm.Spectral(each)
                  for each in np.linspace(0, 1, len(unique_labels))]

        energies = []

        for k, col in zip(unique_labels, colors):
            if k == -1:
                # Black used for noise.
                col = [0, 0, 0, 1]
                continue

            class_member_mask = (labels == k)

            '''
            ### 2D
            ## cores
            xy = X[class_member_mask & core_samples_mask]
            plt.plot(xy[:, 1], xy[:, 2], 'o', markerfacecolor=tuple(col),
                     markeredgecolor='k', markersize=14)
            ## outliers
            xy = X[class_member_mask & ~core_samples_mask]
            plt.plot(xy[:, 1], xy[:, 2], 'o', markerfacecolor=tuple(col),
                     markeredgecolor='k', markersize=6)
            '''
            ## cores
            xyz = X[class_member_mask & core_samples_mask]
            w = sample_weights[class_member_mask & core_samples_mask]

            if len(xyz) > 10:
                energy = np.sum(w) #sample_weights[class_member_mask & core_samples_mask] )

                if energy > 1:

                    print xyz.shape
                    ## run PCA
                    pca = PCA(n_components=3)
                    pca.fit(xyz)

                    #print(pca)
                    #print(pca.components_)
                    #print(pca.mean_)
                    #print(pca.noise_variance_)
                    print("PCA no w", pca.explained_variance_ratio_)
                    #print(pca.explained_variance_)
                    #print(pca.singular_values_)

                    if pca.explained_variance_ratio_[0] < 0.7: continue

                    '''
                    plt.title("PCA:," + " ".join(str(val) for val in pca.explained_variance_ratio_))
                    ## plot axis line
                    barycentre = pca.mean_
                    axis = pca.components_[0]

                    axis_start = barycentre + axis * 1.
                    axis_end = barycentre - axis * 1.
                    ax.plot([axis_start[1],axis_end[1]],[axis_start[0],axis_end[0]],zs=[axis_start[2],axis_end[2]], c='b')
                    '''
                    ### other PCA (with weights)
                    pca = WPCA(n_components=3)
                    #pca = EMPCA(n_components=3)
                    weights = np.column_stack((w,w,w))
                    pca.fit(xyz, weights = weights)

                    print("PCA w/ weight", pca.explained_variance_ratio_)

                    barycentre = pca.mean_
                    axis = pca.components_[0]
                    axis_start = barycentre + axis * 1.
                    axis_end = barycentre - axis * 1.
                    ax.plot([axis_start[1],axis_end[1]],[axis_start[0],axis_end[0]],zs=[axis_start[2],axis_end[2]], c='g')

                    title =  'Event %i' % i_ev
                    title += "PCA:," + " ".join(str(val) for val in pca.explained_variance_ratio_)
                    plt.title(title)

                    '''
                    #pca = WPCA(n_components=3)
                    pca = EMPCA(n_components=3)
                    weights = np.column_stack((w,w,w))
                    pca.fit(xyz, weights = weights)

                    barycentre = pca.mean_
                    axis = pca.components_[0]
                    axis_start = barycentre + axis * 10.
                    axis_end = barycentre - axis * 10.
                    ax.plot([axis_start[1],axis_end[1]],[axis_start[0],axis_end[0]],zs=[axis_start[2],axis_end[2]], c='y')
                    '''

                    ## plot core
                    ax.scatter(xyz[:, 1], xyz[:, 0], xyz[:,2], c = tuple(col), s = w*100)
                    #ax.scatter(xyz[:, 1] - barycentre[1], xyz[:, 0] - barycentre[0], xyz[:,2] - barycentre[2], c = tuple(col), s = w*10)

            ## plot outliers
            xyz = X[class_member_mask & ~core_samples_mask]
            w = sample_weights[class_member_mask & ~core_samples_mask]
            ax.scatter(xyz[:, 1], xyz[:, 0], xyz[:,2], c = tuple(col), s = w*100)

        fig.tight_layout()
        plt.show()
        #exit(0)


    #hitmap.Draw()
    #canv.Draw()
    #canv.Update()

    #q = raw_input("q")

    #continue
    exit(0)

    if False:
        '''
        genParts = event.genParticles()
        #tot_genpart += len(genParts)

        good_genparts = []
        for part in genParts:

            if abs(part.eta()) < 1.6 : continue
            if abs(part.eta()) > 2.8 : continue

            if part.gen() < 1: continue
            if part.pt() < 2: continue
            if part.reachedEE() != 2: continue

            good_genparts.append(part)

        if len(good_genparts) < 1: continue
        tot_genpart += len(good_genparts)

        ##### CLEAN CLOSEBY GEN PARTICLES (cannot resolve in HGCAL)
        ##
        # TODO
        ##
        '''

        print "filling"

        ## filter flag 3 rechits (below 3sigma noise)
        good_hits = []

        '''
        for i_hit, hit in enumerate(event.recHits()):

            #if i_hit > 1000: continue
            if hit.flags() > 3: continue
            if hit.layer() > 28: continue

            if abs(hit.eta()) < 1.7: continue
            if abs(hit.eta()) > 2.1: continue

            ## skip low-energy hits
            #mip_ene = hit.energy() * get_mip_calib(hit)
            #if mip_ene < 2.: continue

            good_hits.append(hit)
        '''

        tot_rechit += len(good_hits)

        # split z+ and z- into two separate events
        for z_sign in [-1,+1]:

            particles = [part for part in good_genparts if part.eta() * z_sign > 0]
            if len(particles) == 0: continue

            hits = [hit for hit in good_hits if hit.z() * z_sign > 0] #[:100] ## take just 10 hits
            if len(hits) == 0: continue

            fill_event_data(part_hit_data, particles, hits)

    print "Finished event loop / event filling"

    #print np.array(part_hit_data)
    event_data = np.array(part_hit_data)

    foutname = fname.replace(".root","_hits")
    np.save(foutname,event_data)


    ## save to pickle

    print("Found %i gen particles and %i rechits in %i events" %(tot_genpart,tot_rechit, tot_nevents))

    #tfile.Close()

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
