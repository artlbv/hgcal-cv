#!/usr/bin/env python
import sys
import ROOT, math
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler

import timeit, math

max_events = 5
#max_events += 1

def cluster_hits(hits):

    hits = np.array(hits)

    print "Getting hit/E arrays ...",
    start_time = timeit.default_timer()
    new_hits = hits[:, [2,0,1]] ## store Z, X, Y
    sample_weights = hits[:, [3]]
    print(timeit.default_timer() - start_time)
    print("done")

    ## log-E weights
    #sample_weights = np.array(map(lambda w: math.log(w), sample_weights))
    ## change Z coordinates
    print new_hits
    #new_hits = np.apply_along_axis(lambda x: x/10., 0, new_hits)
    new_hits[:,0] /= 10.
    print new_hits

    #X = StandardScaler().fit_transform(new_hits)
    X = new_hits

    print "Clustering ...",
    start_time = timeit.default_timer()
    #db = DBSCAN(eps=3, min_samples=10).fit(X)
    #db = DBSCAN(eps=3, min_samples=10, algorithm = "auto").fit(X, sample_weight = sample_weights)
    db = DBSCAN(eps=2, min_samples = 2, algorithm = "auto").fit(X, sample_weight = sample_weights)
    print(timeit.default_timer() - start_time)
    print("done")

    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

    print('Estimated number of clusters: %d' % n_clusters_)

    # #############################################################################
    # Plot result
    fig = plt.figure()
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

        class_member_mask = (labels == k)

        ### 2D
        '''
        xy = X[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=14)

        xy = X[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=6)

        '''

        ### 3D
        xyz = X[class_member_mask & core_samples_mask]

        if len(xyz) > 0:
            ## compute energy of cluster
            energy = np.sum( sample_weights[class_member_mask & core_samples_mask] )
            #print "Cluster energy", energy
            energies.append(energy)

            '''
            ## compute barycentre
            for i in range(3):
                coords = xyz[:, i]
                c_times_ene = coords * sample_weights[class_member_mask & core_samples_mask] # coords X energies
                print "Z centre w/o and w/ energy weight:", np.mean(coords), np.sum(c_times_ene) / energy
            '''

        '''
        print np.mean(xyz[:, 0]), np.mean(xyz[:, 1]), np.mean(xyz[:, 2])
        #print np.mean(xyz, axis =2)
        #print xyz.shape, xyz.ndim
        '''

        ## PLOT
        #ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:,2], c = tuple(col), s = 14)

        ## plot outliers
        xyz = X[class_member_mask & ~core_samples_mask]
        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:,2], c = tuple(col), s = 6)

    plt.show()

    max_ene = max(energies) if len(energies) > 0  else -1
    return max_ene

def draw_hits(fname):

    #db = DBSCAN(eps=0.3, min_samples=3).fit(X)
    print("Loading file")
    events = np.load(fname)
    print("Events loaded")

    all_hits = []#np.array([])

    gr_part = ROOT.TGraph(); n_part = 0
    gr_part.SetMarkerStyle(20)
    gr_part.SetMarkerColor(ROOT.kOrange-5)

    gr_hits = ROOT.TGraph(); n_hits = 0
    gr_hits.SetMarkerStyle(6)
    gr_hits.SetMarkerColor(ROOT.kBlue-5)

    hitmap = ROOT.TH2F("hitmap_xy","hitmap",41,-20,20,41,-20,20)
    #hitmap = ROOT.TH2F("hitmap_etaphi","hitmap",41,-2,2,41,-2,2)

    h_ene_resp = ROOT.TH1F("ene_resp","energy response",50,0.8,1.1)

    for i_ev, event in enumerate(events[:max_events]):

        particles = event[0]
        hits = event[1]
        #print particle, hits

        if i_ev % 1 == 0: print "Event", i_ev

        #if len(particles) > 1: continue

        for particle in particles:
            gr_part.SetPoint(n_part, particle[0],particle[1])
            #gr_part.SetPoint(n_part, 0, 0)
            n_part += 1

            #print "Particle energy", particle[3]
        part_ene = particle[3]

        #if part_ene > 100: continue
        #print part_ene

        '''
        for hit in hits:
            gr_hits.SetPoint(n_hits, hit[0], hit[1])
            #gr_hits.SetPoint(n_hits, particle[0] - hit[0], particle[1] - hit[1])
            n_hits += 1

            #hitmap.Fill(particle[0] - hit[0], particle[1] - hit[1])
            # energy weighted
            hitmap.Fill(particle[0] - hit[0], particle[1] - hit[1], hit[3])
        '''
        # done with hit loop
        '''
        ## analyze each event
        all_hits = hits#[:10000]
        max_ene = cluster_hits(all_hits)
        h_ene_resp.Fill(max_ene / part_ene )
        '''

        all_hits += hits#[:10000]
    cluster_hits(all_hits)

    return 1

    if n_part > 0:
        print("Got %i particles, %i hits" %(n_part, n_hits) )

        canv = ROOT.TCanvas("hits","Hits",1000,600)
        canv.Divide(2,1)

        '''
        canv.cd(1)
        #gr_part.Draw("ap")
        gr_hits.Draw("ap")
        #gr_hits.Draw("p same")
        gr_part.Draw("p same")

        canv.cd(2)
        hitmap.Draw("colz")
        '''
        h_ene_resp.Draw()

        canv.Draw()
        canv.Update()

        q = raw_input()

if __name__ == "__main__":

    if '-b' in sys.argv: sys.argv.remove('-b')

    if len(sys.argv) > 1:
        fname = sys.argv[1]
        print '# Input file is', fname
        draw_hits(fname)
    else:
        print("No input files given!")
        draw_hits("HGCalAnalysis/test.npy")
