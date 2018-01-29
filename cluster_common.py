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

from scipy.optimize import curve_fit

def gauss(x, *p):
    A, mu, sigma = p
    return A*np.exp(-(x-mu)**2/(2.*sigma**2))

def calc_eta(x,y,z):

    theta = math.atan(math.hypot(x,y)/z)
    eta = -math.log(abs(math.tan(theta/2)))

    return eta

def deltaPhi(phi1, phi2):
    dPhi = phi1 - phi2
    if dPhi > math.pi:
        dPhi -= 2.*math.pi
    elif dPhi < -math.pi:
        dPhi += 2.*math.pi
    return dPhi

def deltaR(eta1,phi1, eta2,phi2):

    return math.hypot(eta1-eta2,deltaPhi(phi1,phi2))

class Particle:

    def __init__(self, energy, eta, phi):
        self.eta = eta
        self.phi = phi
        self.energy = energy

    def __repr__(self):
        string ="Particle\tEnergy %f\t" %self.energy
        string += "Position eta/phi: %f, %f" %(self.eta,self.phi)
        return string

class HGCcluster:

    def __init__(self, hits, energies):
        self.hits = hits
        self.nhits = len(hits)
        self.energies = energies
        self.energy = sum(energies)
        # compute barycentre position
        # energy weighted position
        self.x = np.sum(hits[:,1]*energies)/self.energy
        self.y = np.sum(hits[:,2]*energies)/self.energy
        self.z = np.sum(hits[:,0]*energies)/self.energy

        self.eta = calc_eta(self.x,self.y,self.z)
        self.phi = math.atan2(self.y,self.x)

        # default empty PCA
        self.pca = None

    def __repr__(self):
        string ="Cluster\tEnergy %f\t" %self.energy
        string += "Position x/y/z: %f, %f, %f \t" %(self.x,self.y,self.z)
        string += "Position eta/phi: %f, %f" %(self.eta,self.phi)
        return string

    def computePCA(self, withWeights = True):

        if self.nhits < 10: return 0

        if withWeights:
            pca = WPCA(n_components=3)
            weights = np.column_stack((self.energies,self.energies,self.energies))
            pca.fit(self.hits, weights = weights)
        else:
            pca = PCA(n_components=3)
            pca.fit(self.hits)

        self.pca = pca

        return 1

def my_cluster(X, W = None):
    #print "start clusterig"
    db = DBSCAN(eps=3, min_samples=5).fit(X)
    #db = DBSCAN(eps=3, min_samples = 5, algorithm = "auto").fit(X, sample_weight = W)
    #print "done clusterig"

    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    #print('Estimated number of clusters: %d' % n_clusters_)

    if n_clusters_ < 1: return 0

    #### PLOT
    unique_labels = set(labels)
    # Black removed and is used for noise instead.
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]

    clusters = []

    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]
            #continue

        class_member_mask = (labels == k)

        ## cores
        xyz = X[class_member_mask & core_samples_mask]
        w = W[class_member_mask & core_samples_mask]

        if len(xyz) < 1: continue
        cluster = HGCcluster(xyz,w)
        # compute cluster properties
        cluster.computePCA(withWeights = True)

        clusters.append(cluster)

        '''
        ## add outliers
        xyz = X[class_member_mask & ~core_samples_mask]
        w = W[class_member_mask & ~core_samples_mask]

        cluster = HGCcluster(xyz,w)
        clusters.append(cluster)
        '''

        '''
        print cluster
        if cluster.computePCA(withWeights = True):
            print "With weight", cluster.pca.explained_variance_ratio_
        if cluster.computePCA(withWeights = False):
            print "No weight", cluster.pca.explained_variance_ratio_


        energy = sum(w)
        #print "Cluster energy %f" %energy

        if energy < 50: continue

        if ax != None:
            #ax.scatter(xyz[:, 1], xyz[:, 0], xyz[:,2], c = tuple(col), s = w*100)

            ## plot outliers
            #xyz = X[class_member_mask & ~core_samples_mask]
            #w = W[class_member_mask & ~core_samples_mask]
            #ax.scatter(xyz[:, 1], xyz[:, 0], xyz[:,2], c = tuple(col), s = w*100)

            if len(xyz) > 10:
                pca_width1 = calc_widths(xyz,w, ax)

                return pca_width1
        '''
    #return 0
    #return labels
    return clusters

def calc_widths(X, W, ax):

    '''
    pca = PCA(n_components = 3)
    pca.fit(X)
    '''

    pca = WPCA(n_components=3)
    #weights = np.column_stack((1/W,1/W,1/W))
    weights = np.column_stack((W,W,W))
    pca.fit(X, weights = weights)

    barycentre = pca.mean_
    axis = pca.components_[0]
    axis_start = barycentre + axis * 10.
    axis_end = barycentre - axis * 10.
    #ax.plot([axis_start[1],axis_end[1]],[axis_start[0],axis_end[0]],zs=[axis_start[2],axis_end[2]], c='g')

    #title =  'Event %i' % i_ev
    #title += "PCA:," + " ".join(str(val) for val in pca.explained_variance_ratio_)
    ##print pca.explained_variance_ratio_, pca.explained_variance_

    print pca.explained_variance_ratio_
    #return pca.explained_variance_ratio_[0]
    return pca.explained_variance_[1]


def analyzer(part_clust_data, figtitle = 'Figure'):
    print 80*"#"
    print("Analyzing data")

    fig = plt.figure(figsize=(10, 10))
    #ax = plt.subplot()
    fig.canvas.set_window_title(figtitle)

    dRs = []
    eneRatios = []
    pca_widths = []
    pca_variance = []

    ### PARTICLE CLUSTER MATCHING
    for particles, clusters in part_clust_data:
        for particle in particles:
            for cluster in clusters:
                if cluster.energy < 10: continue
                #print particle.energy, cluster.energy
                dR = deltaR(particle.eta,particle.phi, cluster.eta,cluster.phi)
                dRs.append(  dR )
                if dR < 0.1:
                    if cluster.energy  / particle.energy > 0.15:
                        eneRatios.append ( cluster.energy  / particle.energy )
                        if cluster.pca:
                            pcaw = cluster.pca.explained_variance_
                            pca_widths.append(pcaw)
                            pca_variance.append(cluster.pca.explained_variance_ratio_)

                        '''
                        print "Good cluster"
                        print cluster
                        print particle
                        '''

                    else:
                        print "Problem cluster"
                        print cluster
                        print particle


    '''
    ax = plt.subplot(311)
    plt.xlabel("dR")
    plt.hist(dRs,50)
    '''

    ax = plt.subplot(221)
    ax.set_title("mean: %f, stdev: %f" %(np.mean(eneRatios), np.std(eneRatios)) )
    plt.xlabel("Ecl/Egene")
    hist, bins, _  = plt.hist(eneRatios,50)
    bin_centres = (bins[:-1] + bins[1:])/2
    ## fit gaussian
    # initial guess
    p0 = [1., np.mean(eneRatios), np.std(eneRatios)]
    coeff, var_matrix = curve_fit(gauss, bin_centres , hist, p0=p0)

    hist_fit = gauss(bin_centres, *coeff)

    plt.plot(bin_centres, hist_fit, label='Fitted data')

    print coeff


    pca_variance = np.array(pca_variance)
    ax = plt.subplot(222)
    ax.set_title("PCA variance ratios")
    ax.hist(1-pca_variance[:,0], np.linspace(0,0.1,50), alpha = 0.8,
            label = '1 - PCA1, mean/std: %0.2f/%0.2f' %(np.mean(1-pca_variance[:,0]),np.std(1-pca_variance[:,0])))
    ax.hist(pca_variance[:,1], np.linspace(0,0.1,50),
            alpha = 0.8, label = 'PCA2, mean/std: %0.2f/%0.2f' %(np.mean(pca_variance[:,2]),np.std(pca_variance[:,1])))
    ax.hist(pca_variance[:,2], np.linspace(0,0.1,50), alpha = 0.8,
            label = 'PCA3, mean/std: %0.2f/%0.2f' %(np.mean(pca_variance[:,2]),np.std(pca_variance[:,2])))

    plt.legend()

    pca_widths = np.array(pca_widths)
    ax = plt.subplot(223)
    ax.set_title("PCA variance")
    ax.hist(pca_widths[:,0], 40, alpha = 0.8, label = 'PCA1, mean/std: %0.2f/%0.2f' %(np.mean(pca_widths[:,0]),np.std(pca_widths[:,0])))
    plt.legend()

    ax = plt.subplot(224)
    ax.set_title("PCA variance")
    ax.hist(pca_widths[:,1], 40, alpha = 0.8, label = 'PCA2, mean/std: %0.2f/%0.2f' %(np.mean(pca_widths[:,1]),np.std(pca_widths[:,1])))
    ax.hist(pca_widths[:,2], 40, alpha = 0.8, label = 'PCA3, mean/std: %0.2f/%0.2f' %(np.mean(pca_widths[:,2]),np.std(pca_widths[:,2])))
    plt.legend()


    fig.tight_layout()
    print("Plotting")

    plt.show()

    return fig

def analyzeEnergy(part_clust_data, title = ''):
    print 80*"#"
    print("Analyzing data")

    '''
    fig = plt.figure(figsize=(10, 10))
    #ax = plt.subplot()
    fig.canvas.set_window_title(figtitle)
    '''

    dRs = []
    eneRatios = []

    ### PARTICLE CLUSTER MATCHING
    for particles, clusters in part_clust_data:
        for particle in particles:
            for cluster in clusters:
                if cluster.energy < 10: continue
                #print particle.energy, cluster.energy
                dR = deltaR(particle.eta,particle.phi, cluster.eta,cluster.phi)
                dRs.append(  dR )
                if dR < 0.1:
                    if cluster.energy  / particle.energy > 0.85:
                        eneRatios.append ( cluster.energy  / particle.energy )

    '''
    ax = plt.subplot(221)
    ax.set_title("mean: %f, stdev: %f" %(np.mean(eneRatios), np.std(eneRatios)) )
    plt.xlabel("Ecl/Egene")
    '''
    plt.subplot(211)
    hist, bins, _  = plt.hist(eneRatios,np.linspace(0.8,1.1,20), alpha = 0.8, label = str(title), histtype = 'step')

    bin_centres = (bins[:-1] + bins[1:])/2
    ## fit gaussian
    # initial guess
    #p0 = [1., np.mean(eneRatios), np.std(eneRatios)]
    p0 = [1., 1, 0.02]
    coeff, var_matrix = curve_fit(gauss, bin_centres , hist, p0=p0)

    hist_fit = gauss(bin_centres, *coeff)

    plt.plot(bin_centres, hist_fit, label='Fitted of ' + str(title))
    '''
    plt.show()
    '''

    #print coeff
    sigma = abs(coeff[2])

    return sigma
