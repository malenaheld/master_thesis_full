import sys
from pathlib import Path

# Get the parent directory of the script
parent_dir = Path(__file__).resolve().parent.parent

# Add the parent directory to sys.path
sys.path.append(str(parent_dir))

import numpy as np
import pickle
import healpy as hp
#from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
import parameters as pars

path = pars.path_SOM

## functions
def rescale_temps(temp, mask):
    ''' rescale the templates to the range between 0 and 1 in order to have the same sensitivity to all maps '''
    return (temp - np.min(temp[mask==1]))/ (np.max(temp[mask==1]) - np.min(temp[mask==1]))

def CreateClusters(n_clusters, data, som, mask, npix, h, w, N_comp):
    ''' given the number of clusters to create and the som, this function creates n_clust clusters using kmeans clustering or agglomerative clustering '''
    som_clust = np.array(som.get_weights().copy().reshape((h*w, N_comp)))
    kmeans = KMeans(n_clusters=n_clusters).fit(som_clust)
    #kmeans = AgglomerativeClustering(n_clusters=n_clusters).fit(som_clust)
    labels = kmeans.labels_

    labelsr = labels.reshape((w,h))

    label_map = []
    j = 0
    for i in range(npix):
        if mask[i] != 0:
            x, y = som.winner(data[i])
            lab_ind = labelsr[x,y]
            label_map.append(lab_ind)
            j +=1
        else:
            label_map.append(np.nan)
    label_map = np.array(label_map)
    return labels, label_map

def RecoMap(map, labels, label_map, npix):
    ''' given the clusters of the SOM, this function reconstructs, e.g. the galaxy number density map by assigning each pixel the mean number density of the cluster it is assigned to '''
    map_reco = np.ones(npix)

    ngalSOM = np.zeros_like(labels, dtype=float)
    for label in np.unique(labels):
        map_inds = np.where(label_map==label)
        ngals_clust = map[map_inds]

        map_reco[map_inds] = np.nanmean(ngals_clust)
        ngalSOM[labels==label] = np.nanmean(ngals_clust)
    return map_reco, ngalSOM


## set basic parameters
nside = pars.nside
npix = hp.nside2npix(nside)


## set parameters of the SOM
h = pars.SOM_side
w = pars.SOM_side

## list of clusters to create
n_cluster_list = pars.n_cluster_list

## read in mask
mask = hp.ud_grade(hp.read_map(f'../maps_for_salmo/{pars.mask}'), nside)

## read in templates
m_nstar = hp.ud_grade(hp.read_map('../templates/m_nstar.fits'), nside)
m_completeness = hp.ud_grade(hp.read_map('../templates/m_completeness.fits'), nside)
m_extinction = hp.ud_grade(hp.read_map('../templates/m_extinction.fits'), nside)
m_ngal = hp.ud_grade(hp.read_map('../templates/m_ngal.fits'), nside)
m_ngal = m_ngal / np.mean(m_ngal[mask==1])


## read in SOM
with open(f'{path}som_all.p', 'rb') as infile:
    som = pickle.load(infile)



## define templates
temps = np.array([rescale_temps(m_nstar, mask), rescale_temps(m_completeness, mask), rescale_temps(m_extinction, mask)])
temps_names = ['$n_{star}$', 'completeness', 'extinction']

## number of templates
N_comp = temps.shape[0]

## combine templates into one data vector
data = np.zeros(shape=(npix, N_comp))
for i, temp in enumerate(temps):
    data[:,i] = temp


## loop over all n_clusts
for n_cluster in n_cluster_list:
    print(f'for n_cluster={n_cluster}:')
    print('create clusters:')
    labels, label_map = CreateClusters(n_clusters=n_cluster, data=data, som=som, mask=mask, npix=npix, h=h, w=w, N_comp=N_comp)

    
    ngal_reco, ngalSOM = RecoMap(map=m_ngal, labels=labels, label_map=label_map, npix=npix)
    ngal_reco  = ngal_reco*mask

    ## write clustered depth map
    hp.write_map(f'{path}template_multiVD_nclust{n_cluster}.fits', ngal_reco, overwrite=True)

    ## write labels
    np.save(f'{path}labels_nclust{n_cluster}.npy', labels)

    ## write clustered SOM (ngal values)
    np.save(f'{path}ngalSOM_nclust{n_cluster}.npy', ngalSOM.reshape(w,h))
