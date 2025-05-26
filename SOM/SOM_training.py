import sys
from pathlib import Path

# Get the parent directory of the script
parent_dir = Path(__file__).resolve().parent.parent

# Add the parent directory to sys.path
sys.path.append(str(parent_dir))

import healpy as hp
import numpy as np
from minisom import MiniSom
import pickle
import argparse
import parameters as pars

#parser = argparse.ArgumentParser()
#parser.add_argument('path')
#args = parser.parse_args()

path = pars.path_SOM

nside = 256
npix = hp.nside2npix(nside)

mask = hp.ud_grade(hp.read_map(f'../maps_for_salmo/{pars.mask}'), nside)

## read in templates
m_nstar = hp.ud_grade(hp.read_map('../templates/m_nstar.fits'), nside)
m_nstar = m_nstar /np.mean(m_nstar[mask==1])
m_completeness = hp.ud_grade(hp.read_map('../templates/m_completeness.fits'), nside)
m_completeness = m_completeness /np.mean(m_completeness[mask==1])
m_extinction = hp.ud_grade(hp.read_map('../templates/m_extinction.fits'), nside)
m_completeness = m_completeness /np.mean(m_completeness[mask==1])

## functions
def rescale_temps(temp, mask):
    ''' rescale the templates to the range between 0 and 1 in order to have the same sensitivity to all maps '''
    return (temp - np.min(temp[mask==1]))/ (np.max(temp[mask==1]) - np.min(temp[mask==1]))

def fast_norm(x):
    """Returns norm-2 of a
      1-D numpy array.
    """
    return np.sqrt(np.dot(x, x.T))

## define templates
temps = np.array([rescale_temps(m_nstar, mask), rescale_temps(m_completeness, mask), rescale_temps(m_extinction, mask)])
temps_names = ['$n_{star}$', 'completeness', 'extinction']

N_comp = temps.shape[0]

data = np.zeros(shape=(npix, N_comp))
for i, temp in enumerate(temps):
    data[:,i] = temp 




## shuffle trainings sample
train_sample = data[mask==1].copy()
np.random.shuffle(train_sample)

## set up  SOM
h = pars.SOM_side
w = pars.SOM_side



som = MiniSom(x=w, y=h, input_len=N_comp, sigma=pars.sigma, learning_rate=pars.learning_rate, sigma_decay_function='inverse_decay_to_one', decay_function='inverse_decay_to_zero')
som.random_weights_init(train_sample)

n_epochs = 1 #pars.n_epochs
qe_arr = np.zeros(n_epochs)

for i in range(n_epochs):
    ## train the SOM
    som.train(train_sample, 20, use_epochs=True, verbose=False)
    ## get quantization error
    d_arr = []
    for v in train_sample:
        ## index of winner
        x, y = som.winner(v)
        w = som.get_weights()[x,y]

        d = fast_norm(w - v)

        d_arr.append(d)
    qe = np.mean(np.array(d_arr))  
    print(f'epoch {i+1} ----> quantization error: {qe}')
    qe_arr[i] = qe
np.save(f'{path}qe_arr.npy', qe_arr)


## save SOM
with open(f'{path}som_all.p', 'wb') as outfile:
    som = pickle.dump(som, outfile)
