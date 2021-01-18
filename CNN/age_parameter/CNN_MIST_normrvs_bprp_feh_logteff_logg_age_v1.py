import tensorflow.keras.backend as K
K.clear_session()

import numpy as np
import h5py
import random
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, InputLayer, Flatten, Reshape, Concatenate, concatenate
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import HDF5Matrix
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc


#Loading the data

#RVSFlux
pathrvs = '/Users/aishasultan/work/synple-gaia/run/CNN/age_parameter/RVS/spectra/normspectra_feh_pos00+pos.25+neg.25.npy'
norm_rvsflux = np.load (pathrvs)
print('shape of norm rvsflux:', np.shape(norm_rvsflux))

for i in range(np.shape(norm_rvsflux)[0]):
    plt.plot(np.arange(1134), norm_rvsflux[i,:])

#plt.show()

# BPFlux
pathbp = '/Users/aishasultan/work/synple-gaia/run/CNN/age_parameter/BP/spectra/spectra_feh_pos00+pos.25+neg.25.h5'
readbp = h5py.File(pathbp, 'r')
bpflux = readbp.get('bpflux')
bpflux = np.array(bpflux)
print('shape of bpflux:', np.shape(bpflux))

norm_bpflux = []

for ii in range(np.shape(bpflux)[0]):
    max_flux = np.max(bpflux[ii])
    normflux = bpflux[ii] / max_flux
    # plt.plot(np.arange(33), normflux)
    # print('normflux:',normflux)
    norm_bpflux.append(normflux)

norm_bpflux = np.array(norm_bpflux)

# print('shape of norm bpflux:', np.shape(norm_bpflux))
# print('shape of norm bpflux:', print(norm_bpflux[0:10]))

# RPFlux
pathrp = '/Users/aishasultan/work/synple-gaia/run/CNN/age_parameter/RP/spectra/spectra_feh_pos00+pos.25+neg.25.h5'
readrp = h5py.File(pathrp, 'r')
print(readrp.keys())
rpflux = readrp.get('rpflux')
rpflux = np.array(rpflux)
print('shape of rpflux:', np.shape(rpflux))

norm_rpflux = []
for ii in range(np.shape(rpflux)[0]):
    max_flux = np.max(rpflux[ii])
    normflux = rpflux[ii] / max_flux
    # plt.plot(np.arange(40), normflux)
    # print('normflux:',normflux)
    norm_rpflux.append(normflux)

norm_rpflux = np.array(norm_rpflux)

# print('shape of norm_rpflux:', np.shape(norm_rpflux))
# print('output of norm_flux:', print(norm_rpflux[0:10]))

#Importing the MIST parameter data
mist0 = '/Users/aishasultan/work/MIST/feh_pos0.00_100randstr_EEP0.h5'
mist1 = '/Users/aishasultan/work/MIST/feh_pos.25_100randstr_EEP0.h5'
mist2 = '/Users/aishasultan/work/MIST/feh_neg.25_100randstr_EEP0.h5'


readmist0 = h5py.File(mist0, 'r')
readmist1 = h5py.File(mist1, 'r')
readmist2 = h5py.File(mist2, 'r')


#reading MIST0 file
print(readmist0.keys())
mist_logteff0 = readmist0.get('logteff_sel')
teff_sel0 = np.array(mist_logteff0)
mist_logg0 = readmist0.get ('logg_sel')
logg_sel0 = np.array(mist_logg0)
mist_feh0 = readmist0.get ('feh_sel')
feh_sel0 = np.array(mist_feh0)
mist_age0 = readmist0.get ('age_sel')
age_sel0 = np.array(mist_age0)


#reading MIST1 file
#print(readmist1.keys())
mist_logteff1 = readmist1.get('logteff_sel')
teff_sel1 = np.array(mist_logteff1)
mist_logg1 = readmist1.get ('logg_sel')
logg_sel1 = np.array(mist_logg1)
mist_feh1 = readmist1.get ('feh_sel')
feh_sel1 = np.array(mist_feh1)
mist_age1 = readmist1.get ('age_sel')
age_sel1 = np.array(mist_age1)

#reading MIST2 file
#print(readmist1.keys())
mist_logteff2 = readmist2.get('logteff_sel')
teff_sel2 = np.array(mist_logteff2)
mist_logg2 = readmist2.get ('logg_sel')
logg_sel2 = np.array(mist_logg2)
mist_feh2 = readmist2.get ('feh_sel')
feh_sel2 = np.array(mist_feh2)
mist_age2 = readmist2.get ('age_sel')
age_sel2 = np.array(mist_age2)


#MIST all parameters
logteff = np.hstack((teff_sel0, teff_sel1, teff_sel2))
logg = np.hstack ((logg_sel0, logg_sel1, logg_sel2))
feh = np.hstack ((feh_sel0, feh_sel1, feh_sel2))
age = np.hstack ((age_sel0, age_sel1, age_sel2)) #it is in Giga, Mega years

#print( 'maximum age:',age.max())
#print('Fe/H:',feh)
#print('age:',age)
#print('logteff',logteff)

#features
num_tot = len(logteff)
print('total number of input stars =', num_tot)
plim = 0.8
ran_frac = np.random.uniform(0,1,num_tot)
#print('ranfrac=' , ran_frac)




#Input flux data of RVS, BP, RP
x_RVStrain = norm_rvsflux[ran_frac < plim, :]
print('number of RVS training data=', len(x_RVStrain[:,0]))
x_RVScv = norm_rvsflux[ran_frac >= plim, :] #test set we left for the application

x_BPtrain = norm_bpflux[ran_frac < plim, :]
print('number of BP training data=', len(x_BPtrain[:,0]))
x_BPcv = norm_bpflux[ran_frac >= plim, :]

x_RPtrain =norm_rpflux[ran_frac < plim, :]
print('number of RP training data=', len(x_RPtrain[:,0]))
x_RPcv = norm_rpflux[ran_frac >= plim, :]


y_logteff_train = logteff[ran_frac< plim]
y_logteff_cv = logteff[ran_frac >= plim] #test set

y_logg_train = logg[ran_frac< plim]
y_logg_cv = logg[ran_frac >= plim] #test set

y_feh_train = feh[ran_frac< plim]
y_feh_cv = feh[ran_frac >= plim] #test set

y_age_train = age[ran_frac< plim]
y_age_cv = age[ran_frac >= plim] #test set

print('The size of CVS for the input flux data')
print('RVS cvs=', np.shape(x_RVScv))
print('BP cvs=', np.shape(x_BPcv))
print('RP cvs=', np.shape(x_RPcv))
print('the size of output label=', np.shape(y_logteff_train) )


#Normalisation Function


def normalize(labels):
    max_labels = (labels).max()
    min_labels = (labels).min()
    norm_labels = ((labels) - min_labels) / (max_labels - min_labels)
    return (norm_labels, max_labels, min_labels)

#output label which is the logteff and log g
y_logteff_train = normalize(y_logteff_train)
y_logteff_cv = normalize(y_logteff_cv)

y_logg_train = normalize(y_logg_train)
y_logg_cv = normalize(y_logg_cv)

y_feh_train = normalize(y_feh_train)
y_feh_cv = normalize(y_feh_cv) #test set

y_age_train = normalize(y_age_train)
y_age_cv = normalize(y_age_cv) #test set

#CNN parameters

# activation function used following every layer except for the output layers
activation = 'relu'
# activation = 'sigmoid'

# model weight initializer
initializer = 'he_normal'

# number of filters used in the convolutional layers
num_filters = [8, 32]


# length of the filters in the convolutional layers

filter_length = 8

# length of the maxpooling window
pool_length = 4

# number of nodes in each of the hidden fully connected layers
num_hidden = [256, 128]
# num_hidden = [24,12]

# number of spectra fed into model at once during training
batch_size = 64

# maximum number of interations for model training
max_epochs = 100

# initial learning rate for optimization algorithm
lr = 0.000003  # handled by Adam

# exponential decay rate for the 1st moment estimates for optimization algorithm
beta_1 = 0.9

# exponential decay rate for the 2nd moment estimates for optimization algorithm
beta_2 = 0.999

# a small constant for numerical stability for optimization algorithm
optimizer_epsilon = 1e-08

num_RVSfluxes = len(x_RVStrain[0,:])
print('number of RVS fluxes=', num_RVSfluxes)

num_BPfluxes = len(x_BPtrain[0,:])
print('number of BP fluxes=', num_BPfluxes)

num_RPfluxes = len(x_RPtrain[0,:])
print('number of RP fluxes=', num_RPfluxes)


num_labels = 4
print('number of training labels=', num_labels)


# Input RVS spectra
print(' num_RVSfluxes=', num_RVSfluxes)
input_RVSspec = Input(shape=(num_RVSfluxes,), name='rvs_input_x' ) #removed name='starnet_input_x'

# Reshape spectra for RVS layers
cur_rvs = Reshape((num_RVSfluxes, 1))(input_RVSspec)

# CNN layers
cur_rvs = Conv1D(kernel_initializer=initializer, activation=activation,
                padding="same", filters=num_filters[0], kernel_size=filter_length)(cur_rvs) #first CNN layer
cur_rvs = Conv1D(kernel_initializer=initializer, activation=activation,
                padding="same", filters=num_filters[1], kernel_size=filter_length)(cur_rvs) #2nd CNN layer

# Max pooling layer
cur_rvs = MaxPooling1D(pool_size=pool_length)(cur_rvs)

# Flatten the current input for the fully-connected layers
cur_rvs = Flatten()(cur_rvs)


##############################################################################################


# Input BP spectra
print(' num_BPfluxes=', num_BPfluxes)
input_BPspec = Input(shape=(num_BPfluxes,), name='bp_input_x' ) #removed name='starnet_input_x'

# Reshape spectra for BP layers
cur_bp = Reshape((num_BPfluxes, 1))(input_BPspec)

# CNN layers
cur_bp = Conv1D(kernel_initializer=initializer, activation=activation,
                padding="same", filters=num_filters[0], kernel_size=filter_length)(cur_bp) #first CNN layer
cur_bp = Conv1D(kernel_initializer=initializer, activation=activation,
                padding="same", filters=num_filters[1], kernel_size=filter_length)(cur_bp) #2nd CNN layer

# Max pooling layer
cur_bp = MaxPooling1D(pool_size=pool_length)(cur_bp)


# Flatten the current input for the fully-connected layers
cur_bp = Flatten()(cur_bp)


################################################################################################

# Input RP spectra
print(' num_RPfluxes=', num_RPfluxes)
input_RPspec = Input(shape=(num_RPfluxes,), name='rp_input_x' ) #removed name='starnet_input_x'

# Reshape spectra for CNN layers
cur_rp = Reshape((num_RPfluxes, 1))(input_RPspec)

# CNN layers
cur_rp = Conv1D(kernel_initializer=initializer, activation=activation,
                padding="same", filters=num_filters[0], kernel_size=filter_length)(cur_rp) #first CNN layer
cur_rp = Conv1D(kernel_initializer=initializer, activation=activation,
                padding="same", filters=num_filters[1], kernel_size=filter_length)(cur_rp) #2nd CNN layer

# Max pooling layer
cur_rp = MaxPooling1D(pool_size=pool_length)(cur_rp)

# Flatten the current input for the fully-connected layers
cur_rp = Flatten()(cur_rp)

###############################################################################################

#concatenate RVS/BP/RP and then insert it to dense layer

cur_comb = Concatenate()([cur_rvs, cur_bp, cur_rp])

#cur_comb = cur_comb

# Fully-connected layers
cur_final = Dense(units=num_hidden[0], kernel_initializer=initializer,
               activation=activation)(cur_comb)
cur_final = Dense(units=num_hidden[1], kernel_initializer=initializer,
               activation=activation)(cur_final)

# Output nodes
output_final = Dense(units=num_labels, activation="linear",
                    input_dim=num_hidden[1], name='output_y')(cur_final)


model = Model(inputs = [input_RVSspec, input_BPspec, input_RPspec], outputs=output_final)

#model = Model(inputs = [input_RPspec], outputs=output_final)

model.summary()
#optimizer = Adam(lr=lr, beta_1=beta_1, beta_2=beta_2, epsilon=optimizer_epsilon, decay=0.0)

# Default loss function parameters
early_stopping_min_delta = 0.0007
early_stopping_patience = 4
reduce_lr_factor = 0.5
reduce_lr_epsilon = 0.0000009
reduce_lr_patience = 2
reduce_lr_min = 0.00008

# loss function to minimize
loss_function = 'mean_squared_error'

# compute mean absolute deviation
metrics = ['mae', 'mse']
#metrics = ['mae']
optimizer = Adam(lr=0.0001 )

model.compile(optimizer=optimizer, loss=loss_function, metrics=metrics)


early_stopping = EarlyStopping(monitor='val_loss', min_delta=early_stopping_min_delta,
                                       patience=early_stopping_patience, verbose=2, mode='min')

reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, epsilon=reduce_lr_epsilon,
                                  patience=reduce_lr_patience, min_lr=reduce_lr_min, mode='min', verbose=2)


y_train_stack = np.column_stack((y_logteff_train[0], y_logg_train[0], y_feh_train[0], y_age_train[0]))
y_cv_stack = np.column_stack((y_logteff_cv[0], y_logg_cv[0], y_feh_cv[0], y_age_cv[0]))

print('shape of the cv logteff and logg=', np.shape(y_cv_stack))
print(' shape of y teff and logg=', np.shape(y_train_stack))


history = model.fit(x=[x_RVStrain, x_BPtrain, x_RPtrain], y=y_train_stack,
                    validation_data=([x_RVScv, x_BPcv, x_RPcv], y_cv_stack),
                    epochs=max_epochs, verbose=1, shuffle='batch')



hist = pd.DataFrame(history.history)
hist['epoch']= history.epoch
hist.tail()


def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    plt.figure(figsize=(13,9))
    #plt.figure()
    plt.xlabel('Epoch', fontsize= 18)
    plt.ylabel('Mean Abs Error', fontsize= 18)
    plt.plot(hist['epoch'], hist['mae'], label='Train Error')
    plt.plot(hist['epoch'], hist['val_mae'], label = 'Val Error')
    plt.ylim([0,1.2])
    plt.tick_params(labelsize=20)
    plt.legend()
    plt.figure(figsize=(13,9))
    #plt.figure()
    plt.xlabel('Epoch', fontsize = 20)
    plt.ylabel('Mean Square Error', fontsize= 20)
    plt.plot(hist['epoch'], hist['mse'], label='Train Error')
    plt.plot(hist['epoch'], hist['val_mse'], label = 'Val Error')
    plt.ylim([0,10])
    plt.legend()
    plt.show()


plot_history(history)


#Unnormalizing the labels (teff)

print(np.shape(y_train_stack[:,0]))
print(np.shape(y_train_stack[:,1]))
print(np.shape(y_train_stack[:,2]))
print(np.shape(y_train_stack[:,3]))



def denormalize (labels, max_labels, min_labels):
    denorm_labels = ((labels * (max_labels - min_labels)) + min_labels)
    return (denorm_labels)


plt.figure(figsize=(13,9))
test_predictions = model.predict([x_RVStrain, x_BPtrain, x_RPtrain])
#test_predictions = model.predict([x_RPtrain])
print('shape of test_predictions:',np.shape(test_predictions))
print('print 10 elements of test_predictions:', test_predictions[0:10])
plt.scatter(denormalize(y_train_stack[:,0], y_logteff_train[1], y_logteff_train[2] ), denormalize(test_predictions[:,0], y_logteff_train[1], y_logteff_train[2]), s= 4.0, c= 'r', label='LogTeff' )
plt.scatter(denormalize(y_train_stack[:,1], y_logg_train[1], y_logg_train[2]), denormalize(test_predictions[:,1], y_logg_train[1], y_logg_train[2]), s= 4.0, c= 'b', label='Log(g)' )
plt.scatter(denormalize(y_train_stack[:,2], y_feh_train[1], y_feh_train[2]), denormalize(test_predictions[:,2], y_feh_train[1], y_feh_train[2]), s= 4.0, c= 'k', label='[Fe/H]' )
plt.scatter(denormalize(y_train_stack[:,3], y_age_train[1], y_age_train[2]), denormalize(test_predictions[:,3],  y_age_train[1], y_age_train[2]), s= 4.0, c= 'y', label='age' )
plt.xlabel(r"True", fontsize=25)
plt.ylabel(r"Prediction", fontsize=25)
plt.axis('equal')
plt.axis('square')
# increase the x,y tick label size
plt.tick_params(labelsize= 25)
#plt.xlim([0,plt.xlim()[1]])
#plt.ylim([0,plt.ylim()[1]])
#_ = plt.plot([-2, 3], [-2, 3], color='green')
plt.legend(fontsize=20)
plt.show()

print('fe/h denormalized:', denormalize(y_train_stack[:,2], y_feh_train[1], y_feh_train[2]))


