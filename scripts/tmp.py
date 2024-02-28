#%%
import numpy as np
import pickle
from scipy.io import loadmat
from SIMPLe_bimanual.processing_functions import logvar, butter_bandpass, cov, whitening, csp, apply_mix

# Define constants
SUBJECT = 'g'
MODEL = 'LR' # Choose the model to use among: 'LDA', 'DT', 'RF', 'LR', 'NB' or 'SVM' 
MAT_FILE = f'/home/platonics/Documents/costanza_workspace/src/Robot-Control-by-EEG-with-ML/data/BCICIV_calib_ds1{SUBJECT}.mat'
TRAINED_MODEL_FILE_PATH = f"/home/platonics/Documents/costanza_workspace/src/Robot-Control-by-EEG-with-ML/code/classification/Subject_{SUBJECT}_lab/Trained_Models/{MODEL}_model.pkl"
W_MATRIX_FILE_PATH = f"/home/platonics/Documents/costanza_workspace/src/Robot-Control-by-EEG-with-ML/code/classification/Subject_{SUBJECT}_lab/Trained_Models/CSP_matrix_W.pkl"
TRAIN_PERCENTAGE = 0.6

# Load the classifier from the file
with open (TRAINED_MODEL_FILE_PATH, 'rb') as file:
    classifier = pickle.load(file)

# Load the W matrix from the file
with open (W_MATRIX_FILE_PATH, 'rb') as file:
    W = pickle.load(file)

# Load the .mat file
EEG_data = loadmat(MAT_FILE, struct_as_record = True)
sfreq = EEG_data['nfo']['fs'][0][0][0][0]
EEGdata   = EEG_data['cnt'].T 
nchannels, nsamples = EEGdata.shape
event_onsets  = EEG_data['mrk'][0][0][0] # Time points when events occurred
event_codes   = EEG_data['mrk'][0][0][1] # It contains numerical or categorical labels associated with each event.
labels = np.zeros((1, nsamples), int)
labels[0, event_onsets] = event_codes
cl_lab = [s[0] for s in EEG_data['nfo']['classes'][0][0][0]]
cl1    = cl_lab[0]
cl2    = cl_lab[1]

#%% Segmentation
trials = {}
win = np.arange(int(0.5 * sfreq), int(4.5 * sfreq))
nsamples = len(win)
for cl, code in zip(cl_lab, np.unique(event_codes)):
    cl_onsets = event_onsets[event_codes == code]
    trials[cl] = np.zeros((nchannels, nsamples, len(cl_onsets)))
    for i, onset in enumerate(cl_onsets):
        trials[cl][:,:,i] = EEGdata[:, win + onset]

#%% Filter the trials
trials_filt = {cl1: butter_bandpass(trials[cl1], 8, 30, sfreq, nsamples),
                cl2: butter_bandpass(trials[cl2], 8, 30, sfreq, nsamples)}

train_percentage = TRAIN_PERCENTAGE
# Calculate the number of trials for each class the above percentage boils down to
ntrain_l = int(trials_filt[cl1].shape[2] * train_percentage)
ntrain_r = int(trials_filt[cl2].shape[2] * train_percentage)
ntest_l = trials_filt[cl1].shape[2] - ntrain_l
ntest_r = trials_filt[cl2].shape[2] - ntrain_r

# Splitting the frequency filtered signal into a train and test set
train = {cl1: trials_filt[cl1][:,:,:ntrain_l],
         cl2: trials_filt[cl2][:,:,:ntrain_r]}

test = {cl1: trials_filt[cl1][:,:,ntrain_l:],
        cl2: trials_filt[cl2][:,:,ntrain_r:]}

# Apply the CSP algorithm
train[cl1] = apply_mix(W, train[cl1], nchannels, nsamples)
train[cl2] = apply_mix(W, train[cl2], nchannels, nsamples)
test[cl1] = apply_mix(W, test[cl1], nchannels, nsamples)
test[cl2] = apply_mix(W, test[cl2], nchannels, nsamples)

# Select only the first and last components for classification
comp = np.array([0,-1])
train[cl1] = train[cl1][comp,:,:]
train[cl2] = train[cl2][comp,:,:]
test[cl1] = test[cl1][comp,:,:]
test[cl2] = test[cl2][comp,:,:]

# Calculate the log-var
train[cl1] = logvar(train[cl1])
train[cl2] = logvar(train[cl2])
test[cl1] = logvar(test[cl1])
test[cl2] = logvar(test[cl2])

#%% Get a random sample from the test set of the class1
random_index = np.random.choice(test[cl1].shape[1])
sample = test[cl1][:, random_index]
print(sample.shape)
print(sample.reshape(-1,1).shape)
#%% Get the prediction and the probability of the prediction
y_pred = classifier.predict(sample.reshape(1,-1))
print(y_pred)
if y_pred == 1:
    print('Predicted Movement: Right')
    # Move the right robot
else:
    print('Predicted Movement: Left')
    # Move the left robot

# %%

pred_proba = classifier.predict_proba(sample.reshape(1,-1))
print(pred_proba)
# %%

threshold = 0.2
if pred_proba[0][1] > (0.5 + threshold) | pred_proba[0][0] > (0.5 + threshold):
    print(pred_proba)