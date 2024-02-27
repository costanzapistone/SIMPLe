"""
Authors: Giovanni Franzese 
Email: g.franzese@tudelft.nl
Cognitive Robotics, TU Delft
This code is part of TERI (TEaching Robots Interactively) project
"""
#%%
from SIMPLe_bimanual.dual_panda import DualPanda
from pynput.keyboard import Listener, Key
import time
import rospy
import numpy as np
import pickle
from scipy.io import loadmat
from SIMPLe_bimanual.processing_functions import logvar, butter_bandpass, cov, whitening, csp, apply_mix
import csv
import os

class Coordination(DualPanda):
    def __init__(self):
        super().__init__()

        # Define constants
        self.SUBJECT = 'c'
        self.MODEL = 'LR' # Choose the model to use among: 'LDA', 'DT', 'RF', 'LR', 'NB' or 'SVM' 
        self.MAT_FILE = f'/home/platonics/Documents/costanza_workspace/src/Robot-Control-by-EEG-with-ML/data/BCICIV_calib_ds1{self.SUBJECT}.mat'
        self.TRAINED_MODEL_FILE_PATH = f"/home/platonics/Documents/costanza_workspace/src/Robot-Control-by-EEG-with-ML/code/classification/Subject_{self.SUBJECT}_lab/Trained_Models/{self.MODEL}_model.pkl"
        self.W_MATRIX_FILE_PATH = f"/home/platonics/Documents/costanza_workspace/src/Robot-Control-by-EEG-with-ML/code/classification/Subject_{self.SUBJECT}_lab/Trained_Models/CSP_matrix_W.pkl"
        self.CSV_FOLDERPATH = f'/home/platonics/Documents/costanza_workspace/src/Robot-Control-by-EEG-with-ML/results'
        self.experiment_number = 10
        self.THRESHOLD = None  # Set the threshold if available, otherwise set to None

        # Initialize csv_writer as None
        self.csv_writer = None

        # Load the classifier from the file
        with open (self.TRAINED_MODEL_FILE_PATH, 'rb') as file:
            self.classifier = pickle.load(file)

        # Load the W matrix from the file
        with open (self.W_MATRIX_FILE_PATH, 'rb') as file:
            self.W = pickle.load(file)

        # Define the file name for the CSV file
        self.csv_filename = "experiment_results.csv"


        # Variables to record the time
        self.start_time = None  # Variable to store the start time
        self.end_time = None    # Variable to store the end time

        if self.end_time is not None and self.start_time is not None:
            self.execution_time = self.end_time - self.start_time
        else:
            self.execution_time = None  # Or any default value you prefer

        # Load the .mat file
        self.EEG_data = loadmat(self.MAT_FILE, struct_as_record = True)
        self.sfreq = self.EEG_data['nfo']['fs'][0][0][0][0]
        self.EEGdata   = self.EEG_data['cnt'].T 
        self.nchannels, self.nsamples = self.EEGdata.shape
        self.event_onsets  = self.EEG_data['mrk'][0][0][0] # Time points when events occurred
        self.event_codes   = self.EEG_data['mrk'][0][0][1] # It contains numerical or categorical labels associated with each event.
        self.labels = np.zeros((1, self.nsamples), int)
        self.labels[0, self.event_onsets] = self.event_codes
        self.cl_lab = [s[0] for s in self.EEG_data['nfo']['classes'][0][0][0]]
        self.cl1    = self.cl_lab[0]
        self.cl2    = self.cl_lab[1]

        # Segmentation      
        self.trials = {}
        win = np.arange(int(0.5 * self.sfreq), int(4.5 * self.sfreq))
        self.nsamples = len(win)
        for cl, code in zip(self.cl_lab, np.unique(self.event_codes)):
            cl_onsets = self.event_onsets[self.event_codes == code]
            self.trials[cl] = np.zeros((self.nchannels, self.nsamples, len(cl_onsets)))
            for i, onset in enumerate(cl_onsets):
                self.trials[cl][:,:,i] = self.EEGdata[:, win + onset]

        # Filter the trials
        self.trials_filt = {self.cl1: butter_bandpass(self.trials[self.cl1], 8, 30, self.sfreq, self.nsamples),
                self.cl2: butter_bandpass(self.trials[self.cl2], 8, 30, self.sfreq, self.nsamples)}

        self.train_percentage = 0.7
        # Calculate the number of trials for each class the above percentage boils down to
        self.ntrain_l = int(self.trials_filt[self.cl1].shape[2] * self.train_percentage)
        self.ntrain_r = int(self.trials_filt[self.cl2].shape[2] * self.train_percentage)
        self.ntest_l = self.trials_filt[self.cl1].shape[2] - self.ntrain_l
        self.ntest_r = self.trials_filt[self.cl2].shape[2] - self.ntrain_r

        # Splitting the frequency filtered signal into a train and test set
        self.train = {self.cl1: self.trials_filt[self.cl1][:,:,:self.ntrain_l],
                 self.cl2: self.trials_filt[self.cl2][:,:,:self.ntrain_r]}

        self.test = {self.cl1: self.trials_filt[self.cl1][:,:,self.ntrain_l:],
                     self.cl2: self.trials_filt[self.cl2][:,:,self.ntrain_r:]}

        # Apply the CSP algorithm
        self.test[self.cl1] = apply_mix(self.W, self.test[self.cl1], self.nchannels, self.nsamples)
        self.test[self.cl2] = apply_mix(self.W, self.test[self.cl2], self.nchannels, self.nsamples)

        # Select only the first and last components for classification
        self.comp = np.array([0,-1])
        self.test[self.cl1] = self.test[self.cl1][self.comp,:,:]
        self.test[self.cl2] = self.test[self.cl2][self.comp,:,:]

        # Calculate the log-var
        self.test[self.cl1] = logvar(self.test[self.cl1])
        self.test[self.cl2] = logvar(self.test[self.cl2])

        self.listener_arrow = Listener(on_press=self._on_press_arrow, interval=1/self.control_freq)

        self.listener_arrow.start()

        self.look_ahead=10 # this how many steps in the future the robot is going to move after one input on the keyboard

        # Inizialite dictionary with two keys (the type of key pressed) and their attributes (the counts and the prediction made)
        self.key_press_data = {
            'left': {'count': 0, 'good_pred': 0, 'bad_pred': 0,},
            'right': {'count': 0, 'good_pred': 0, 'bad_pred': 0}            
        }


    def _on_press_arrow(self, key):
        # Based on the key pressed, pick a random sample from the test set of that class     
        # This function runs on the background and checks if a keyboard key was pressed
        
        if key == Key.esc:
            self.end_time = time.time()  # Record the end time when Esc key is pressed
            self.execution_time = self.end_time - self.start_time
            print(f"Execution Time: {self.execution_time} seconds")

            return False  # Stop the listener

        if self.start_time is None:
            self.start_time = time.time()  # Record the start time when the first arrow key is pressed

        
        if key == Key.right:

            self.key_press_data['right']['count'] += 1            

            # self.right = True
            print('Key Pressed: Right')
            random_index_cl1 = np.random.choice(self.test[self.cl1].shape[1])
            sample_cl1 = self.test[self.cl1][:, random_index_cl1]
            y_pred_cl1 = self.classifier.predict(sample_cl1.reshape(1,-1))

            if y_pred_cl1 == 1:
                print('Predicted Movement: Right')
                # Move the right robot
                self.right = True
                self.index_right=int(self.index_right+self.look_ahead)
                self.key_press_data['right']['good_pred'] += 1
            else:
                print('Predicted Movement: Left')
                # Move the left robot
                self.right = True
                self.index_left=int(self.index_left+self.look_ahead)
                self.key_press_data['right']['bad_pred'] += 1

        if key == Key.left:

            self.key_press_data['left']['count'] += 1

            # self.right = True
            print('Key Pressed: Left')

            random_index_cl2 = np.random.choice(self.test[self.cl2].shape[1])
            sample_cl2 = self.test[self.cl2][:, random_index_cl2]
            y_pred_cl2 = self.classifier.predict(sample_cl2.reshape(1,-1))

            if y_pred_cl2 == 1:
                print('Predicted Movement: Right')
                # Move the right robot
                self.right = True
                self.index_right=int(self.index_right+self.look_ahead)
                self.key_press_data['left']['bad_pred'] += 1
            else:
                print('Predicted Movement: Left')
                # Move the left robot
                self.right = True
                self.index_left=int(self.index_left+self.look_ahead)
                self.key_press_data['left']['good_pred'] += 1


    def syncronize(self):
        
        self.index_right=int(0)
        self.index_left=int(0)

        ind_left=0
        ind_right=0
        self.end=False
        r = rospy.Rate(self.control_freq)

        print("Press Esc to stop controlling the robot")


        attractor_pos_right = [self.Panda_right.recorded_traj[0][self.index_right],  self.Panda_right.recorded_traj[1][self.index_right],  self.Panda_right.recorded_traj[2][self.index_right]]
        attractor_pos_left = [self.Panda_left.recorded_traj[0][self.index_left],  self.Panda_left.recorded_traj[1][self.index_left],  self.Panda_left.recorded_traj[2][self.index_left]]

        if np.linalg.norm(np.array(attractor_pos_right)-self.Panda_right.cart_pos) > 0.05 or np.linalg.norm(np.array(attractor_pos_left)-self.Panda_left.cart_pos) > 0.05:
            print("Robots are too far away from the starting position, send them to start first")
            self.end=True
            return

        while not self.end:
            ind_right=np.min([ind_right+np.clip(self.index_right-ind_right,0,1), self.Panda_right.recorded_traj.shape[1]-1])
            ind_left=np.min([ind_left+np.clip(self.index_left-ind_left,0,1), self.Panda_left.recorded_traj.shape[1]-1])
            attractor_pos_right = [self.Panda_right.recorded_traj[0][ind_right],  self.Panda_right.recorded_traj[1][ind_right],  self.Panda_right.recorded_traj[2][ind_right]]
            attractor_ori_right = [ self.Panda_right.recorded_ori[0][ind_right],  self.Panda_right.recorded_ori[1][ind_right],   self.Panda_right.recorded_ori[2][ind_right],  self.Panda_right.recorded_ori[3][ind_right]]

            attractor_pos_left = [self.Panda_left.recorded_traj[0][ind_left],  self.Panda_left.recorded_traj[1][ind_left],  self.Panda_left.recorded_traj[2][ind_left]]
            attractor_ori_left = [ self.Panda_left.recorded_ori[0][ind_left],  self.Panda_left.recorded_ori[1][ind_left],   self.Panda_left.recorded_ori[2][ind_left],  self.Panda_left.recorded_ori[3][ind_left]]

            self.Panda_right.set_attractor(attractor_pos_right, attractor_ori_right)
            self.Panda_right.move_gripper(self.Panda_right.recorded_gripper[0, ind_right])

            self.Panda_left.set_attractor(attractor_pos_left, attractor_ori_left)
            self.Panda_left.move_gripper(self.Panda_left.recorded_gripper[0, ind_left])
            
            r.sleep()

    def store_experiment_data(self):
        """
        Store experiment data into a CSV file.
        """
        # Define the data to be written into the CSV file
        data = [
            self.SUBJECT,
            self.experiment_number,
            self.MODEL,
            self.THRESHOLD,
            self.execution_time,
            self.key_press_data['left']['count'],
            self.key_press_data['left']['good_pred'],
            self.key_press_data['left']['bad_pred'],
            self.key_press_data['right']['count'],
            self.key_press_data['right']['good_pred'],
            self.key_press_data['right']['bad_pred']
        ]

        # Combine the directory path and file name
        self.csv_file_path = os.path.join(self.CSV_FOLDERPATH, self.csv_filename)

        # Check if the file exists
        if not os.path.isfile(self.csv_file_path):
            # If the file doesn't exist, write headers to it
            with open(self.csv_file_path, 'w', newline='') as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(['Subject','Experiment Number', 'Model', 'Threshold Value', 'Execution Time', 'Left Count', 'Left Good Pred - TN', 'Left Bad Pred - FP', 'Right Count', 'Right Good Pred - TP', 'Right Bad Pred - FN'])
        
        # Write data to the CSV file
        with open(self.csv_file_path, 'a', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(data)

        
#%%
if __name__ == '__main__':

    BiManualTeaching=Coordination()
    time.sleep(1)
    #%%
    BiManualTeaching.Panda_right.load()
    BiManualTeaching.Panda_left.load()  
    #%%
    BiManualTeaching.Panda_right.go_to_start()
    #%%
    BiManualTeaching.Panda_left.go_to_start()  
    #%%
    BiManualTeaching.Panda_left.home()
    #%%
    BiManualTeaching.Panda_right.home()
    #%%
    BiManualTeaching.Panda_left.home_gripper()

    #%%
    BiManualTeaching.Panda_right.home_gripper()
    #%%
    BiManualTeaching.Panda_left.Kinesthetic_Demonstration()

    # %%
    BiManualTeaching.Panda_left.home()
    #%%
    BiManualTeaching.Panda_right.Kinesthetic_Demonstration()
    #%%
    BiManualTeaching.Panda_right.home()
    # %%
    BiManualTeaching.Panda_left.save() 
    #%%
    BiManualTeaching.Panda_right.save()      

    #%%
    BiManualTeaching.syncronize()
    # %%
    BiManualTeaching.store_experiment_data()
# %%
