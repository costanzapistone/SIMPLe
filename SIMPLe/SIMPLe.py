from ILoSA import ILoSA #you need to pip install ILoSA first 
import numpy as np
from ILoSA import InteractiveGP
from ILoSA.data_prep import slerp_sat
import pickle
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
import rospy
class SIMPLe(ILoSA):

    def __init__(self):
        super(SIMPLe, self).__init__()
        self.rec_freq = 20  # [Hz]
        self.control_freq=20 # [Hz]
        self.r_control=rospy.Rate(self.control_freq)
        self.r_rec=rospy.Rate(self.rec_freq)
    def Train_GPs(self):
        print("SIMPLe does not need to be trained")
        if len(self.nullspace_traj)>0 and len(self.nullspace_joints)>0:
            print("Training of Nullspace")
            kernel = C(constant_value = 0.1, constant_value_bounds=[0.0005, self.attractor_lim]) * RBF(length_scale=[0.1, 0.1, 0.1], length_scale_bounds=[0.025, 0.1]) + WhiteKernel(0.00025, [0.0001, 0.0005]) 
            self.NullSpaceControl=InteractiveGP(X=self.nullspace_traj, Y=self.nullspace_joints, y_lim=[-self.attractor_lim, self.attractor_lim], kernel=kernel, n_restarts_optimizer=20)
            self.NullSpaceControl.fit()
            with open('models/nullspace.pkl','wb') as nullspace:
                pickle.dump(self.NullSpaceControl,nullspace)
        else: 
            print('No Null Space Control Policy Learned')    



    def GGP(self, look_ahead=3):
        n_samples=self.training_traj.shape[0]
        look_ahead=3 # how many steps forward is the attractor for any element of the graph

        labda_position=0.05 #lengthscale of the position    
        lamda_time=0.05 #lenghtscale of the time
        lambda_index=self.rec_freq*lamda_time #convert the lenghtscale to work with indexes
        
        # we use an exponential kernel. The uncertainty can be estimated as simga= 1- exp(- (position_error)/lambda_error - (time_error)/lambda_time)
        #We consider uncertainty points that have a distance of at least 2 times the sum of the normalized errors with the lengthscales
        sigma_treshold= 1 - np.exp(-2) 
        #calcolation of correlation in space
        position_error= np.linalg.norm(self.training_traj - np.array(self.cart_pos), axis=1)/labda_position

        #calcolation of correlation in time
        index=np.min([self.mu_index+look_ahead, n_samples-1])

        index_error= np.abs(np.arange(n_samples)-index)/lambda_index
        index_error_clip= np.clip(index_error, 0, 1) # we saturate the time error, to avoid that points that are far away in time cannot be activated in case of perturbation of the robot

        # Calculate the product of the two correlation vectors
        k_start_time_position=np.exp(-position_error-index_error_clip)

        # Compute the uncertainty only as a function of the correlation in space and time
        sigma_position_time= 1- np.max(k_start_time_position)

        # Compute the scaling factor for the stiffness Eq 15
        if sigma_position_time > sigma_treshold: 
            beta= (1-sigma_position_time)/(1-sigma_treshold)
            self.mu_index = int(np.argmax(k_start_time_position))
        else:
            beta=1
            self.mu_index = int(self.mu_index+ 1.0*np.sign((int(np.argmax(k_start_time_position))- self.mu_index)))
     
        control_index=np.min([self.mu_index+look_ahead, n_samples-1])

        return  control_index, beta


    def initialize_mu_index(self):
        position_error= np.linalg.norm(self.training_traj - np.array(self.cart_pos), axis=1)
        self.mu_index = int(np.argmin(position_error))

    def control(self):
        self.initialize_mu_index()
        self.Interactive_Control()

    def step(self):
        
        i, beta = self.GGP()

        pos_goal  = self.training_traj[i,:]
        pos_goal=self.cart_pos+ np.clip([pos_goal[0]-self.cart_pos[0],pos_goal[1]-self.cart_pos[1],pos_goal[2]-self.cart_pos[2]],-0.05,0.05)
        quat_goal = self.training_ori[i,:]
        quat_goal=slerp_sat(self.cart_ori, quat_goal, 0.1)
        gripper_goal=self.training_gripper[i,0]
        
        self.set_attractor(pos_goal,quat_goal)
        self.move_gripper(gripper_goal) #TODO write a better logic for the gripper 
            
        K_lin_scaled =beta*self.K_mean
        K_ori_scaled =beta*self.K_ori
        pos_stiff = [K_lin_scaled,K_lin_scaled,K_lin_scaled]
        rot_stiff = [K_ori_scaled,K_ori_scaled,K_ori_scaled]

        self.set_stiffness(pos_stiff, rot_stiff, self.null_stiff)    