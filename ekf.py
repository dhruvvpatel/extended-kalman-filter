# ************************************** #
#          Extended Kalman Filter        #
# ************************************** #
#                  by Matthew Postell    #
#                     Hirdayesh Shrestha #
#                     Dhruv Patel        #
# ************************************** #
#  Department of Mechanical Engineering  #
#       University of Cincinnati         #
# *************************************  #

import math
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt

# Importing csv files and reading the data
true_odo = genfromtxt('true_odometry.csv', delimiter=',')
sen_odo = genfromtxt('sensor_odom.csv',delimiter=',')
sen_pos_x, sen_pos_y = sen_odo[1:,1], sen_odo[1:,2]
sen_pos_theta = sen_odo[1:,3]

true_x, true_y, true_theta = true_odo[1:,1], true_odo[1:,2], true_odo[1:,3]
v, w = true_odo[1:,4], true_odo[1:,5]
time = sen_odo[1:,0]


# Observation that we are making - x and y position
z = np.c_[sen_pos_x, sen_pos_y]


# Defining Prediction Function
def Prediction(x_t, P_t, F_t, B_t, U_t, G_t, Q_t):
    x_t = F_t.dot(x_t) + B_t.dot(U_t) 
    P_t = (G_t.dot(P_t).dot(G_t.T)) + Q_t

    return x_t, P_t


# Defining Update Function
def Update(x_t, P_t, Z_t, R_t, H_t):
    S = np.linalg.inv( (H_t.dot(P_t).dot(H_t.T)) + R_t )
    K = P_t.dot(H_t.T).dot(S)

    x_t = x_t + K.dot( Z_t - H_t.dot(x_t) )
    P_t = P_t - K.dot(H_t).dot(P_t)

    return x_t, P_t



# Transition Matrix
F_t = np.array([  [1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 1]  ])


# Initial Covariance State
P_t = 0.5 * np.identity(3)

# # Process Covariance
Q_t = 1 * np.identity(3)

# # Measurement Covariance
R_t = 1 * np.identity(2)

# Measurement Matrix
H_t = np.array([    [1, 0, 0],
                    [0, 1, 0]   ])

# Initial State
x_t = np.array([[sen_pos_x[0]], [sen_pos_y[0]], [sen_pos_theta[0]] ])



kal_x, kal_y, kal_theta = [], [], []

dyn_x, dyn_y, dyn_z = [], [], []

check_dhruv = 0
threshold_x = ( 0.1 / 1 ) 
threshold_y = ( 0.1 / 1 )
threshold_z = ( 0.02 / 1 )

change_r = 1000



for i in range(2113):
    if i > 0:
        dt = time[i] - time[i-1]
    else:
        dt = 0

    # Jacobian Matrix - G
    G_t = np.array([    [1, 0, -v[i]*(math.sin(sen_pos_theta[i]))*dt],
                        [0, 1, v[i]*(math.cos(sen_pos_theta[i]))*dt],
                        [0, 0, 1]                                      ])
    # Input Transition Matrix - B
    B_t = np.array([    [dt * (math.cos(sen_pos_theta[i])), 0],
                        [dt * (math.sin(sen_pos_theta[i])), 0],
                        [0, dt]                                    ])
    # Input to the system - v and w ( velocity and turning rate )
    U_t = np.array([    [v[i]],
                        [w[i]]     ])

    # Condition : Variable Q_t and R_t
    

    # Prediction Step
    x_t, P_t = Prediction(x_t, P_t, F_t, B_t, U_t, G_t, Q_t)

    dyn_x.append(x_t[0])
    dyn_y.append(x_t[1])
    dyn_z.append(x_t[2])

    # *********************************
    # Dynamic Q and R : Threshold 
    # *********************************

    if i > 1 and i < 2113:

        if (abs(dyn_x[i] - kal_x[i-1]) > threshold_x) or \
           (abs(dyn_y[i] - kal_y[i-1]) > threshold_y) or \
           (abs(dyn_z[i] - kal_theta[i-1])) > threshold_z:


            # Believe in the sensor more :: Q > R
            R_t = (1/change_r) * np.identity(2)

            print("*********")
            check_dhruv = check_dhruv + 1

        else:
            # Believe in the process more :: R > Q

            # Q_t = 10 * np.identity(3)

            # # Measurement Covariance
            R_t = (change_r) * np.identity(2)   

    # Reshaping the measurement data
    Z_t = z[i].transpose()
    Z_t = Z_t.reshape(Z_t.shape[0], -1)

    # Update Step
    x_t, P_t = Update(x_t, P_t, Z_t, R_t, H_t)
    
    kal_x.append(x_t[0])
    kal_y.append(x_t[1])
    kal_theta.append(x_t[2])
        
    
    

print('\n')
print('*'*80)
print('\n'," Final Filter State Matrix : \n", x_t,'\n')

# For Plotting Purposes
kal_x = np.concatenate(kal_x).ravel()
kal_y = np.concatenate(kal_y).ravel()
kal_theta = np.concatenate(kal_theta).ravel()
# x = np.linspace(0, 71, 2113)

dyn_x = np.concatenate(dyn_x).ravel()
dyn_y = np.concatenate(dyn_y).ravel()
dyn_z = np.concatenate(dyn_z).ravel()



plt.figure(1)
plt.title('Estimated (Kalman) Pos X vs True Pos X', fontweight='bold') 
plt.plot(time,kal_x[:],'g--')
plt.plot(time,true_x, linewidth=3)


plt.figure(2)
plt.title('Estimated (Kalman) Pos Y vs True Pos Y', fontweight='bold') 
plt.plot(time,kal_y[:],'g--')
plt.plot(time,true_y, linewidth=3)

plt.figure(3)
plt.title('Estimated (Kalman) Theta vs True Theta', fontweight='bold') 
plt.plot(time,kal_theta[:],'g--')
plt.plot(time,true_theta, linewidth=2)


plt.figure(4)
plt.title('Robot Position : Kalman vs True', fontweight='bold')
plt.plot(kal_x,kal_y,'g--')
plt.plot(true_x,true_y, linewidth=3)

plt.figure(5)
# plt.plot(time[1500:2213], dyn_x[1500:2213], linewidth=2, label="After Process")
plt.plot(time[0:400], kal_x[0:400], linewidth=2, label="Kalman")
plt.plot(time[0:400], true_x[0:400], linewidth=2, label="Ground Truth")
# plt.plot(time[1500:2213],sen_pos_x[1500:2213], linewidth=2, label="Sensor Input")
plt.legend(loc='best')

# **********************************
# Curve - Fitting
# **********************************

# import statsmodels.api as sm

from sklearn import linear_model
from scipy.optimize import curve_fit

start = 0
limit =  600


X = time[start:limit].reshape(-1,1)
y = sen_pos_x[start:limit].reshape(-1,1)
True_XX = true_x[start:limit]

# ***************************

# model = sm.OLS(y, X).fit()
# predictions = model.predict(X)
# model.summary()

##########################

lm = linear_model.LinearRegression()
model = lm.fit(X,y)
predictions = lm.predict(X)
score = lm.score(X, y)
print("\n\n score : ", score)

##########################


plt.figure(6)
plt.plot(X, predictions, 'r-', label='Linear Fit')
plt.plot(X, y, label="Sensor Input")
plt.plot(X, True_XX, label="Ground Truth")
plt.legend(loc="best")


# [0:1000]

plt.show()


# Comparing True Data and Kalman (Estimated) Data for position to see how close they are Statistically
std_k_x = np.std(kal_x)
std_true_x = np.std(true_x)

print(' Standard Deviation Kalman : ', std_k_x)
print(' Standard Deviation True : ', std_true_x)

mean_k_x = np.mean(kal_x)
mean_true_x = np.mean(true_x)

print(' Mean Kalman : ', mean_k_x)
print(' Mean True : ', mean_true_x, '\n')

std_k_y = np.std(kal_y)
std_true_y = np.std(true_y)

print(' Standard Deviation Kalman : ', std_k_y)
print(' Standard Deviation True : ', std_true_y)

mean_k_y = np.mean(kal_y)
mean_true_y = np.mean(true_y)

print(' Mean Kalman : ', mean_k_y)
print(' Mean True : ', mean_true_y, '\n')

std_k_theta = np.std(kal_theta)
std_true_theta = np.std(true_theta)

print(' Standard Deviation Kalman : ', std_k_theta)
print(' Standard Deviation True : ', std_true_theta)

mean_k_theta = np.mean(kal_theta)
mean_true_theta = np.mean(true_theta)

print(' Mean Kalman : ', mean_k_theta)
print(' Mean True : ', mean_true_theta, '\n')

print(check_dhruv)