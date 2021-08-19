# Extened Kalman Filter

## To clone this repo 

` https://github.com/ruvate/extended.kalman.filter.git `


In this project, we look at how one-dimensional *Kalman Filter* works. The problem we would look over in this project can be thought of as a target tracking problem in one-dimensional space.

Here's how the files are laid out.

` EKF.pdf ` : : To understand the basics of how **Extended Kalman Filter** works, the theory behind it and the problem statement with relevant equations.

` ekf.py ` : : The code 

` true_odometry.csv ` | ` sensor_odom.csv `   : : These are the data files that will be used in running the code. They represents the **True Data** without any noise (representing a perfect world - which will be used to compare our estimated states) and **Sensor Data** with added noise (representing real-world equivalent - which will be used as measurement taken from a sensor). 



