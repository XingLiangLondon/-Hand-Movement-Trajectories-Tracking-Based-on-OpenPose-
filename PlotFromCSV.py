import numpy as np
import time
import cv2
import matplotlib.pyplot as plt
from datetime import datetime
from mpl_toolkits.mplot3d import Axes3D
import csv
import pandas as pd

#Get current date 
DATE= datetime.now().strftime('%Y:%m:%d')

#Plot trajectories X-Y
def plot_trajectories(center,str, clr):
    xs = [x[0] for x in center]
    ys = [x[1] for x in center]
    
    #xlim 640 * ylim 360   #xlim 960* ylim 540
    #plt.xlim(0,960) #Setting X axis range   (0,960)
    #plt.ylim(0,540) #Setting Y axis range  (0,540)
    plt.xlim(0,640) #Setting X axis range (0,640)  
    plt.ylim(0,360) #Setting Y axis range (0,360)  
    plt.plot(xs, ys, color= clr)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(str + ' Hand Trajectory')
    plt.gca().invert_yaxis()  #Reverse Y-Axis in PyPlot (opencv choose the coordinate system of points/images from Top-Left corner)
    #plt.gca().invert_xaxis()  #Reverse X-Axis in PyPlot (Make trajectories like a Mirror View)
    plt.show()
    return None


#Plot 3D trjectories with Timeline in Z
def plot_trajectories_3d(center, str, clr):
    xs = [x[0] for x in center]
    ys = [x[1] for x in center]
    ts = [x[2] for x in center]  
    fig = plt.figure()
    ax = plt.axes(projection='3d')   
    #ax.set_xlim3d(0,960) #Setting X axis range (0,960)
    #ax.set_zlim3d(0,540) #Setting Z axis range (0,540)
    ax.set_xlim3d(0,640) #Setting X axis range (0,640)
    ax.set_zlim3d(0,360) #Setting Z axis range (0,360)
    ax.plot3D(xs, ts, ys, color= clr, marker ='o') 
    #ax.set_yticks =(0, -1, 100)
    ax.set_xlabel('X')
    ax.set_ylabel('Time (ms)')
    ax.set_zlabel('Y')
    ax.set_title(str + ' Hand Trajectory')
    plt.gca().invert_zaxis()  #Reverse Z-Axis in PyPlot (to revert y)
    #plt.gca().invert_xaxis()  #Reverse X-Axis in PyPlot (Make trajectories like a Mirror View)
    plt.show()
    return None

# Plot trajectores with time
def plot_trajectories_vstime(center,str):
    xs = [x[0] for x in center]
    ys = [x[1] for x in center]
    ts = [x[2] for x in center]
    plt.plot(ts, xs, color='b', marker ='o', label='$X-Trajectory$')
    plt.plot(ts, ys, color='y', marker ='^', label='$Y-Trajectory$')
    plt.xlabel('Time')
    plt.ylabel('X-Y')
    plt.title(str + ' hand trajectories - MCI')
    plt.gca().invert_yaxis()  #Reverse Y-Axis in PyPlot (y reverted for:opencv choose the coordinate system of points/images from Top-Left corner; x reverted for: mirror effect)                   
    plt.legend(loc='upper right')
    plt.show()
    return None

def plot_trajectory_diagrams_left(csv):
    plot_trajectories(csv, "Left", "red")
    plot_trajectories_vstime(csv,(DATE+" Left"))
    plot_trajectories_3d(csv,(DATE+" Left"),  "red")
    return None

def plot_trajectory_diagrams_right(csv):  
    plot_trajectories(csv, "Right", "green")
    plot_trajectories_vstime(csv, (DATE+" Right"))   
    plot_trajectories_3d(csv,(DATE+" Right"), "green")
    return None

###Read CSV Data ()  
#df_left = pd.read_csv('C:/Users/user/Documents/Dunhill Project/Project Data/Data Preprocessing segmented/3_1/parameters_left.csv')
#df_right = pd.read_csv('C:/Users/user/Documents/Dunhill Project/Project Data/Data Preprocessing segmented/3_1/parameters_right.csv')
#
#df_left = pd.read_csv('E:/Dunhill Medical Research Project/Dunhill Project Data/UCL_Tyron/Reduced Size 640x 360/MCI/segmented/0041b/0041b_1/parameters_left.csv')
#df_right = pd.read_csv('E:/Dunhill Medical Research Project/Dunhill Project Data/UCL_Tyron/Reduced Size 640x 360/MCI/segmented/0041b/0041b_1/parameters_right.csv')  
#print("Values of df_left.left:", df_left.values)
#plot_trajectory_diagrams_left(df_left.values)
#plot_trajectory_diagrams_right(df_right.values)
