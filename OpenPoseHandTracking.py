import tf_pose 
from tf_pose.estimator import TfPoseEstimator
import numpy as np
from tf_pose.networks import get_graph_path, model_wh
import argparse
import logging
import sys
import time
from tf_pose import common
import cv2
import matplotlib.pyplot as plt
from datetime import datetime
from mpl_toolkits.mplot3d import Axes3D
import csv
import PlotFromCSV

# Define the root of a working folder
#file_root = "C:\\Users\\user\\Desktop\\CD-MAKE\\"
#file_root = "C:/Users/user/Documents/Dunhill Project/Project Data/Data Preprocessing/1/"
#file_root = "C:/Users/user/Documents/Dunhill Project/Project Data/Data Preprocessing segmented/CD-MAKE/3 MCI 30s/"
#file_root = "C:/Users/user/Documents/Dunhill Project/Project Data/Data Preprocessing segmented/test_seg_quality_high/"
#file_root= "E:/Dunhill Medical Research Project/Dunhill Project Data/Segmented videos/conversations/65+/BF30c/BF30c_01/"
#######file_root = "C:\\Users\\user\\Documents\\Dunhill Project\\Project Data\\Data Preprocessing segmented_4 minutes\\BF20c_05\\"

file_root = "E:\\Dunhill Medical Research Project\\Dunhill Project Data\\UCL_Tyron\\Reduced Size 640x 360\\MCI\\segmented\\0041b\\0041b_2\\"
#Define current working video name
video_name = "0041b_2.mpg"
 
# returns the elapsed milliseconds since the start of the program
def milliseconds():
   dt = datetime.now() - start_time_tracker
   ms = (dt.days * 24 * 60 * 60 + dt.seconds) * 1000 + dt.microseconds / 1000.0
   return ms

#Plot trajectories X-Y
def plot_trajectories(center,str, clr):
    xs = [x[0] for x in center]
    ys = [x[1] for x in center]
    
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

# Plot trajectores with time
def plot_trajectories_vstime(center,str):
    xs = [x[0] for x in center]
    ys = [x[1] for x in center]
    ts = [x[2] for x in center]
    plt.ylim(0,640) #Setting Y axis range (0,640)
    plt.plot(ts, xs, color='b', marker ='o', label='$X-Trajectory$')
    plt.plot(ts, ys, color='y', marker ='^', label='$Y-Trajectory$')
    plt.xlabel('Time')
    plt.ylabel('X-Y')
    plt.title(str + ' hand trajectories - MCI')
    plt.gca().invert_yaxis()  #Reverse Y-Axis in PyPlot (y reverted for:opencv choose the coordinate system of points/images from Top-Left corner; x reverted for: mirror effect)                   
    plt.legend(loc='upper right')
    plt.show()
    return None


#Plot 3D trjectories with Timeline in Z
def plot_trajectories_3d(center, str, clr):
    xs = [x[0] for x in center]
    ys = [x[1] for x in center]
    ts = [x[2] for x in center]  
    fig = plt.figure()
    ax = plt.axes(projection='3d')   
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


def plot_trajectory_diagrams():
    plot_trajectories(points_left, "Left", "red")
    plot_trajectories(points_right, "Right", "green")
    plot_trajectories_vstime(points_left,(DATE+" Left"))
    plot_trajectories_vstime(points_right, (DATE+" Right"))  
    plot_trajectories_3d(points_left,(DATE+" Left"),  "red")
    plot_trajectories_3d(points_right,(DATE+" Right"), "green")
    return None

def write__lefttrajectoryfeatures_tocsv(left):
    with open(file_root+"parameters_left.csv", "w") as csv_files:
        write= csv.writer(csv_files, quoting=csv.QUOTE_ALL)
        write.writerow(['X', 'Y', 'Time'])
        for row in left:
            write.writerow(row)

def write__righttrajectoryfeatures_tocsv(right):
    with open(file_root+ "parameters_right.csv", "w") as csv_files:
        write= csv.writer(csv_files, quoting=csv.QUOTE_ALL)
        write.writerow(['X', 'Y', 'Time'])
        for row in right:
            write.writerow(row)

#def write__lefttrajectoryfeatures_tocsv(left):
    #with open(file_root+"parameters_left.csv", "w") as csv_files:
        #write= csv.writer(csv_files, quoting=csv.QUOTE_ALL)
        #write.writerow(left)

#def write__righttrajectoryfeatures_tocsv(right):
    #with open(file_root+ "parameters_right.csv", "w") as csv_files:
        #write= csv.writer(csv_files, quoting=csv.QUOTE_ALL)
        #write.writerow(right)            
        
# Create empty points array for hand trajectories tracking 
points_left = []
points_right = []

#Initialise individual sampling face count 
count =0 

#Get current date 
DATE= datetime.now().strftime('%Y:%m:%d')
#Get tracker start time 
start_time_tracker = datetime.now()

# Try different trained models
e = TfPoseEstimator(get_graph_path(model_name='mobilenet_thin'), target_size=(432, 368))
#e = TfPoseEstimator(get_graph_path(model_name='mobilenet_v2_large'), target_size=(432, 368))
#e = TfPoseEstimator(get_graph_path(model_name='mobilenet_v2_small'), target_size=(432, 368))

### Video Input
#cam = cv2.VideoCapture("E:/Dunhill Medical Research Project/Dunhill Project Data/Converstation/BF1c.MOV")
cam =cv2.VideoCapture(file_root+ video_name)

while cam.isOpened(): 
    #This is for tracking the frame processing speed - PFS:
    start_time_per_frame=time.time()
    
    # Read webcam/video image
    ret, image = cam.read()
    
   
    # when there is a video input
    if ret == True:
        """
        # Get video/camera input details
        lengthVideo = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))
        print("Video Length",lengthVideo)
        widthVideo  = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        print("Video Width", widthVideo)
        heightVideo = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print("Video Height",heightVideo)

        # fps for video input with a stream header (need nb_frames field) 
        fpsVideo = int(cam.get(cv2.CAP_PROP_FPS)) 
        print("FPS", fpsVideo)
          
        """
        # Convert image from RBG/BGR to HSV 

        hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
        # Face Detection Using HAAR CASCADE 
        """
        hc_face = cv2.CascadeClassifier("C:/Users/user/Documents/Dunhill Project/Project Data/Data Preprocessing/haarcascade_frontalface_alt/haarcascade_frontalface_alt.xml")

        faces = hc_face.detectMultiScale(hsv_img)

        for (x,y,w,h) in faces:

            # If we draw a box on face to avoid face skin detection, then use the code below 

            cv2.rectangle(hsv_img, (x-10,y-30), (x+w+10, y+h+80), (255,255,255), -1)
            
            # Crop and Save face image to a file folder
            crop_img = image[y+2:y+w, x+2:x+h]
            count+=1
            cv2.imwrite(file_root +"face" +str(count)+".jpg",crop_img)
            cv2.imshow('Face Detection', crop_img)
        """   
        humans = e.inference(image, None, upsample_size=4.0)
        #print("Human:", humans)

        npimg = np.copy(image)
        image_h, image_w = npimg.shape[:2]
        centers = {}
        for human in humans:
            
            # draw point
            for i in range(common.CocoPart.Background.value):
                #print('i:', i)
                if i not in human.body_parts.keys():
                    continue               
                body_part = human.body_parts[i]
                center = (int(body_part.x * image_w + 0.5), int(body_part.y * image_h + 0.5), milliseconds())
                #print("center", center)
                #print('i:', i)
                centers[i] = center
                #print('centers[i]:', centers[i])
                
                if i == 4:
                    #print("i=4")
                    #print("Right center (green):", center)
             
                    cv2.circle(npimg, centers[4][:2], 3, color=(0,255,0), thickness=8, lineType=8, shift=0)
                    points_right.append(centers[4])
                    for l in range(1, len(points_right)):
                        try:
                            cv2.line(npimg, points_right[l - 1][:2], points_right[l][:2], (0, 255, 0), 2)
                        except:
                            pass 
                     
                elif i == 7:
                    #print("i=7")
                    #print("left center (red):", center)

                    cv2.circle(npimg, centers[7][:2], 3, color=(0,0,255), thickness=8, lineType=8, shift=0)
                    points_left.append(centers[7])
                    for l in range(1, len(points_left)):
                        try:
                            cv2.line(npimg, points_left[l - 1][:2], points_left[l][:2], (0, 0, 255), 2)
                        except:
                            pass
                else:
                    continue
                
            # draw line
            for pair_order, pair in enumerate(common.CocoPairsRender):
                if pair[0] not in human.body_parts.keys() or pair[1] not in human.body_parts.keys():
                    #print("pair_order", pair_order)
                    #print("pair", pair)
                    #print("pair[0]", pair[0])
                    #print("pair[1]", pair[1])
                    
                    #print("centers[pair[0]]", centers[pair[0]])
                    #print("centers[pair[1]]", centers[pair[1]])
                    continue

                npimg = cv2.line(npimg, centers[pair[0]][:2], centers[pair[1]][:2], common.CocoColors[pair_order], 3)
                cv2.line(npimg, centers[pair[0]][:2], centers[pair[1]][:2], common.CocoColors[pair_order], 3)
        cv2.putText(npimg, "FPS: %f" % (1.0 / (time.time() - start_time_per_frame)), (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.imshow('Skeleton image:', npimg) 
       
        if cv2.waitKey(1) == 13: #13 is the Enter Key
            plot_trajectory_diagrams()
            write__lefttrajectoryfeatures_tocsv(points_left)
            write__righttrajectoryfeatures_tocsv(points_right)
            points_left = []
            points_right = [] 
            break   
    #print("FPS: ", 1.0 / (time.time() - start_time))

                        

    else:
        if cv2.waitKey(1) == 13: #13 is the Enter Key
            plot_trajectory_diagrams()
            write__lefttrajectoryfeatures_tocsv(points_left)
            write__righttrajectoryfeatures_tocsv(points_right)
            points_left = []
            points_right = [] 
            break


cam.release()
cv2.destroyAllWindows() 
