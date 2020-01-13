from pykinect2 import PyKinectV2
from pykinect2.PyKinectV2 import *
from pykinect2 import PyKinectRuntime

import numpy as np
from sklearn import neural_network
from sklearn.neural_network import *

import ctypes
import _ctypes
import pygame
import sys
import csv
import time


from tkinter import *
import tkinter.filedialog

import argparse
from pythonosc import osc_message_builder
from pythonosc import osc_bundle_builder
from pythonosc import udp_client

if sys.hexversion >= 0x03000000:
    import _thread as thread
else:
    import thread

# colors for drawing different bodies 
SKELETON_COLORS = [pygame.color.THECOLORS["red"], 
                  pygame.color.THECOLORS["blue"], 
                  pygame.color.THECOLORS["green"], 
                  pygame.color.THECOLORS["orange"], 
                  pygame.color.THECOLORS["purple"], 
                  pygame.color.THECOLORS["yellow"], 
                  pygame.color.THECOLORS["violet"]]

IP = "10.3.208.18"
PORT = 5005

JT_LEGS = [PyKinectV2.JointType_KneeLeft, #13
            PyKinectV2.JointType_AnkleLeft, #14
            PyKinectV2.JointType_KneeRight, #17
            PyKinectV2.JointType_AnkleRight] #18

JT_ARMS = [PyKinectV2.JointType_ElbowLeft, #5
            PyKinectV2.JointType_WristLeft, #6
            PyKinectV2.JointType_ElbowRight, #9
            PyKinectV2.JointType_WristRight] #10

JT_BODY = [PyKinectV2.JointType_Head, #3
            PyKinectV2.JointType_SpineShoulder, #20
            PyKinectV2.JointType_SpineBase,  #0
            PyKinectV2.JointType_ShoulderLeft,
            PyKinectV2.JointType_ShoulderRight]

JT_ALL = JT_LEGS + JT_ARMS + JT_BODY


class FileHandler:
    def __init__(self, f, fileName, jointTypes):
        self._f = f
        self._fileName = fileName
        self._jointTypes = jointTypes

    def open(self):
        self._f = open(self._fileName,'w+')

    def close(self):
        self._f.close()

    def writeHeader(self):
        for i, listitem in enumerate(self._jointTypes):
            self._f.write('%s,,' % listitem)
            if i is not (len(self._jointTypes) - 1):
                self._f.write(',')
        self._f.write('\n')

    def writeData(self, lst):
        for i, listitem in enumerate(lst):
            self._f.write('%s' % listitem)
            if i is not (len(lst) - 1):
                self._f.write(',')
        self._f.write('\n')
        

class NeuralNetwork:
    def __init__(self, train_data, train_target, jointTypes, addresses):
        self._reg = None
        self._train_data = train_data
        self._train_target = train_target
        self._jointTypes = jointTypes
        self._addresses = addresses

    def train(self):
        self._reg = neural_network.MLPRegressor (hidden_layer_sizes=(5,), #create neural network
                                       activation='relu',
                                       solver='lbfgs', #optimized for small datasets
                                       #learning_rate='adaptive',
                                       max_iter=1000,
                                       #learning_rate_init=0.01,
                                       alpha=0.01)
        
        self._reg = self._reg.fit(self._train_data, self._train_target) #train neural network
        print("model trained")
    
class Parameter:
    def __init__(self, value, address):
        self._value = value
        self._address = address

class BodyGameRuntime(object):
    
    def __init__(self):
        
        pygame.init()

        self._autoTraining = True

        # Used to manage how fast the screen updates
        self._clock = pygame.time.Clock()

        # Set the width and height of the screen [width, height]
        self._infoObject = pygame.display.Info()
        self._screen = pygame.display.set_mode((self._infoObject.current_w >> 1, self._infoObject.current_h >> 1), 
                                               pygame.HWSURFACE|pygame.DOUBLEBUF|pygame.RESIZABLE, 32)

        pygame.display.set_caption("Kinect for Windows v2 Body Game")

        # States
        self._done = False
        self._listening = False
        self._recording = False

        # Kinect runtime object, we want only color and body frames 
        self._kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color | PyKinectV2.FrameSourceTypes_Body)

        # back buffer surface for getting Kinect color frames, 32bit color, width and height equal to the Kinect color frame size
        self._frame_surface = pygame.Surface((self._kinect.color_frame_desc.Width, self._kinect.color_frame_desc.Height), 0, 32)

        # here we will store skeleton data 
        self._bodies = None

        #here we will store the regression model
        self._neuralNets = []

        self._fileHandler = FileHandler(None, 'test.csv', [])


    def draw_body_bone(self, joints, jointPoints, color, joint0, joint1):
        joint0State = joints[joint0].TrackingState;
        joint1State = joints[joint1].TrackingState;

        # both joints are not tracked
        if (joint0State == PyKinectV2.TrackingState_NotTracked) or (joint1State == PyKinectV2.TrackingState_NotTracked): 
            return

        # both joints are not *really* tracked
        if (joint0State == PyKinectV2.TrackingState_Inferred) and (joint1State == PyKinectV2.TrackingState_Inferred):
            return

        # ok, at least one is good 
        start = (jointPoints[joint0].x, jointPoints[joint0].y)
        end = (jointPoints[joint1].x, jointPoints[joint1].y)

        try:
            pygame.draw.line(self._frame_surface, color, start, end, 8)
        except: # need to catch it due to possible invalid positions (with inf)
            pass

    def draw_body(self, joints, jointPoints, color):
        # Torso
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_Head, PyKinectV2.JointType_Neck);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_Neck, PyKinectV2.JointType_SpineShoulder);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_SpineShoulder, PyKinectV2.JointType_SpineMid);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_SpineMid, PyKinectV2.JointType_SpineBase);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_SpineShoulder, PyKinectV2.JointType_ShoulderRight);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_SpineShoulder, PyKinectV2.JointType_ShoulderLeft);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_SpineBase, PyKinectV2.JointType_HipRight);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_SpineBase, PyKinectV2.JointType_HipLeft);
    
        # Right Arm    
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_ShoulderRight, PyKinectV2.JointType_ElbowRight);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_ElbowRight, PyKinectV2.JointType_WristRight);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_WristRight, PyKinectV2.JointType_HandRight);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_HandRight, PyKinectV2.JointType_HandTipRight);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_WristRight, PyKinectV2.JointType_ThumbRight);

        # Left Arm
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_ShoulderLeft, PyKinectV2.JointType_ElbowLeft);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_ElbowLeft, PyKinectV2.JointType_WristLeft);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_WristLeft, PyKinectV2.JointType_HandLeft);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_HandLeft, PyKinectV2.JointType_HandTipLeft);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_WristLeft, PyKinectV2.JointType_ThumbLeft);

        # Right Leg
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_HipRight, PyKinectV2.JointType_KneeRight);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_KneeRight, PyKinectV2.JointType_AnkleRight);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_AnkleRight, PyKinectV2.JointType_FootRight);

        # Left Leg
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_HipLeft, PyKinectV2.JointType_KneeLeft);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_KneeLeft, PyKinectV2.JointType_AnkleLeft);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_AnkleLeft, PyKinectV2.JointType_FootLeft);


    def draw_color_frame(self, frame, target_surface):
        target_surface.lock()
        address = self._kinect.surface_as_array(target_surface.get_buffer())
        ctypes.memmove(address, frame.ctypes.data, frame.size)
        del address
        target_surface.unlock()

    def get_skeleton_data(self, joints, jointTypes, timer):
        
        positionLst = []
        for jointType in jointTypes:
            relJoint = PyKinectV2.JointType_SpineMid #the joint we use to normalise our data with
            normPosX = joints[jointType].Position.x - joints[relJoint].Position.x;
            normPosY = joints[jointType].Position.y - joints[relJoint].Position.y;
            normPosZ = joints[jointType].Position.z - joints[relJoint].Position.z;

            positionLst.append(round(normPosX, 4))
            positionLst.append(round(normPosY, 4))
            positionLst.append(round(normPosZ, 4))
        
        return positionLst

    def fetch_skeleton_data(self, joints, timer):
        print ("Predicting outputs based on skeleton data...")

        outputs = []

        print("predicted outputs: ")
        for neuralNet in self._neuralNets:
            #Get Skeleton Data
            positionLst = self.get_skeleton_data(joints, neuralNet._jointTypes, timer)
            #Predict output based on new sample
            output = neuralNet._reg.predict([positionLst])

            #add each predicted output value to the outputs list
            for i, value in enumerate(output.flatten()):
                print(value)
                outputs.append(Parameter(value, neuralNet._addresses[i]))

        return outputs;
    
    def print_skeleton_data(self, joints, timer):
        print ("Writing skeleton data to file...")
        
        #Get Skeleton Data
        positionLst = self.get_skeleton_data(joints, self._fileHandler._jointTypes, timer)

        #Write to file
        self._fileHandler.writeData(positionLst)
        
        
    def prepare_training_data(self, fileNames, outputVectors):
        
        data = [[] for x in range(len(fileNames))] #create a data-array with a length equal to the amount of fileNames
        out = [[] for x in range(len(outputVectors))] #create an output-array with a length equal to the amount of outputVectors
        
        #Read out jointTypes from the first fileName
        #Optionally: read out all jointTypes and compare
        jointTypes = []
        
        for i, fileName in enumerate(fileNames):
            with open(fileName, "r") as f:
                for j, line in enumerate(f):
                    test = line[:-1]
                    lst = [str(k) for k in test.split(',')]
                    # If file header of feature vector, append to jointTypes
                    if j == 0:
                        if i==0: jointTypes = lst
                        continue
                    # If feature vector, append to data[i]
                    lst = [float(i) for i in lst]
                    data[i].append(lst)

        print("Jointtypes: " + str(jointTypes))
        
        if '' in jointTypes: jointTypes = list(filter(('').__ne__, jointTypes)) #remove empty strings from list
        jointTypes = [int(i) for i in jointTypes] #convert strings to integers
        
        print("Jointtypes: " + str(jointTypes))

        for i, outputVector in enumerate(outputVectors):
            out[i] = outputVector * len(data[i])

        #set dataset by flattening both lists
        train_data = [item for sublist in data for item in sublist]
        train_target = [item for sublist in out for item in sublist]

        print("Data prepared")

        return train_data, train_target, jointTypes

##        self.train_model(train_data, train_target)


    def send_msg_over_osc(self, msg, path):
        if __name__ == "__main__":
            parser = argparse.ArgumentParser()
            parser.add_argument("--ip", default=IP,
              help="The ip of the OSC server")
            parser.add_argument("--port", type=int, default=PORT,
              help="The port the OSC server is listening on")
            args = parser.parse_args()
            client = udp_client.SimpleUDPClient(args.ip, args.port)
            client.send_message(path, msg)
            print("message sent")
            time.sleep(0.035)

    def open_train_dialog(self):

        selectedFiles = []

        root = Tk()
        root.title("Welcome to LikeGeeks app")
        root.geometry('350x200')
        
        def createNNDialog():
            for i, file in enumerate(selectedFiles):
                lbl = Label(root, text=(str(i+1) + ": " + file))
                lbl.grid(column=0, row=0)
        
        def openFileDialog():
            print("opening file dialog")
            files = tkinter.filedialog.askopenfilenames(parent=dialog,title="Select training set",filetypes = (("CSV Datasets","*.csv"),("all files","*.*")))
            files = root.tk.splitlist(files)
            selectedFiles = list(files)
            
        def continueGame():
            print("continuing game dialog")
            root.destroy()
            return [], []
        
        lbl = Label(root, text=("Current Neural Networks: " + str(len(self._neuralNets))))
        lbl.grid(column=0, row=0)
        btn = Button(root, text="Add New", command=openFileDialog)
        btn.grid(column=0, row=1)
        btn2 = Button(root, text="Continue", command=continueGame)
        btn2.grid(column=1, row=1)
        root.mainloop()

        #spinbox

        
        return trainingFilesSet, outputVectors
        
            
    def run(self):
        global filename, positionBuffer, f
        
        timer = 0
        
        # -------- Main Program Loop -----------------------------------------------------------------------------------------------------------------------
        while not self._done:
            
            # -- Main event loop ###########################################################################################
            for event in pygame.event.get(): # User did something
                
                if event.type == pygame.QUIT: # If user clicked close
                    self._done = True # Flag that we are done so we exit this loop

                elif event.type == pygame.VIDEORESIZE: # window resized
                    self._screen = pygame.display.set_mode(event.dict['size'], 
                                               pygame.HWSURFACE|pygame.DOUBLEBUF|pygame.RESIZABLE, 32)
                    
                elif event.type == pygame.KEYUP:
                    
                    if event.key == pygame.K_SPACE:
                        self._listening = not self._listening
                        print("Listening = " + str(self._listening))
                        positionBuffer = [] #empties the positionbuffer before and after listening
                        
                    elif event.key == pygame.K_TAB:
                        self._recording = not self._recording
                        print("Recording = " + str(self._recording))
                        if self._recording:
                            self._fileHandler._fileName = input("Enter filename: ")
                            
                            jointTypes = []
                            userinput = input("Choose Jointtypes: ")
                            if "legs" in userinput:     jointTypes = JT_LEGS; print("you chose legs")
                            elif "arms" in userinput:   jointTypes = JT_ARMS; print("you chose arms")
                            elif "body" in userinput:   jointTypes = JT_BODY; print("you chose body")
                            elif "all" in userinput:    jointTypes = JT_ALL;  print("you chose all")
                            else:                       jointTypes = JT_ALL;  print("unknown input, you chose default=all")
                            self._fileHandler._jointTypes = jointTypes
                            
                            self._fileHandler.open()
                            self._fileHandler.writeHeader()
                        else:
                            self._fileHandler.close()
                            
            # -- /END Main event loop ######################################################################################

                    
            # --- Game logic should go here

            # -- Getting frames and drawing  ###############################################################################
            # --- Woohoo! We've got a color frame! Let's fill out back buffer surface with frame's data 
            if self._kinect.has_new_color_frame():
                frame = self._kinect.get_last_color_frame()
                self.draw_color_frame(frame, self._frame_surface)
                frame = None

            # --- Cool! We have a body frame, so can get skeletons
            if self._kinect.has_new_body_frame(): 
                self._bodies = self._kinect.get_last_body_frame()

            # --- draw skeletons to _frame_surface
            if self._bodies is not None:
                bodyJoints = []
                for i in range(0, self._kinect.max_body_count):
                    body = self._bodies.bodies[i]
                    if not body.is_tracked: 
                        continue 
                    
                    joints = body.joints
                    bodyJoints.append(joints)
                    
                    # convert joint coordinates to color space 
                    joint_points = self._kinect.body_joints_to_color_space(joints)
                    self.draw_body(joints, joint_points, SKELETON_COLORS[i])                    

                # - DanceMachine ########################################
                timer += self._clock.get_time()
                if self._listening and timer > 300 and len(bodyJoints) > 0:
                    outputs = self.fetch_skeleton_data(bodyJoints[0], timer)

                    #Send predicted outputs over OSC to Ableton
                    for out in outputs:
                        self.send_msg_over_osc(round(out._value, 4), out._address)

                    timer = 0

                        
                if self._recording and timer > 150 and len(bodyJoints) > 0:
                    self.print_skeleton_data(bodyJoints[0], timer)
                    timer = 0
                # - /END DanceMachine ###################################                   
            # -- /END Getting frames and drawing  ###############################################################################################
                    
            # --- copy back buffer surface pixels to the screen, resize it if needed and keep aspect ratio
            # --- (screen size may be different from Kinect's color frame size) 
            h_to_w = float(self._frame_surface.get_height()) / self._frame_surface.get_width()
            target_height = int(h_to_w * self._screen.get_width())
            surface_to_draw = pygame.transform.scale(self._frame_surface, (self._screen.get_width(), target_height));
            self._screen.blit(surface_to_draw, (0,0))
            surface_to_draw = None
            pygame.display.update()

            # --- Go ahead and update the screen with what we've drawn.
            pygame.display.flip()

            # --- Limit to 30 frames per second
            self._clock.tick(30)
            
        # ------/END Main Program Loop ---------------------------------------------------------------------------------------------------------------------
        # Close our Kinect sensor, close the window and quit.
        self._kinect.close()
        pygame.quit()


__main__ = "DanceMachine | Kinect v2 Body Game"

game = BodyGameRuntime()

if game._autoTraining:
    trainingFilesSet = [["arms-low.csv", "arms-high.csv"],
                        ["legs-low.csv", "legs-high.csv"],
                        ["body-low.csv", "body-high.csv"]]

    #Should be same size as trainingFilesSet
    outputVectors = [[[[0.20, 0.20]], [[0.80, 0.80]]],
                      [[[0.20, 0.20]], [[0.80, 0.80]]],
                      [[[0.20, 0.20]], [[0.80, 0.80]]] ]

    addresses = [["/cc/2", "/cc/7"],
                 ["/cc/3", "/cc/6"],
                 ["/cc/4", "/cc/5"]]

##else:
##    trainingFilesSet, outputVectors = game.open_train_dialog()

#Create a new Neural Network for each set of training files:
for i, trainingFiles in enumerate(trainingFilesSet):
    train_data, train_target, jointTypes = game.prepare_training_data(trainingFiles, outputVectors[i])
    game._neuralNets.append(NeuralNetwork(train_data, train_target, jointTypes, addresses[i]))

#Train all Neural Netwoks
for neuralNet in game._neuralNets:
    neuralNet.train()
    
game.run()



'''
JOINT TYPES ENUM

JointType_SpineBase = 0
JointType_SpineMid = 1
JointType_Neck = 2
JointType_Head = 3
JointType_ShoulderLeft = 4
JointType_ElbowLeft = 5
JointType_WristLeft = 6
JointType_HandLeft = 7
JointType_ShoulderRight = 8
JointType_ElbowRight = 9
JointType_WristRight = 10
JointType_HandRight = 11
JointType_HipLeft = 12
JointType_KneeLeft = 13
JointType_AnkleLeft = 14
JointType_FootLeft = 15
JointType_HipRight = 16
JointType_KneeRight = 17
JointType_AnkleRight = 18
JointType_FootRight = 19
JointType_SpineShoulder = 20
JointType_HandTipLeft = 21
JointType_ThumbLeft = 22
JointType_HandTipRight = 23
JointType_ThumbRight = 24
JointType_Count = 25
'''
