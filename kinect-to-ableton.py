from pykinect2 import PyKinectV2
from pykinect2.PyKinectV2 import *
from pykinect2 import PyKinectRuntime

import numpy as np
from sklearn import linear_model

import ctypes
import _ctypes
import pygame
import sys
import keyboard
import csv

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

listening = False
reg = None
positionBuffer = []

class BodyGameRuntime(object):
    
    def __init__(self):
        global reg
        
        pygame.init()

        trainingFiles = ["test-c.csv", "test-o.csv"];
        outputVectors = [[[0.30, 0.90, 0.10]], [[0.90, 0.20, 0.90]]]
        self.prepare_training_data(trainingFiles, outputVectors)

        # Used to manage how fast the screen updates
        self._clock = pygame.time.Clock()

        # Set the width and height of the screen [width, height]
        self._infoObject = pygame.display.Info()
        self._screen = pygame.display.set_mode((self._infoObject.current_w >> 1, self._infoObject.current_h >> 1), 
                                               pygame.HWSURFACE|pygame.DOUBLEBUF|pygame.RESIZABLE, 32)

        pygame.display.set_caption("Kinect for Windows v2 Body Game")

        # Loop until the user clicks the close button.
        self._done = False

        # Kinect runtime object, we want only color and body frames 
        self._kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color | PyKinectV2.FrameSourceTypes_Body)

        # back buffer surface for getting Kinect color frames, 32bit color, width and height equal to the Kinect color frame size
        self._frame_surface = pygame.Surface((self._kinect.color_frame_desc.Width, self._kinect.color_frame_desc.Height), 0, 32)

        # here we will store skeleton data 
        self._bodies = None


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

    def fetch_skeleton_data(self, joints, timer):

        print ("reading skeleton data...")
        
        global reg, positionBuffer
        
        '''
        print("x: " + str(round(joints[PyKinectV2.JointType_Head].Position.x, 2)) +
                          ", y: " + str(round(joints[PyKinectV2.JointType_Head].Position.y, 2)) +
                          ", z: " + str(round(joints[PyKinectV2.JointType_Head].Position.z, 2)) )
        '''

        #Selecting which joints to train our algorithm on
        jointTypes = [PyKinectV2.JointType_Head, #3
                      PyKinectV2.JointType_SpineShoulder, #20
                      PyKinectV2.JointType_ElbowLeft, #5
                      PyKinectV2.JointType_WristLeft, #6
                      PyKinectV2.JointType_ElbowRight, #9
                      PyKinectV2.JointType_WristRight, #10
                      PyKinectV2.JointType_SpineBase, #0
                      PyKinectV2.JointType_KneeLeft, #13
                      PyKinectV2.JointType_AnkleLeft, #14
                      PyKinectV2.JointType_KneeRight, #17
                      PyKinectV2.JointType_AnkleRight] #18
                      
        positionLst = []
        for jointType in jointTypes:
            relJoint = PyKinectV2.JointType_SpineMid #the joint we use to normalise our data with
            normPosX = joints[jointType].Position.x - joints[relJoint].Position.x;
            normPosY = joints[jointType].Position.y - joints[relJoint].Position.y;
            normPosZ = joints[jointType].Position.z - joints[relJoint].Position.z;

            #normPos = np.subtract(joints[jointType].Position, joints[PyKinectV2.JointType_SpineBase].Position)
            positionLst.append(round(normPosX, 4))
            positionLst.append(round(normPosY, 4))
            positionLst.append(round(normPosZ, 4))

        if not positionBuffer:
            positionBuffer = positionLst
            return

        velocityLst = []
        for i, pos in enumerate(positionLst):
            velocity = abs(float(1000*(pos - positionBuffer[i])/timer)) #velocity in /s
            velocityLst.append(velocity)

        output = reg.predict([positionLst]) #predicted output
        print("predicted output: " + str(output))

        for i, out in enumerate(output[0]):
            self.send_msg_over_osc(round(output[0][i], 4), "/cc/" + str(i+1))

##        self.send_msg_over_osc_alt(output[0], "/cc")

        positionBuffer = positionLst
        
    def train_model(self, train_data, train_target):
        global reg
        
        #machine learning
        reg = linear_model.Ridge (alpha = .5) #create regressioner
        reg = reg.fit(train_data, train_target) #train regressioner

        print("model trained")
    
    def prepare_training_data(self, fileNames, outputVectors):

        #LOAD DATA
        
        data = [[] for x in range(len(fileNames))]
        print(data)
        out = [[] for x in range(len(outputVectors))]
        print(out)
        
        for j, fileName in enumerate(fileNames):
            with open(fileName, "r") as f:
                for line in f:
                    test = line[:-1]
                    example = [float(i) for i in test.split(',')]
                    data[j].append(example)

        for i, outputVector in enumerate(outputVectors):
            out[i] = outputVector * len(data[i])

        #set dataset by flattening both lists
        train_data = [item for sublist in data for item in sublist]
        train_target = [item for sublist in out for item in sublist]

        self.train_model(train_data, train_target)

    def send_msg_over_osc(self, msg, path):
        if __name__ == "__main__":
            parser = argparse.ArgumentParser()
            parser.add_argument("--ip", default="192.168.10.6",
              help="The ip of the OSC server")
            parser.add_argument("--port", type=int, default=5005,
              help="The port the OSC server is listening on")
            args = parser.parse_args()

            client = udp_client.SimpleUDPClient(args.ip, args.port)

            client.send_message(path, msg)
            print("message sent")


    def send_msg_over_osc_alt(self, arguments, path):

        if __name__ == "__main__":
            parser = argparse.ArgumentParser()
            parser.add_argument("--ip", default="192.168.10.6",
              help="The ip of the OSC server")
            parser.add_argument("--port", type=int, default=5005,
              help="The port the OSC server is listening on")
            args = parser.parse_args()
            client = udp_client.SimpleUDPClient(args.ip, args.port)

            bundle = osc_bundle_builder.OscBundleBuilder(osc_bundle_builder.IMMEDIATELY)
            msg = osc_message_builder.OscMessageBuilder(address=path)
            
            for i in range(0, arguments.size):
                msg.add_arg(arguments[i])
                bundle.add_content(msg.build())

            sub_bundle = bundle.build()
            bundle.add_content(sub_bundle)
            bundle = bundle.build()

            client.send(bundle)
            print("message sent")
            
    def run(self):
        global listening, positionBuffer
        timer = 0
        
        # -------- Main Program Loop -----------
        while not self._done:
            # --- Main event loop
            for event in pygame.event.get(): # User did something
                if event.type == pygame.QUIT: # If user clicked close
                    self._done = True # Flag that we are done so we exit this loop

                elif event.type == pygame.VIDEORESIZE: # window resized
                    self._screen = pygame.display.set_mode(event.dict['size'], 
                                               pygame.HWSURFACE|pygame.DOUBLEBUF|pygame.RESIZABLE, 32)
                elif event.type == pygame.KEYUP:
                    if event.key == pygame.K_SPACE:
                        listening = not listening
                        positionBuffer = [] #empties the positionbuffer before and after listening
                    
            # --- Game logic should go here

            # --- Getting frames and drawing  
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
                timer += self._clock.get_time()
                for i in range(0, self._kinect.max_body_count):
                    body = self._bodies.bodies[i]
                    if not body.is_tracked: 
                        continue 
                    
                    joints = body.joints
                    if listening and timer > 300:
                        self.fetch_skeleton_data(joints, timer)
                        timer = 0
                    
                    # convert joint coordinates to color space 
                    joint_points = self._kinect.body_joints_to_color_space(joints)
                    self.draw_body(joints, joint_points, SKELETON_COLORS[i])

                    
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

        # Close our Kinect sensor, close the window and quit.
        self._kinect.close()
        pygame.quit()


__main__ = "Kinect v2 Body Game"
game = BodyGameRuntime();
game.run();


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
