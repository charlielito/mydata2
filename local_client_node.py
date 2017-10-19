#!/usr/bin/env python

import sys, os
import time, random, cv2, math, subprocess
import numpy as np
import rospy, base64, argparse, cv2

from pilot_cli.environments import CarEnv
from sensor_msgs.msg import CompressedImage
from local_client.msg import Control
from web_client.msg import Control as WebControl
from capture_3cameras.msg import Status
from std_msgs.msg import Bool
from std_msgs.msg import String
from mavros_msgs.msg import State


class WsLocalClient:
    def __init__(self, capture_topic = "capture/status", steer_topic = "local_client/control", queue=2):
        self._capture_topic = capture_topic
        self._steer_topic = steer_topic

        rospy.loginfo('Creating ws client')
        self._ws_client = CarEnv(mode="client",host="localhost", wait_next=False) #initiates a local WS client in port 4567
        rospy.loginfo('done!')

        self._capture_subcriber = rospy.Subscriber(self._capture_topic, Status, self.callback_fn, queue_size = queue)

        self._steer_publisher = rospy.Publisher(self._steer_topic, Control, queue_size=queue)

        self.recording = False
        self.left_camera_mean = -1
        self.center_camera_mean = -1
        self.right_camera_mean = -1

        self._mavros_subcriber = rospy.Subscriber("mavros/state", State, self.callback_fn2, queue_size = queue)
        self.armed = False

        self._nnet_subcriber = rospy.Subscriber("skynet/state", Bool, self.callback_fn3, queue_size = queue)
        self.nnet_active = False

    def callback_fn(self, data):
        self.recording = data.recording
        self.left_camera_mean = data.left_camera_mean
        self.center_camera_mean = data.center_camera_mean
        self.right_camera_mean = data.right_camera_mean

    def callback_fn2(self, data):
        self.armed = data.armed

    def callback_fn3(self, data):
        self.nnet_active = data.data

def main():
    # In ROS, nodes are uniquely named. If two nodes with the same name are launched, the previous one is kicked off.
    rospy.init_node('local_client_node', anonymous=True)

    # Publish that it is not ready yeet
    local_pub = rospy.Publisher("local_client/status", Bool, queue_size=2)
    time.sleep(1)
    local_pub.publish(False)


    rate = 20 #args.rate

    client = WsLocalClient()

    # Publish if it connected!
    local_pub.publish(True)

    def fn():
        local_pub.publish(False)
    rospy.on_shutdown(fn)

    r = rospy.Rate(rate)

    # Publisher to open lid and start recording stuff
    web_pub = rospy.Publisher("web_client/control", WebControl, queue_size=2)
    web_msg = WebControl()

    # Publisher to arm and disarm Pix
    arm_pub = rospy.Publisher("rover_teleop/arm_request", Bool, queue_size=2)
    start_msg = Bool()
    start_msg.data = True

    lid_opened = False
    recording = False

    enable_sended = False
    autonomous_sended = False
    local_sended = False

    fail_safe_sended = False
    return_sended = False

    msg = Control()
    pub_camera_config = rospy.Publisher('local_client/camera_config', String, queue_size=10)


    while not rospy.is_shutdown():

        response = client._ws_client.get_data()
        neural_net_active = client.nnet_active

        #print(response, client.recording)

        local_time = int(time.time() * 1000)
        # Logic to handle connection problems with local web interface
        if local_time - client._ws_client.last_message >= 2000:
            return_sended = False
            if not fail_safe_sended:
                print("no connection to local interface")
                local_pub.publish(False)
                fail_safe_sended = True
        else:
            fail_safe_sended = False
            if not return_sended:
                print("reconncetion to local interface")
                local_pub.publish(True)
                return_sended = True


        msg.speed = float(response.get("throttle", 0.0))
        msg.turn = float(response.get("steering_angle", 0.0))
        lid_res = response.get("lid_opened", "false") == "true"
        recording_res = response.get("recording", "false") == "true"

        enable_driving = response.get("enable_driving", "false") == "true"

        calibrate_cameras = response.get("calibrate_cameras", "false") == "true"

        left_camera_num = response.get("left_camera_num", "0")
        center_camera_num = response.get("center_camera_num", "1")
        right_camera_num = response.get("right_camera_num", "2")

        send_camera = response.get("send_camera", "false") == "true"

        autonomous = response.get("autonomous", "false") == "true"
        local = response.get("local", "false") == "true"

        if "left_camera_num" in response:
            pub_camera_config.publish(",".join([left_camera_num,center_camera_num,right_camera_num]))

        if calibrate_cameras:
            client._ws_client.emit(left_camera=client.left_camera_mean, center_camera=client.center_camera_mean, right_camera=client.right_camera_mean)

        # ARM request logic **********
        if enable_driving:
            if not enable_sended:
                print("enable!")
                arm_pub.publish(start_msg)
                enable_sended = True
        else:
            enable_sended = False

        # Auto mode logic **********
        if autonomous:
            if not autonomous_sended:
                print("set autonomous!")
                rospy.set_param('/rover_teleop/mode', 'auto')
                autonomous_sended = True
        else:
            autonomous_sended = False

        # Local control mode logic **********
        if local:
            if not local_sended:
                print("set manual!")
                rospy.set_param('/rover_teleop/mode', 'local')
                local_sended = True
        else:
            local_sended = False

        if lid_opened != lid_res:
            lid_opened = lid_res
            web_msg.door_toggle = True
            web_pub.publish(web_msg)
            web_msg.door_toggle = False
            print("lid opened")


        if recording_res != recording:
            print("start recording")
            web_msg.capture_toggle = True
            web_pub.publish(web_msg)
            web_msg.capture_toggle = False
            recording = recording_res


        client._steer_publisher.publish(msg)
        client._ws_client.emit(recording = client.recording, armed = client.armed, mode = rospy.get_param('/rover_teleop/mode'), auto_active = neural_net_active)

        r.sleep()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
    except:
        rospy.loginfo(sys.exc_info())
