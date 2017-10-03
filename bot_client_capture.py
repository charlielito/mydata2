#!/usr/bin/env python
import rospy, time, sys, base64, std_srvs.srv, cv2
from pilot_cli.environments import CarEnv 
import numpy as np
from sensor_msgs.msg import CompressedImage 
from bot_server.msg import Control 
from web_client.msg import Control as WebControl 
from capture_3cameras.msg import Status 
from std_msgs.msg import Bool

class CarServer:
    def __init__(self, steer_topic = "/bot_server/control", camera_topic = "/camera/image/compressed", queue=2):
        self._steer_topic = steer_topic
        self._camera_topic = camera_topic
        rospy.loginfo('Creating server')
        self._ws_server = CarEnv(mode="client",host="localhost", wait_next=False) #initiates a WS server in port 4567
        rospy.loginfo('done!')
        self._publisher = rospy.Publisher(self._steer_topic, Control, queue_size=queue)
        self._subcriber = rospy.Subscriber("capture/status", Status, self.callback_fn, queue_size = queue)
        self.recording = False
    def callback_fn(self, data):
        self.recording = data.recording 

def main():
    # In ROS, nodes are uniquely named. If two nodes with the same name are launched, the previous one is kicked off. The anonymous=True flag means that rospy will choose a unique 
    # name for our 'listener' node so that multiple listeners can run simultaneously.
    rospy.init_node('bot_client_capture', anonymous=True)
    #rospy.on_shutdown(fn)
    server = CarServer()
    r = rospy.Rate(20)
    msg = Control()
    web_pub = rospy.Publisher("web_client/control", WebControl, queue_size=2)
    web_msg = WebControl()
    web_msg.auto_toggle = True
    web_msg.speed = 0.0
    web_msg.turn = 3.0
    web_msg.pan = 5.0
    web_msg.tilt = 0.0
    web_msg.auto_toggle = True
    web_msg.door_toggle = False
    web_msg.capture_toggle = False
    web_msg.shift = False

  
    pub = rospy.Publisher("web_client/arm_request", Bool, queue_size=2)
    start_msg = Bool()
    start_msg.data = True
    time.sleep(1)
    print("publishing to... start")
    pub.publish(start_msg)
    time.sleep(1)
    web_pub.publish(web_msg)

    lid_opened = False
    recording = False
    while not rospy.is_shutdown():
        response = server._ws_server.get_data()
        server._ws_server.emit(recording=server.recording)
        #print(response, server.recording)
        msg.speed = float(response.get("throttle", 0.0))
        msg.turn = float(response.get("steering_angle", 0.0))
        lid_res = response.get("lid_opened", "false") == "true"
        recording_res = response.get("recording", "false") == "true"
       
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

        server._publisher.publish(msg)
  
        r.sleep() 

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
    except:
        rospy.loginfo(sys.exc_info())
