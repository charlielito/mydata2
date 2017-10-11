#!/usr/bin/env python

import sys, os
dir_path = os.path.dirname(os.path.realpath(__file__))
# folder = "pilot-net/scripts"
# sys.path.insert(0, os.path.join(dir_path,folder))
# os.environ["MODEL_PATH"] = os.path.join(dir_path, folder)
sys.path.append(os.path.join(dir_path, "pilot-net", "scripts"))
os.environ["MODEL_PATH"] = os.path.join(dir_path, "pilot-net")
os.environ["PARAMS_FOLDER"] = os.path.join(dir_path, "pilot-net")

import utils
from parameters import Parameters

import time, random, cv2, math
import tensorflow as tf
import cytoolz as cz
import numpy as np
from model import Model
from model import get_templates
from name import network_name, model_path
from tfinterface.supervised import SupervisedInputs

import rospy, base64, std_srvs.srv, argparse, cv2, time
from pilot_cli.environments import CarEnv
from sensor_msgs.msg import CompressedImage
from bot_server.msg import Control
from web_client.msg import Control as WebControl
from capture_3cameras.msg import Status
from std_msgs.msg import Bool
from sensor_msgs.msg import CompressedImage



def set_video_size(video, size):
    video.set(cv2.CAP_PROP_FRAME_HEIGHT, size[0])
    video.set(cv2.CAP_PROP_FRAME_WIDTH, size[1])

def init_model():
    # seed: resultados repetibles
    seed = 32
    np.random.seed(seed=seed)
    random.seed(seed)

    graph = tf.Graph()
    sess = tf.Session(graph=graph)

    # inputs
    inputs_t, model_t = get_templates(sess, graph, seed, queue = False, batch_size = 100, variable_input = True)

    inputs = inputs_t()

    model = model_t(inputs)

    restore = True
    # initialize variables
    print("Initializing Model: restore = {}".format(restore))
    model.initialize(restore = restore)

    return model



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

    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--rate', dest='rate', type=int,
                        default=20, help='Number of workers.')
    parser.add_argument('-c', '--camera', dest='camera', type=int,
                        default=1, help='Size of the queue.')
    parser.add_argument('-s', '--speed', dest='speed', type=float,
                        default = 0.3, help = 'recording status')
    parser.add_argument('-q', '--quality', dest='quality', type=int,
                        default = 50, help = 'recording status')
    args = parser.parse_args()


    # In ROS, nodes are uniquely named. If two nodes with the same name are launched, the previous one is kicked off. The anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can run simultaneously.
    rospy.init_node('bot_autocontroler', anonymous=True)


    #quality = 10
    size = (480/1,640/1)
    rate = args.rate
    camera = args.camera
    speed = args.speed
    quality = args.quality

    server = CarServer()
    r = rospy.Rate(rate)

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
    # pub.publish(start_msg)
    time.sleep(1)
    # web_pub.publish(web_msg)
    web_msg.auto_toggle = False

    lid_opened = False
    recording = False


    video = cv2.VideoCapture(camera)

    msg = Control()

    def fn():
        video.release()

    rospy.on_shutdown(fn)

    set_video_size(video, size)

    params = Parameters()
    model = init_model()
    print("Model Loaded!")

    time_counter = 0

    while not rospy.is_shutdown():

        _, img = video.read()
        assert img is not None, "device on /dev/video{} is not beeing recognized".format(camera)
        img = cv2.flip(img,-1)
        img = utils.crop_image(img, params)

        response = server._ws_server.get_data()
        server._ws_server.emit(recording=server.recording)
        #print(response, server.recording)
        msg.speed = float(response.get("throttle", 0.0))
        msg.turn = float(response.get("steering_angle", 0.0))
        lid_res = response.get("lid_opened", "false") == "true"
        recording_res = response.get("recording", "false") == "true"

        autonomous = response.get("autonomous", "false") == "true"
        auto_throttle = float(response.get("auto_throttle", speed))
        auto_steering = float(response.get("auto_steering", 1.0))

        send_camera = response.get("send_camera", "false") == "true"

        if send_camera:

            if time_counter%(rate*2) == 0:
                _, buff = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
                print("Sending image...")
                response = server._ws_server.emit(image=base64.b64encode(buff))
                time_counter = -1

            time_counter += 1

        else:
            time_counter = 0


        if autonomous and (msg.speed == 0 and msg.turn == 0):
            predictions = model.predict(image=[img])
            msg.speed = auto_throttle
            msg.turn = predictions[0,0]*auto_steering
            server._ws_server.emit(steering=msg.turn)

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
