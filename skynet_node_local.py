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
import time, random, cv2, math, subprocess
import tensorflow as tf
import cytoolz as cz
import numpy as np
import rospy, base64, argparse

from model import Model
from model import get_templates
from name import network_name, model_path
from tfinterface.supervised import SupervisedInputs
from parameters import Parameters
from pilot_cli.environments import CarEnv
from skynet.msg import Control
from std_msgs.msg import Bool
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String


def checkout_download_model(dir_path, branch):

    os.chdir(dir_path)
    process = subprocess.check_call("git checkout " + branch, shell=True, stdout=subprocess.PIPE)
    process = subprocess.check_call(os.path.join(dir_path,"floyd","download-model"), shell=True, stdout=subprocess.PIPE)


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

    model = model_t(inputs, mode = tf.estimator.ModeKeys.PREDICT)

    restore = True
    # initialize variables
    print("Initializing Model: restore = {}".format(restore))
    model.initialize(restore = restore)

    return model



class WsSkynetClient:
    def __init__(self, steer_topic = "skynet/control", camera_topic = "capture/image/compressed", queue=2):
        self._steer_topic = steer_topic
        self._camera_topic = camera_topic

        rospy.loginfo('Creating ws client')
        self._ws_client = CarEnv(mode="client",host="localhost", wait_next=False) #initiates a WS client in port 4567
        rospy.loginfo('done!')

        self._steer_publisher = rospy.Publisher(self._steer_topic, Control, queue_size=queue)



def main():

    # parser = argparse.ArgumentParser()
    # parser.add_argument('-r', '--rate', dest='rate', type=int,
    #                     default=20, help='rate of node')
    # parser.add_argument('-s', '--speed', dest='speed', type=float,
    #                     default = 0.3, help = 'speed of pilot-net')
    # args = parser.parse_args()


    # In ROS, nodes are uniquely named. If two nodes with the same name are launched, the previous one is kicked off. The anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can run simultaneously.
    rospy.init_node('skynet_node', anonymous=True)

    # Publish that it is not ready yeet
    skynet_pub = rospy.Publisher("skynet/status", Bool, queue_size=2)
    skynet_status = Bool()
    skynet_status = False
    time.sleep(1)
    skynet_pub.publish(skynet_status)

    cam = 1
    quality = 50
    size = (480/1,640/1)
    rate = 20 #args.rate
    speed = 0.33 #args.speed

    client = WsSkynetClient()
    r = rospy.Rate(rate)

    params = Parameters()
    model = init_model()
    print("Model Loaded!")

    # Publish that Network is ready!
    skynet_status = True
    skynet_pub.publish(skynet_status)

    video = cv2.VideoCapture(cam)

    def fn():
        skynet_status = False
        skynet_pub.publish(skynet_status)
        video.release()

    rospy.on_shutdown(fn)

    msg = Control()

    time_counter = time.time()

    while not rospy.is_shutdown():

        response = client._ws_client.get_data()

        ret, img = video.read()

        img = cv2.flip(img[...,::-1],-1) #converts from BGR to RGB and flips image

        auto_throttle = float(response.get("auto_throttle", speed))
        auto_steering = float(response.get("auto_steering", 1.0))

        desired_branch = response.get("network_branch", "")

        send_camera = response.get("send_camera", "false") == "true"

        autonomous = rospy.get_param('/rover_teleop/mode')

        if "network_branch" in response:
            try:
                skynet_pub.publish(False)
                checkout_download_model(os.path.join(dir_path,"pilot-net"),desired_branch)
                rospy.signal_shutdown("Changing AI")
            except Exception as e:
                print(e)


        if img is not None:
            # print(np.ndarray.mean(img))
            img = utils.crop_image(img, params)

            if autonomous == "auto":

                predictions = model.predict(image=[img])
                msg.speed = auto_throttle
                msg.turn = predictions[0,0]*auto_steering

                # Publish data to Local Front and Main Controler
                client._ws_client.emit(steering=msg.turn)
                client._steer_publisher.publish(msg)


        if send_camera:

            if time.time()-time_counter >= 3 and img is not None:
                _, buff = cv2.imencode(".jpg", img[...,::-1], [cv2.IMWRITE_JPEG_QUALITY, quality])
                print("Sending image...")
                response = client._ws_client.emit(image=base64.b64encode(buff))
                time_counter = time.time()

        else:
            time_counter = time.time()


        r.sleep()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
    except:
        rospy.loginfo(sys.exc_info())
