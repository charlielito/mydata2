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
import time, random, cv2, math, subprocess, csv, binascii
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
from mavros_msgs.msg import ActuatorControl


#### CAPTURE ####################################
from capture_3cameras.msg import Status
from web_client.msg import Control

####### CAPTURE ############################
def set_video_size(video, size):
    video.set(cv2.CAP_PROP_FRAME_HEIGHT, size[0])
    video.set(cv2.CAP_PROP_FRAME_WIDTH, size[1])

def space_left(device_path, percentage = True):
    disk = os.statvfs(device_path)
    totalBytes = float(disk.f_bsize*disk.f_blocks)
    totalAvailSpace = float(disk.f_bsize*disk.f_bfree)
    if percentage:
        return totalAvailSpace/totalBytes * 100
    else:
        return totalAvailSpace/1024/1024/1024

def test_capture(video0, video1, video2, msg):
    ret, frame0 = video0.read()
    ret, frame1 = video1.read()
    ret, frame2 = video2.read()

    assert frame0 is not None, "device on /dev/video0 is not beeing recognized"
    assert frame1 is not None, "device on /dev/video1 is not beeing recognized"
    assert frame2 is not None, "device on /dev/video2 is not beeing recognized"

    msg.left_camera_mean = np.ndarray.mean(frame0)
    msg.center_camera_mean = np.ndarray.mean(frame1)
    msg.right_camera_mean = np.ndarray.mean(frame2)

    return msg, cv2.flip(frame1,-1)


def capture_write(video0, video1, video2, dest, quality):
    ret, frame0 = video0.read()
    ret, frame1 = video1.read()
    ret, frame2 = video2.read()

    assert frame0 is not None, "device on /dev/video0 is not beeing recognized"
    assert frame1 is not None, "device on /dev/video1 is not beeing recognized"
    assert frame2 is not None, "device on /dev/video2 is not beeing recognized"

    timestamp = int (time.time() * 1000)
    prefix = binascii.b2a_hex(os.urandom(2))
    name_l = '{}-{}_{}.jpg'.format(prefix, timestamp, "l")
    name_c = '{}-{}_{}.jpg'.format(prefix, timestamp, "c")
    name_r = '{}-{}_{}.jpg'.format(prefix, timestamp, "r")
    frame1 = cv2.flip(frame1,-1)
    cv2.imwrite(os.path.join(dest, name_l),cv2.flip(frame0,-1), [cv2.IMWRITE_JPEG_QUALITY, quality])
    cv2.imwrite(os.path.join(dest, name_c),frame1, [cv2.IMWRITE_JPEG_QUALITY, quality])
    cv2.imwrite(os.path.join(dest, name_r),cv2.flip(frame2,-1), [cv2.IMWRITE_JPEG_QUALITY, quality])

    return timestamp, name_l, name_c, name_r, frame1


def create_status_msg(size=(480,640),quality=80, device="/media"):
    msg = Status()
    msg.quality = quality
    msg.resolution = str(size[0])+"x"+str(size[1])
    msg.space_left = space_left(device_path=device, percentage = True)
    msg.fps = 0
    msg.recording = False
    msg.left_camera_mean = -1
    msg.center_camera_mean = -1
    msg.right_camera_mean = -1
    return msg


class KiwiBot:
    def __init__(self, mavros_topic = "mavros/actuator_control", webclient_topic = "web_client/control", queue=2):
        self._mavros_topic = mavros_topic
        self._webclient_topic = webclient_topic

        self._mavros_subcriber = rospy.Subscriber(self._mavros_topic, ActuatorControl, self.callback_fn,  queue_size = queue)
        self._webclient_subcriber = rospy.Subscriber(self._webclient_topic, Control, self.callback_fn2,  queue_size = queue)

        self._cameras_config_subcriber = rospy.Subscriber('local_client/camera_config', String, self.callback_fn3, queue_size=queue)

        self.steering_angle = 0.0
        self.throttle = 0.0
        self.capturing = False

        self.left_camera_num = 0
        self.center_camera_num = 1
        self.right_camera_num = 2


    def callback_fn(self, data):

        if not data.group_mix:
            self.steering_angle = data.controls[2]
            self.throttle = data.controls[3]
            #rospy.loginfo("steering_angle {}, throtle: {}".format(self.steering_angle, self.throttle) )

    def callback_fn2(self, data):
        if data.capture_toggle:
             self.capturing = not self.capturing


    def callback_fn3(self, data):
        numbers = map(int, data.data.split(","))
        self.left_camera_num, self.center_camera_num, self.right_camera_num = numbers[0], numbers[1], numbers[2]


#############################################################


########### NEURAL NET ###########################
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

########################################################


def main():

    # parser = argparse.ArgumentParser()
    # parser.add_argument('-r', '--rate', dest='rate', type=int,
    #                     default=20, help='rate of node')
    # parser.add_argument('-s', '--speed', dest='speed', type=float,
    #                     default = 0.3, help = 'speed of pilot-net')
    # args = parser.parse_args()



    # In ROS, nodes are uniquely named. If two nodes with the same name are launched, the previous one is kicked off. The anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can run simultaneously.
    rospy.init_node('skynet_capture_node', anonymous=True)

    # Publish that it is not ready yeet
    skynet_pub = rospy.Publisher("skynet/status", Bool, queue_size=2)
    skynet_status = False
    time.sleep(1)
    skynet_pub.publish(skynet_status)


    ### Node Params #######
    quality = 80
    size = (480/1,640/1)
    rate = 20 #args.rate
    speed = 0.33 #args.speed

    date = time.strftime("%d-%m-%y")
    bot_id = int(os.environ['KIWIBOT_ID'])

    device = "/media"
    dest = os.path.join(device,'kiwibot'+str(bot_id)+'-'+date)
    #dest = "/home/pi/data"

    ### Create folder for capturing data
    if os.path.exists(dest):
        print('exists')

    else:
        print('does not')
        os.mkdir(dest)


    client = WsSkynetClient()
    r = rospy.Rate(rate)

    params = Parameters()
    model = None


    bot = KiwiBot()

    publisher = rospy.Publisher('capture/status', Status, queue_size=2)

    msg = create_status_msg(size=size,quality=quality, device=device)

    video0 = cv2.VideoCapture(0)
    video1 = cv2.VideoCapture(1)
    video2 = cv2.VideoCapture(2)

    def fn():
        video0.release()
        video1.release()
        video2.release()
        msg = create_status_msg()
        publisher.publish(msg)
        skynet_pub.publish(False)


    set_video_size(video0, size)
    set_video_size(video1, size)
    set_video_size(video2, size)

    rospy.on_shutdown(fn)

    msg = Control()


    ###### Create data.csv headers if it not exists
    csv_file = os.path.join(dest,'data.csv')

    if not os.path.isfile(csv_file):
        with open(csv_file, 'a') as fd:
            writer = csv.writer(fd)
            row = ['bot_id', 'timestamp', 'filename', 'camera', 'throttle', 'steering']
            writer.writerow(row)

    videos = [video0, video1, video2]


    time_counter = time.time()
    time_counter2 = time.time()

    while not rospy.is_shutdown():


        ############# CAPTURING #########################

        if bot.capturing:

            start_time = time.time()

            # CVS Structure:
            # bot_id, timestamp, filename, camera, throttle, steering_angle

            timestamp, left, center, right, image = capture_write(videos[bot.left_camera_num], videos[bot.center_camera_num], videos[bot.right_camera_num], dest, quality)
            throttle = bot.throttle
            steering_angle = bot.steering_angle

            rows = [[bot_id, timestamp, left, 0, throttle, steering_angle],
                   [bot_id, timestamp, center, 1, throttle, steering_angle],
                   [bot_id, timestamp, right, 2, throttle, steering_angle]]

            with open(csv_file, 'a') as fd:
               writer = csv.writer(fd)
               writer.writerows(rows)

            fps = 1.0/(-start_time+time.time())
            print("FPS: {}".format(fps))

            msg.recording = True
            msg.fps = fps


        else:
            print("Not capturing")
            msg.fps = 0
            msg.recording = False
            msg, image = test_capture(videos[bot.left_camera_num], videos[bot.center_camera_num], videos[bot.right_camera_num], msg)


            ###################################### ###########################

            ########################## SKYNET NODE #########################################

            response = client._ws_client.get_data()

            img = image[...,::-1] #converts from BGR to RGB

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

                        if model is None:
                            print("Loading Model...")
                            model = init_model()
                            print("Model Loaded!")
                            # Publish that Network is ready!
                            skynet_pub.publish(True)

                        else:
                            predictions = model.predict(image=[img])
                            msg.speed = auto_throttle
                            msg.turn = predictions[0,0]*auto_steering

                            # Publish data to Local Front and Main Controler
                            client._ws_client.emit(steering=msg.turn)
                            client._steer_publisher.publish(msg)


            if send_camera:

                if time.time()-time_counter >= 3 and img is not None:
                    _, buff = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
                    print("Sending image...")
                    response = client._ws_client.emit(image=base64.b64encode(buff))
                    time_counter = time.time()

            else:
                time_counter = time.time()

            ########################################################################################




        ##### DATA CAPTURE ########################

        if time.time()-time_counter2 >= 60: #each minute reads from disk
            msg.space_left = space_left(device_path=device, percentage = True)
            print('Reading Usb Space')
            time_counter2 = time.time()

        publisher.publish(msg)


        ##################################-------------################################################

        r.sleep()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
    except:
        system_info = sys.exc_info()
        pub = rospy.Publisher('web_client/message', String, queue_size=2)
        time.sleep(1)
        pub.publish(str(system_info))
        rospy.loginfo(system_info)
