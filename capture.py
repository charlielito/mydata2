#!/usr/bin/env python

#### Node to colect data from 3 USB cameras
#### The data is going to be saved on an USB mounted on /media in folder /media/test-run/

import rospy, time
from mavros_msgs.msg import ActuatorControl
import numpy as np
import cv2, os, csv, sys, binascii
from capture_3cameras.msg import Status
from web_client.msg import Control

from std_msgs.msg import String



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

    return msg


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
    cv2.imwrite(os.path.join(dest, name_l),cv2.flip(frame0,-1), [cv2.IMWRITE_JPEG_QUALITY, quality])
    cv2.imwrite(os.path.join(dest, name_c),cv2.flip(frame1,-1), [cv2.IMWRITE_JPEG_QUALITY, quality])
    cv2.imwrite(os.path.join(dest, name_r),cv2.flip(frame2,-1), [cv2.IMWRITE_JPEG_QUALITY, quality])

    msg.left_camera_mean = np.ndarray.mean(frame0)
    msg.center_camera_mean = np.ndarray.mean(frame1)
    msg.right_camera_mean = np.ndarray.mean(frame2)

    return name_l, name_c, name_r


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

        self._subcriber = rospy.Subscriber(self._mavros_topic, ActuatorControl, self.callback_fn,  queue_size = queue)
        self._subcriber = rospy.Subscriber(self._webclient_topic, Control, self.callback_fn2,  queue_size = queue)

        self._subcriber_cameras_config = rospy.Subscriber('bot_server/camera_config', String, self.callback_fn3, queue_size=queue)

        self.steering_angle = 0.0
        self.throttle = 0.0
        self.capturing = False

        self.left_camera_num = 0
        self.center_camera_num = 1
        self.right_camera_num = 2

    def callback_fn3(self, data):
        numbers = map(int, data.data.split(","))
        self.left_camera_num, self.center_camera_num, self.right_camera_num = numbers[0], numbers[1], numbers[2]

    def callback_fn(self, data):

        if not data.group_mix:
            self.steering_angle = data.controls[2]
            self.throttle = data.controls[3]
            #rospy.loginfo("steering_angle {}, throtle: {}".format(self.steering_angle, self.throttle) )

    def callback_fn2(self, data):
        if data.capture_toggle:
             self.capturing = not self.capturing



def main():



    date = time.strftime("%d-%m-%y")
    bot_id = int(os.environ['KIWIBOT_ID'])

    device = "/media"
    dest = os.path.join(device,'kiwibot'+str(bot_id)+'-'+date)
    #dest = "/home/pi/data"

    quality = 80.0
    #size = (1080,1024)
    size = (480,640)

    if os.path.exists(dest):
        print('exists')

    else:
        print('does not')
        os.mkdir(dest)


    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('capture', anonymous=True)

    bot = KiwiBot()

    r = rospy.Rate(100)

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


    set_video_size(video0, size)
    set_video_size(video1, size)
    set_video_size(video2, size)

    rospy.on_shutdown(fn)

    csv_file = os.path.join(dest,'data.csv')

    if not os.path.isfile(csv_file):
        with open(csv_file, 'a') as fd:
            writer = csv.writer(fd)
            row = ['bot_id', 'timestamp', 'filename', 'camera', 'throttle', 'steering']
            writer.writerow(row)

    videos = [video0, video1, video2]

    with open(csv_file, 'a') as fd:
        writer = csv.writer(fd)
        i = 0 #counter to check disk for space
        while not rospy.is_shutdown():


            if bot.capturing:

                start_time = time.time()

                # CVS Structure:
                # bot_id, timestamp, left, center, right, throttle, steering_angle
                # bot_id, timestamp, filename, camera, throttle, steering_angle

                left, center, right = capture_write(videos[bot.left_camera_num], videos[bot.center_camera_num], videos[bot.right_camera_num], dest, quality)

                throttle = bot.throttle
                steering_angle = bot.steering_angle
                # row = [bot_id, start_time, left, center, right, throttle, steering_angle]
                # writer.writerow(row)
                rows = [[bot_id, start_time, left, 0, throttle, steering_angle],
                       [bot_id, start_time, center, 1, throttle, steering_angle],
                       [bot_id, start_time, right, 2, throttle, steering_angle]]
                writer.writerows(rows)

                fps = 1.0/(-start_time+time.time())
                print("FPS: {}".format(fps))

                msg.recording = True
                msg.fps = fps


            else:
                print("Not capturing")
                msg.fps = 0
                msg.recording = False

                msg = test_capture(videos[bot.left_camera_num], videos[bot.center_camera_num], videos[bot.right_camera_num], msg)
                time.sleep(0.1)

            if i%(7*60)==0: #each minute reads from disk
                msg.space_left = space_left(device_path=device, percentage = True)
                i = 0
                print('Reading Usb Space')

            publisher.publish(msg)
            i += 1
            r.sleep()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        rospy.loginfo("Some Error Ocurred!!")
    except:
        rospy.loginfo(sys.exc_info())
