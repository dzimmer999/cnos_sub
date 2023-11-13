#!/usr/bin/env python
import rospy
import os, sys
# sys.path.append(os.path.join(os.path.dirname(__file__), "..", "lib", "cnos"))
# import cnos
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

import subprocess
import json

class Cnos_Sub:
    def __init__(self):
        # Initializing node
        rospy.init_node("cnos_sub")
        rospy.loginfo("Initialised cnos node")

        self.img_sub = rospy.Subscriber('in_img_topic', Image, queue_size=10, callback=self.img_callback)
        self.img_pub = rospy.Publisher('out_img_topic', Image, queue_size=10)
        self.bridge = CvBridge()
        self.rgb_path = None
        self.dir_path = None

        # RGB path is just for testing
        self.cnos_path = os.path.join(os.path.dirname(__file__), "..", "lib", "cnos")
        self.template_dir = os.path.join(self.cnos_path, "tmp", "custom_dataset")
        self.rgb_path = os.path.join(self.cnos_path, "media", "hsr","test_image1.png")
        self.stability_sore_thresh = 0.5
        self.max_num_dets = 1
        self.confg_threshold = 0.5
        # In order to save the model (can't be tested since my imports don't work)
        # self.model = cnos.src.scripts.create_template_custom.run_inference(self.template_dir, self.stability_sore_thresh, return_model=True)

    def img_callback(self, msg):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except CvBridgeError as e:
            print(e)

        # If self.model can be saved, try this:
        # cnos.src.scripts.run_inference_from_custom_template.run_inference(self.template_dir, img, 3, 0.5, self.model)
        # If it doesn't there is also no need to run create_template_custom
        # (change rgb_path to rgb in run_inference)
        # cnos.src.scripts.inference_custom.run_inference(self.template_dir, img, self.num_max_dets, self.confg_threshold, self.stability_sore_thresh)

        command = ["python", "-m", "src.scripts.inference_custom", "--template_dir", self.template_dir, "--rgb_path", self.rgb_path, \
                    "--stability_score_thresh", str(self.thresh)]
        os.chdir(self.cnos_path)
        proc = subprocess.run(command)


if __name__ == '__main__':
    
    cnos_node = Cnos_Sub()

