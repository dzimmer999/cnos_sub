#!/usr/bin/env python
import rospy
import os, sys
import numpy as np
import cv2
from sensor_msgs.msg import Image
from PIL import Image as PILImage
from cv_bridge import CvBridge, CvBridgeError
import subprocess

# Cnos
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "lib", "cnos"))
import cnos
from cnos.src.scripts.inference_custom import run_inference as run_inference_custom_without_model
# from cnos.src.scripts.create_template_custom import run_inference as create_template
# from cnos.src.scripts.run_inference_from_custom_template import run_inference as run_inference_custom_with_model
class Cnos_Sub:
    def __init__(self):
        # Initializing node
        rospy.init_node("cnos_sub")
        rospy.loginfo("Initialised cnos node")

        self.img_sub = rospy.Subscriber('/usb_cam/image_raw', Image, queue_size=10, callback=self.img_callback)
        self.bridge = CvBridge()

        self.cnos_path = os.path.join(os.path.dirname(__file__), "..", "lib", "cnos")
        self.template_dir = os.path.join(self.cnos_path, "tmp", "custom_dataset")
        self.rgb_path = os.path.join(self.cnos_path, "media","temp.png")
        self.stability_sore_thresh = 0.5
        self.max_num_dets = 1
        self.confg_threshold = 0.5
        self.num_max_dets = 1
        #os.chdir(self.cnos_path)
        #self.model, self.ref_feats = create_template(self.template_dir, self.stability_sore_thresh, return_model=True)
        self.multiple_runs = False
        self.inference_flag = False

    def img_callback(self, msg):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            img = PILImage.fromarray(np.uint8(img))
            img.save(self.rgb_path)
        except CvBridgeError as e:
            print(e)

        if self.multiple_runs is False and self.inference_flag is False:
            print("Running inference.")
            self.inference_flag = True
            os.chdir(self.cnos_path)
            # run_inference_custom_with_model(self.template_dir, self.rgb_path, 3, 0.5, self.model, self.ref_feats)
            run_inference_custom_without_model(self.template_dir, self.rgb_path, self.num_max_dets, self.confg_threshold, self.stability_sore_thresh)
            self.inference_flag = False


if __name__ == '__main__':
    
    cnos_node = Cnos_Sub()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
        cv2.destroyAllWindows()

