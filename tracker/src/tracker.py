#!/usr/bin/python3

from jetson_inference import detectNet
from jetson_utils import videoSource, videoOutput

import rospy
from std_msgs.msg import Float32MultiArray

class Tracker:
    def __init__(self, network="ssd-mobilenet-v2", camera_source="/dev/video0", threshold=0.5, tracking_frames=0):
        rospy.init_node('tracker_node', anonymous=True)
        self.pub = rospy.Publisher('tracker_topic', Float32MultiArray, queue_size=10)

        self.tracking_frames = tracking_frames
        self.net = detectNet(network, threshold=threshold)
        self.camera = videoSource(camera_source)
        if self.tracking_frames:
            self.net.SetTrackingEnabled(True)
            self.net.SetTrackingParams(minFrames=3, dropFrames=self.tracking_frames, overlapThreshold=0.5)

        try:
            self.display = videoOutput("display://0")
            self.display_show = True
        except:
            self.display_show = False

    def publish_message(self, message):
        msg = Float32MultiArray()
        msg.data = message
        self.pub.publish(msg)

    def run(self):
        tracked_id = -1     # default status for untracked
        lost_frame_count = 0    # count this in order to know when we have officially lost track of our target

        while not rospy.is_shutdown():
            is_tracked_object_in_frame = False
            if tracked_id == -1:
                largest_height = 0        # This is to allow us to find human with the largest box size so we can start tracking
                temporary_id = -1
                self.publish_message([0, 0])
            img = self.camera.Capture()

            if img is None: # capture timeout
                continue

            detections = self.net.Detect(img)
            for detection in detections:
                if detection.ClassID == 1:     # we only track humans
                    # width = detection.Width / img.width
                    height = detection.Height / img.height
                    # area = width * height
                    center_x = detection.Center[0] / img.width

                    # if there is currently no tracked object, we will track
                    if self.tracking_frames > 0 and tracked_id == -1 and height > largest_height:
                        temporary_id = detection.TrackID

                    # if we are already tracking an object and the object is in our frame
                    elif self.tracking_frames > 0 and detection.TrackID == tracked_id:
                        is_tracked_object_in_frame = True
                        rospy.loginfo("Center: %f ; Height: %f" % (center_x, height))
                        self.publish_message([center_x, height])   

            if tracked_id == -1 and temporary_id != -1:
                tracked_id = temporary_id
                lost_frame_count = 0                 
            
            if is_tracked_object_in_frame == False:
                lost_frame_count += 1
                if lost_frame_count > self.tracking_frames:
                    tracked_id = -1
                # self.publish_message([0,0])
            print("\n\n")

            if self.display_show:
                self.display.Render(img)
                self.display.SetStatus("Object Detection | Network {:.0f} FPS".format(self.net.GetNetworkFPS()))

if __name__=="__main__":
    try:
        tracker = Tracker(tracking_frames=30)
        rospy.loginfo("Tracker node is running.")
        tracker.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("Tracker node has stopped.")
        pass

