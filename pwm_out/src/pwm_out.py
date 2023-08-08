#!/usr/bin/python3

import rospy
from std_msgs.msg import Float32MultiArray
import sys
import signal
from helper import cmd_out

class Pwm_out:
    def __init__(self):
        rospy.init_node('pwm_node', anonymous=True)
        self.pub = rospy.Publisher('pwm_topic', Float32MultiArray, queue_size=10)
        self.sub = rospy.Subscriber('tracker_topic', Float32MultiArray, self.callBack)
        
    def publish_message(self, message):
        msg = Float32MultiArray()
        msg.data = message
        self.pub.publish(msg)
    
    def callBack(self, message):
        center_x, height = message.data
        if center_x == 0 and height == 0:
            self.publish_message([0,0])
            rospy.loginfo("leftRightPwm: %f ; forwardBackwardPwm: %f" % (0, 0))
        else:
            leftRightPwm, forwardBackwardPwm = cmd_out(center_x, height, LeftRightThreshold=0.2, min_height=0.7, max_height=0.9)
            self.publish_message([leftRightPwm, forwardBackwardPwm])
            rospy.loginfo("leftRightPwm: %f ; forwardBackwardPwm: %f" % (leftRightPwm, forwardBackwardPwm))

    def cleanup(self, sig, frame):
        rospy.loginfo("Cleaning up")
        for _ in range(10):
            self.publish_message([0,0])
        sys.exit(0)

    def run(self):
        signal.signal(signal.SIGINT, self.cleanup)
        rospy.spin()

if __name__ == '__main__':
    try:
        pwm_out = Pwm_out()  
        pwm_out.run()
    except:
        pass
