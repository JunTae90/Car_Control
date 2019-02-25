#!/usr/bin/env python
import rospy
from self_driving.msg import four_floats

def callback(data):
    print("keyboard data :", data.keyboard)
    print("ai data :", data.ai)
    print("cam data :", data.cam)
    print("lds data :", data.lds)
    
def listener():

    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('four_floats_subscriber', anonymous=True)

    rospy.Subscriber("four_floats", four_floats, callback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    listener()
