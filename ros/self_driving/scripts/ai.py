#!/usr/bin/env python
# license removed for brevity
import rospy
from self_driving.msg import four_floats

def talker():
    pub = rospy.Publisher('ai', four_floats, queue_size=10)
    rospy.init_node('AI', anonymous=True)
    rate = rospy.Rate(10) # 10hz

    msg = four_floats()
    while not rospy.is_shutdown():
        
        msg.ai = 0
        
        
        pub.publish(msg)
        rate.sleep()

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass
