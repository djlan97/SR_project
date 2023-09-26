#!usr/bin/env python3
import rospy
from custom_msg.msg import custom

if __name__ == '__main__':
    rospy.init_node('det_pub')
    rospy.loginfo('Node has been started')

    pub = rospy.Publisher('/detection_topic',custom,queue_size=10)

    rate = rospy.Rate(1) # 2 times x s

    while not rospy.is_shutdown():
        #publish
        msg = custom()
        msg.x1 = 341
        msg.y1 = 226
        msg.a1 = -0.13
        msg.c1 = 1
        msg.x2 = 1.2
        msg.y2 = 10.2
        msg.a2 = 1.2
        msg.c2 = 2
        msg.x3 = 1.3
        msg.y3 = 10.3
        msg.a3 = 1.3
        msg.c3 = 3
        msg.x4 = 1.4
        msg.y4 = 10.4
        msg.a4 = 1.4
        msg.c4 = 4
        pub.publish(msg)
        rate.sleep() #regola il loop secondo rate
