#!usr/bin/env python3
import rospy
from custom_msg.msg import custom

def pose_callback(msg: custom):
    rospy.loginfo('msg: (' + str(msg.x) + ', '+ str(msg.y) + ', ' + str(msg.a) + ', ' + str(msg.c) + ' )')


if __name__ == '__main__':
    rospy.init_node('turtle_pose_subscirber')
    sub = rospy.Subscriber('/detection_topic', custom, callback=pose_callback)
    rospy.loginfo('Node has been started')

    rospy.spin() #block until node is shutdown, sort of infinite loop
