#!/usr/bin/env python
# license removed for brevity
import rospy
from self_driving.msg import four_floats
import pygame, sys

def talker():
    pub = rospy.Publisher('keyboard', four_floats, queue_size=10)
    rospy.init_node('KEYBOARD', anonymous=True)
    rate = rospy.Rate(10) # 10hz

    msg = four_floats()
    step = 1
    x = 0
    y = 0
    y_before = 0
    pygame.init()
    pygame.display.set_caption('Car Control Keyboard')
    size = [640, 480]
    screen = pygame.display.set_mode(size)
    clock = pygame.time.Clock()

    msg.keyboard = [0, 0, 0, 0]
    pygame.key.set_repeat(50, 50)
    while not rospy.is_shutdown():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    x -= step
                    if(x <- 15):
                        x = -15

                if event.key == pygame.K_RIGHT:
                    x += step
                    if(x > 15):
                        x = 15

                if event.key == pygame.K_UP:
                    y += step
                    if(y > 3):
                        y = 3
   
                if event.key == pygame.K_DOWN:
                    y -= step
                    if (y< -1):
                        y = -1
                if event.key == pygame.K_s:
                    z = 1                   
                    print('keyboard running start')
                if event.key == pygame.K_q:
                    z = 0
                    print('keyboard running stop')

                msg.keyboard[0] = z
                msg.keyboard[1] = x
                msg.keyboard[2] = y
                msg.keyboard[3] = y_before
                              
        clock.tick(10)

        pub.publish(msg)
        print(msg.keyboard)
        y_before = y
        rate.sleep()

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass
