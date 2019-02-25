#!/usr/bin/env python
import rospy
from self_driving.msg import four_floats
import serial
from time import sleep


class Server:
    def __init__(self):
        self.keyboard = [0, 0, 0, 0]  # [keyboard authority 0 or 1, steer, speed, speed_before]
        self.ai = None  # ai object detection detection 0 or 1
        self.cam = [0, 0, 0]  # [steer, speed, speed_before]
        self.lds = [0, 0, 0]  # [lds object detection 0 or 1 , angle, distance]

        try:
            self.ser = serial.Serial('/dev/ttyACM0', 115200)
        except serial.serialutil.SerialException:
            print("Can't Find Serial Port")

    def value_2_msg(self, steer_float, speed_float):  # convert 2 floats to uart msg
        steer_int = int(steer_float)
        speed_int = int(speed_float)
        if steer_int <= 0:
            steer_int = abs(steer_int)
            if steer_int > 15:
                steer_int = 15
        else:
            steer_int = steer_int + 16
            if steer_int > 31:
                steer_int = 31
        if speed_int == -1:
            speed_int = 4
        print(steer_int, speed_int)
        msg_int = (steer_int << 3) | speed_int
        msg = str(msg_int)
        msg = msg.zfill(3)
        return msg

    def keyboard_callback(self, msg):
        self.keyboard = msg.keyboard
        self.car_control()

    def ai_callback(self, msg):
        self.ai = msg.ai
        self.car_control()

    def cam_callback(self, msg):
        self.cam = msg.cam
        self.car_control()

    def lds_callback(self, msg):
        self.lds = msg.lds
        self.car_control()

    def car_control(self):
        if self.keyboard[0] == 1:
            
            a = self.keyboard[1:4]
            speed_int = int(a[1])
            speed_before = a[2]

            msg = self.value_2_msg(a[0], a[1])
            msg_zero = self.value_2_msg(a[0], 0)


            if speed_int == -1:
                if speed_before != -1:
                    self.ser.write(msg)
                    self.ser.flush()
                    sleep(0.1)
                    self.ser.write(msg_zero)
                    self.ser.flush()
                    sleep(0.1)
            self.ser.write(msg)
            #except AttributeError:
                #print('msg from keyboard : Please Confirm Serial Port /dev/ttyACM0')

        elif self.keyboard[0] == 0:
            if self.ai or self.lds[0]:
                a = self.cam
                msg_zero = self.value_2_msg(a[0], 0)
                try:
                    self.ser.write(msg_zero)
                    if self.ai:
                        print('ai object detection')
                    else:
                        print('lds object detection / angle: %f distance: %f' % (self.lds[1], self.lds[2]))

                except AttributeError:
                    print('msg from ai or lds : Please Confirm Serial Port /dev/ttyACM0')

            else:
                print('self driving running')
                a = self.cam
                print(a)
                speed_int = int(a[1])        
                print(a[0])
                speed_before = a[2]
                
                msg = self.value_2_msg(a[0], a[1])
                msg_zero = self.value_2_msg(a[0], 0)


                if speed_int == -1:
                    if speed_before != -1:
                        self.ser.write(msg)
                        self.ser.flush()
                        sleep(0.1)
                        self.ser.write(msg_zero)
                        self.ser.flush()
                        sleep(0.1)
                self.ser.write(msg)
                print(msg)
  


if __name__ == '__main__':
    rospy.init_node('car_control', anonymous=True)

    server = Server()
    rospy.Subscriber("keyboard", four_floats, server.keyboard_callback)
    rospy.Subscriber("ai", four_floats, server.ai_callback)
    rospy.Subscriber("cam", four_floats, server.cam_callback)
    rospy.Subscriber("lds", four_floats, server.lds_callback)

    rospy.spin()


