#!/usr/bin/env python
# license removed for brevity
import rospy
from self_driving.msg import four_floats
import cv2
import numpy as np
import math


def drawLine(image, point1, point2, color=(255, 0, 0), thickness=5, lineType=cv2.LINE_AA):
    result = imageCopy(image)
    return cv2.line(result, point1, point2, color, thickness, lineType)

def draw_lane_lines(image, lines, color=[255, 0, 0], thickness=10):
    # make a separate image to draw lines and combine with the orignal later
    line_image = np.zeros_like(image)
    for line in lines:
        if line is not None:
            cv2.line(line_image, line[0], line[1],  color, thickness)

    # image1 and image2 must be the same shape.
    return cv2.addWeighted(image, 1.0, line_image, 0.95, 0.0)

def lane_lines(image, lines):
    if lines is None:
        return None
    
    left_lane, right_lane = average_slope_intercept(lines)
    
    y1 = image.shape[0]  # bottom of the image
    y2 = y1 * 0.6  # slightly lower than the middle

    left_line = make_line_points(y1, y2, left_lane)

    right_line = make_line_points(y1, y2, right_lane)
    
    if not left_line:
        return None, right_line
    if not right_line:
        return left_line, None 

    return left_line, right_line

def make_line_points(y1, y2, line):
    """
    Convert a line represented in slope and intercept into pixel points
    """
    if line is None:
        return None

    slope, intercept = line
    print(slope)
    
    if abs(slope)<=0.1:
        print(slope)
        return None

    # make sure everything is integer as cv2.line requires it
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    y1 = int(y1)
    y2 = int(y2)

    return ((x1, y1), (x2, y2))

def average_slope_intercept(lines):
    left_lines = []  # (slope, intercept)
    left_weights = []  # (length,)
    right_lines = []  # (slope, intercept)
    right_weights = []  # (length,)
    if lines is None:
        return None
    for line in lines:
        for x1, y1, x2, y2 in line:
            if x2 == x1:
                continue  # ignore a vertical line
            x1 = float(x1)
            y1 = float(y1)
            x2 = float(x2)
            y2 = float(y2)
            slope = (y2 - y1) / (x2 - x1)


            intercept = y1 - slope * x1
            length = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
            if slope < 0:  # y is reversed in image
                left_lines.append((slope, intercept))
                left_weights.append((length))
            else:
                right_lines.append((slope, intercept))
                right_weights.append((length))

    # add more weight to longer lines
    left_lane = np.dot(left_weights, left_lines) / np.sum(left_weights) if len(left_weights) > 0 else None

    right_lane = np.dot(right_weights, right_lines) / np.sum(right_weights) if len(right_weights) > 0 else None

    return left_lane, right_lane  # (slope, intercept), (slope, intercept)

def draw_lines(image, lines, color=[255, 0, 0], thickness=2, make_copy=True):
    # the lines returned by cv2.HoughLinesP has the shape (-1, 1, 4)
    if make_copy:
        image = np.copy(image) # don't want to modify the original
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(image, (x1, y1), (x2, y2), color, thickness)
    return image

def hough_lines(image):
    """
    `image` should be the output of a Canny transform.

    Returns hough lines (not the image with lines)
    """
    return cv2.HoughLinesP(image, rho=1, theta=np.pi / 180, threshold=20, minLineLength=20, maxLineGap=300)

def select_region(image):
    """
    It keeps the region surrounded by the `vertices` (i.e. polygon).  Other area is set to 0 (black).
    """
    # first, define the polygon by vertices
    rows, cols = image.shape[:2]
    bottom_left  = [0, rows]
    top_left     = [0, rows*0.85]
    bottom_right = [cols, rows]
    top_right    = [cols, rows*0.85]
    # the vertices are an array of polygons (i.e array of arrays) and the data type must be integer
    vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    return filter_region(image, vertices)

def filter_region(image, vertices):
    """
    Create the mask using the vertices and apply it to the input image
    """
    mask = np.zeros_like(image)
    if len(mask.shape)==2:
        cv2.fillPoly(mask, vertices, 255)
    else:
        cv2.fillPoly(mask, vertices, (255,)*mask.shape[2]) # in case, the input image has a channel dimension
    return cv2.bitwise_and(image, mask)

def detect_edges(image, low_threshold=50, high_threshold=150):
    return cv2.Canny(image, low_threshold, high_threshold)

def apply_smoothing(image, kernel_size=15):
    """
    kernel_size must be postivie and odd
    """
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def convert_gray_scale(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def convert_hls(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2HLS)

def select_white_yellow(image):
    converted = convert_hls(image)
    # white color mask
    lower = np.uint8([  0, 200,   0])
    upper = np.uint8([255, 255, 255])
    white_mask = cv2.inRange(converted, lower, upper)
    # yellow color mask
    lower = np.uint8([ 10,   0, 100])
    upper = np.uint8([ 40, 255, 255])
    yellow_mask = cv2.inRange(converted, lower, upper)
    # combine the mask
    mask = cv2.bitwise_or(white_mask, yellow_mask)
    return cv2.bitwise_and(image, image, mask = mask)

def imageCopy(src):
    return np.copy(src)

def frameProcessing(frame):
    result = imageCopy(frame)
    result = select_white_yellow(result)
    result = convert_gray_scale(result)
    result = apply_smoothing(result)
    result = detect_edges(result)
    result = select_region(result)


    result_lines = hough_lines(result)

    result_lines = lane_lines(result, result_lines)
    print(result_lines)

    degree = 0
    speed = 0

    if result_lines is None:
        degree = 0
        speed = 0
    
    elif result_lines is (None, None):
        degree = 0
        speed = 0    

    elif result_lines[0] and result_lines[1]:

        left_line_bottom = result_lines[0][0]
        left_line_top = result_lines[0][1]

        right_line_bottom = result_lines[1][0]
        right_line_top = result_lines[1][1]    
        middle_line_bottom = (int((left_line_bottom[0]+right_line_bottom[0])/2), left_line_bottom[1])

        middle_line_top = (int((left_line_top[0]+right_line_top[0])/2), left_line_top[1])    
        middle_line = (middle_line_bottom, middle_line_top)
        middle_line_slope = (float(middle_line_top[0]-middle_line_bottom[0])/float(middle_line_bottom[1]-middle_line_top[1]))
        result = drawLine(result, middle_line_bottom, middle_line_top)
        rad = math.atan(middle_line_slope)
        degree = (180 / 3.141592) * rad * 0.35
        speed = 1
        result = draw_lane_lines(result, result_lines)
    elif result_lines[0]:
        degree = 5
        speed = 1
        result = draw_lane_lines(result, result_lines)
    elif result_lines[1]:
        degree = -5
        speed = 1
        result = draw_lane_lines(result, result_lines)

    return (result, degree, speed)


def talker():
    pub = rospy.Publisher('cam', four_floats, queue_size=10)
    rospy.init_node('CAM', anonymous=True)
    rate = rospy.Rate(10)  # 10hz
    cap = cv2.VideoCapture('/dev/video1')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cv2.namedWindow("Input", cv2.WINDOW_GUI_EXPANDED)
    cv2.namedWindow("Output", cv2.WINDOW_GUI_EXPANDED)

    msg = four_floats()
    speed_before = 0
    while not rospy.is_shutdown():
        ret, frame = cap.read()
        if ret:
            # Our operations on the frame come here
            output = frameProcessing(frame)
            # Write frame-by-frame
            # Display the resulting frame
            cv2.imshow("Input", frame)
            cv2.imshow("Output", output[0])

            steer = output[1]
            speed = output[2]

        else:
            break

        # waitKey(int(1000.0/fps)) for matching fps of video
        if cv2.waitKey(int(1000.0 / fps)) & 0xFF == ord('q'):
            break
        msg.cam = [steer, speed, speed_before]
        print(msg.cam)

        pub.publish(msg)
        speed_before = speed
        rate.sleep()


if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass
