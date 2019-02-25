import cv2
import numpy as np


def imageCopy(src):
    return np.copy(src)


def polyROI(image, points):
    if len(image.shape) == 2:
        channels = 1
    else:
        channels = image.shape[2]
    mask = np.zeros_like(image)
    ignore_mask_color = (255,) * channels
    cv2.fillPoly(mask, points, ignore_mask_color)
    return cv2.bitwise_and(image, mask)


def convertColor(image, flag=None):
    if flag is None:
        return image
    else:
        return cv2.cvtColor(image, flag)


def rangeColor(image, lower, upper):
    return cv2.inRange(image, lower, upper)

def splitColor(image, lower, upper, flag=cv2.COLOR_BGR2HSV):
    converted = convertColor(image, flag)
    mask = rangeColor(converted, lower, upper)
    return cv2.bitwise_and(image, image, mask=mask)

def cannyEdge(image, threshold1=100, threshold2=200):
    return cv2.Canny(image, threshold1, threshold2)


def addWeightedImage(image1, w1, imagw2, w2=None):
    if w2 is None:
        return cv2.addWeighted(image1, float(w1) * 0.01, imagw2, float(100 - w1) * 0.01, 0)
    else:
        return cv2.addWeighted(image1, w1 * 0.01, imagw2, w2 * 0.01, 0)


def houghLinesP(image, rho=1.0, theta=np.pi/180, threshold=100, minLineLength=10, maxLineGap=100):
    return cv2.HoughLinesP(image, rho, theta, threshold, minLineLength=minLineLength, maxLineGap=maxLineGap)


def drawHoughLinesP(image, lines):
    result = imageCopy(image)
    if len(image.shape) == 2:
        result = convertColor(image, cv2.COLOR_GRAY2BGR)
    for i in range(len(lines)):
        for x1, y1, x2, y2 in lines[i]:
            cv2.line(result, (x1, y1), (x2, y2), (0, 0, 255), 3)
    return result


def frameProcessing(frame):
    result = imageCopy(frame)

    lower_white_hls = np.array([0, 150, 0])
    upper_white_hls = np.array([255, 255, 255])

    result = splitColor(result, lower_white_hls, upper_white_hls, cv2.COLOR_BGR2HLS)

    result = convertColor(result, cv2.COLOR_BGR2GRAY)
    result = cannyEdge(result, 100, 200)
    
    height, width = result.shape[:2]
    pt1 = (width*0.45, height*0.65)
    pt2 = (width*0.55, height*0.65)
    pt3 = (width, height*1.0)
    pt4 = (0, height*1.0)
    roi_corners = np.array([[pt1, pt2, pt3, pt4]], dtype=np.int32)

    result = polyROI(result, roi_corners)
    result_lines = houghLinesP(result, 1, np.pi/180, 30)

    try:
         result = drawHoughLinesP(frame, result_lines)
    except TypeError:
         result = result
    except IndexError:
         result = result
    return result


cap = cv2.VideoCapture('2.avi')

# Get frame per second information
fps = cap.get(cv2.CAP_PROP_FPS)
print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))
# Get width and height information
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print(width)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(height)
# Define the codec and create VideoWriter object
fourcc = int(cv2.VideoWriter_fourcc(*'DIVX'))
print(fourcc)
out = cv2.VideoWriter('output_Line_ROI.avi', fourcc, fps, (width, height), True)

cv2.namedWindow("Output", cv2.WINDOW_GUI_EXPANDED)

while cap.isOpened():
    # Capture frame-by-frame
    ret, frame = cap.read()

    if ret:
        # Our operations on the frame come here
        output = frameProcessing(frame)
        # Write frame-by-frame
        # Display the resulting frame
        cv2.imshow("Output", output)

    else:
        break

    # waitKey(int(1000.0/fps)) for matching fps of video
    if cv2.waitKey(int(1000.0/fps)) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
out.release()

cv2.destroyAllWindows()
