import cv2
import numpy as np
import math


class OCVWindow:
    """
    OpenCV window to attach different filters
    """
    # Meta Image operations

    def __init__(self, name):
        self.name = name
        self.height = 0
        self.width = 0
        self.channels = 0
        self.img = 0
        self.img_array = []
        self.img_key = 0
        self.orgimg = 0
        self.cap = None
        cv2.namedWindow(name)  # cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        self.trackbar_dict = {}
        self.filter_list = []
        self.filter_list_prev = []
        self.filter_dict = {}
        self.filter_dict_prev = {}
        self.file = None

    def CopyFrom(self, copy, blank=False):
        """
        Copy the parameters from a child onto another.
        """
        # self.name = copy.name
        self.height = copy.height
        self.width = copy.width
        self.channels = copy.channels
        self.img = copy.img.copy()
        self.orgimg = self.img.copy()
        self.cap = copy.cap
        self.file = copy.file
        self.filter_list_prev = copy.filter_list_prev + copy.filter_list
        self.filter_dict_prev = dict(copy.filter_dict_prev)
        self.filter_dict_prev.update(copy.filter_dict)

        if blank:
            # creating a blank to draw lines on
            self.img = np.copy(self.img) * 0
            self.orgimg = np.copy(self.orgimg) * 0

    def AddImgFile(self, file):
        """
        Add given image to the namedWindow of CV2
        """
        self.img = cv2.imread(file, 1)

        self.file = file
        self.AddParameters()

    def AddImgBatch(self, file):
        """
        Add multiple images to a queue.
        also run CycleImg() during the While loop
        """
        self.img_array.append(file)
        #self.img_key += 1

    def CycleImg(self, key=-1):
        if key == -1:
            # cycle through images
            key = self.img_key
        else:
            # use specific image key
            pass

        self.AddImgFile(self.img_array[self.img_key])
        key += 1
        self.img_key = key
        if self.img_key > len(self.img_array)-1:
            self.img_key = 0

    def AddParameters(self):

        self.orgimg = self.img.copy()  # if self.img != None else None
        Height, Width, Channels = self.img.shape
        self.height = Height
        self.width = Width
        self.channels = Channels

    def AddCam(self, cameraNum=0, startSec=0):
        """
        Add a videostream to self.img
        """
        self.cap = cv2.VideoCapture(cameraNum)

        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_POS_MSEC, startSec*1000)    # start at 10sec
            # self.cap.set(cv2.CAP_PROP_FPS,10)              # doesn't work
            # self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 300)    # doesn't work
            # self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 300)     # doesn't work
            self.GetFrame()
            self.AddParameters()
        else:
            print("could not read camera")

    def SetAsOrgImg(self):
        """
        set the self.orgimg  = self.img
        in order to proceed with further processing. For instance tweaking with trackbars.
        """
        self.orgimg = self.img.copy()

    def PrintSize(self):
        # height, width, channels = self.GetSize()
        print(f"height:{self.height}, width:{self.width}, channels:{self.channels}")

    def GetSize(self):
        """
        returns 'height', 'width' and 'channels'
        """
        if len(self.img.shape) == 3:
            Height, Width, Channels = self.img.shape
        else:
            Height, Width = self.img.shape
            Channels = 1
        self.height = Height
        self.width = Width
        self.channels = Channels
        return Height, Width, Channels

    def imgResize(self, Height, Width="auto"):
        if Width == "auto":
            orgHeight, orgWidth, _ = self.GetSize()
            factor = orgHeight/Height
            Width = orgWidth/factor

        self.img = cv2.resize(self.img, (int(Width), (int(Height))))
        self.height = int(Height)
        self.width = int(Width)
        self.filter_list.append("resize")
        self.filter_dict["resize"] = [Height, Width]

    def ShowImg(self, waitFurtherProcessing=False):
        while (True):

            cv2.imshow(str(self.name), self.img)
            if waitFurtherProcessing:
                k = cv2.waitKey(1) & 0xFF
                if k == 27:
                    break
            else:
                break
        # cv2.destroyWindow(self.name)

    def GetFrame(self):
        """
        Grabs and shows the currentvideo frame,
        +resize to 500 height
        """
        # print ("loop")
        # self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)     # doesn't work
        # self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)    # doesn't work
        ret, frame = self.cap.read()

        if ret:
            self.img = frame
            self.imgResize(500)
            # self.ShowImg(False)
        # cv2.destroyWindow(self.name)

    def Terminate(self):
        cv2.destroyWindow(self.name)

    # Filters

    def AddFilterBGR2Gray(self):
        """
        Add an standard 'cv2.COLOR_BGR2GRAY' filter on self.img
        """
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        self.GetSize()
        self.filter_list.append("bgr2gray")
        self.filter_dict["bgr2gray"] = None

    def AddFilterGray2BGR(self):
        self.img = cv2.cvtColor(self.img, cv2.COLOR_GRAY2BGR)

    def AddFilterBGR2LAB(self):
        # https://www.learnopencv.com/color-spaces-in-opencv-cpp-python/
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2LAB)
        # cv2.imshow("lab",lab)
        # -----Splitting the LAB image to different channels-------------------
        l, a, b = cv2.split(self.img)
        # cv2.imshow('l_channel', l)
        # cv2.imshow('a_channel', a)
        # cv2.imshow('b_channel', b)

    def AddFilterBGR2HSV(self):
        # https://www.learnopencv.com/color-spaces-in-opencv-cpp-python/
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        # cv2.imshow("lab",lab)
        # -----Splitting the LAB image to different channels-------------------
        h, s, v = cv2.split(self.img)
        cv2.imshow('h_channel', h)
        cv2.imshow('s_channel', s)
        cv2.imshow('v_channel', v)

    def AddFilterBGR2HLS(self):
        # https://www.learnopencv.com/color-spaces-in-opencv-cpp-python/
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2HLS)
        # cv2.imshow("lab",lab)
        # -----Splitting the LAB image to different channels-------------------
        h, l, s = cv2.split(self.img)
        cv2.imshow('h_channel', h)
        cv2.imshow('l_channel', l)
        cv2.imshow('s_channel', s)

    def AddFilterBGR2LUV(self):
        # https://www.learnopencv.com/color-spaces-in-opencv-cpp-python/
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2Luv)
        # cv2.imshow("lab",lab)
        # -----Splitting the LAB image to different channels-------------------
        l, u, v = cv2.split(self.img)
        cv2.imshow('l_channel', l)
        cv2.imshow('u_channel', u)
        cv2.imshow('v_channel', v)

    def AddFilterThreshold_binary(self, split):
        """
        Add an standard threshold 'cv2.THRESH_BINARY' filter on self.img
        """
        _, self.img = cv2.threshold(self.orgimg, split, 255, cv2.THRESH_BINARY)
        self.GetSize()
        self.filter_list.append("threshold_binary")
        self.filter_dict["threshold_binary"] = split

    def AddFilterThreshold_adaptive_gaussiany(self, blockSize=11, C=2):
        """
        Add an addaptive threshold 'cv.ADAPTIVE_THRESH_GAUSSIAN_C' filter on self.img
        """
        self.img = cv2.adaptiveThreshold(self.img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blockSize, C)
        # self.img = cv2.adaptiveThreshold(self.img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blockSize, C)
        self.GetSize()
        self.filter_list.append("threshold_adaptive_gaussiany")
        self.filter_dict["threshold_adaptive_gaussiany"] = [blockSize, C]

    def AddFilterBlurAvg(self, kernelSize=5):
        """
        Homogeneous blurring. Every pixel in the kernel has the same weight.
        """
        self.img = cv2.blur(self.img, (kernelSize, kernelSize))

    def AddFilterBlurGaussian(self, kernelSize=5):
        """
        weighted average blurring. Where closeby pixels weigh more.
        """
        self.img = cv2.GaussianBlur(self.img, (kernelSize, kernelSize), 0)

    def AddFilterBlurMedian(self, kernelSize=5):
        """
        takes median of all the pixels under kernel area and central element is replaced with this median value.
        This is highly effective against salt-and-pepper noise in the images.
        Interesting thing is that, in the AVG or Gaussian filters, central element is a newly calculated value
        which may be a pixel value in the image or a new value. But in median blurring, central element is always
        replaced by some pixel value in the image. It reduces the noise effectively.

        Its kernel size should be a positive odd integer.
        """
        self.img = cv2.medianBlur(self.img, kernelSize)

    def AddFilterBlurBilateral(self, Diameter=5, Multiplier=4):
        """
        highly effective in noise removal while keeping edges sharp. But slower.
        Diameter<=5 for realtime
        """
        self.img = cv2.bilateralFilter(self.img, Diameter, Diameter*Multiplier, Diameter*Multiplier)

    def AddFilterCanny(self, min, max):
        # if "canny_max" in self.trackbar_dict:
        #    max = self.trackbar_dict["canny_max"]
        #    min = self.trackbar_dict["canny_min"]

        self.img = cv2.Canny(self.orgimg, min, max)
        # self.GetSize()
        if "canny" not in self.filter_list:
            self.filter_list.append("canny")
            self.filter_dict["canny"] = [min, max]
        # @2Do if min or max is different then the stored min or max. change the value in the filter_dict

    def AddContrast(self):
        """
        Add contrast by convert the BGR tot LAB. apply CLAHE on the L channel and merg the CLAHE, A & B back together.
        CLAHE: Contrast Limited Adaptive Histogram Equalization

        input:
        - BGR image

        output:
        - BGR image
        """
        # https://www.learnopencv.com/color-spaces-in-opencv-cpp-python/
        self.GetSize()
        if self.channels == 1:
            print("input for AddContrast function needs to be BGR")
            return

        # https://stackoverflow.com/questions/39308030/how-do-i-increase-the-contrast-of-an-image-in-python-opencv
        # -----Converting image to LAB Color model-----------------------------
        lab = cv2.cvtColor(self.img, cv2.COLOR_BGR2LAB)
        # cv2.imshow("lab",lab)
        # -----Splitting the LAB image to different channels-------------------
        l, a, b = cv2.split(lab)
        # cv2.imshow('l_channel', l)
        # cv2.imshow('a_channel', a)
        # cv2.imshow('b_channel', b)
        # -----Applying CLAHE to L-channel-------------------------------------
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        # cv2.imshow('CLAHE output', cl)
        # -----Merge the CLAHE enhanced L-channel with the a and b channel-----
        limg = cv2.merge((cl, a, b))
        # cv2.imshow('limg', limg)
        # -----Converting image from LAB Color model to RGB model--------------
        final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        # cv2.imshow('final', final)
        self.img = final

    def AddDilation(self, kernelSize):
        """
        just opposite of erosion. Here, a pixel element is ‘1’ if atleast one pixel under the kernel is ‘1’.
        So it increases the white region in the image or size of foreground object increases.
        Normally, in cases like noise removal, erosion is followed by dilation.
        Because, erosion removes white noises, but it also shrinks our object. So we dilate it.
        Since noise is gone, they won’t come back, but our object area increases.
        It is also useful in joining broken parts of an object.
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernelSize, kernelSize))
        self.img = cv2.dilate(self.img, kernel)

    def AddErosion(self, kernelSize):
        """
        Erodes away the boundaries of foreground object (Always try to keep foreground in white)
        So what happends is that, all the pixels near boundary will be discarded depending upon the size of kernel.
        So the thickness or size of the foreground object decreases or simply white region decreases in the image.
        It is useful for removing small white noises (as we have seen in colorspace chapter), detach two connected
        objects etc.
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernelSize, kernelSize))
        self.img = cv2.erode(self.img, kernel)

    def AddOpening(self, kernelSize):
        """
        It is useful in removing noise, as we explained above. Here we use the function, cv2.morphologyEx()
        Opening is just another name of erosion followed by dilation.
        """
        # https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernelSize, kernelSize))
        self.img = cv2.morphologyEx(self.img, cv2.MORPH_OPEN, kernel)

    def AddClosing(self, kernelSize):
        """
        It is useful in closing small holes inside the foreground objects, or small black points on the object.
        Closing is reverse of Opening, Dilation followed by Erosion.
        """
        # https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernelSize, kernelSize))
        self.img = cv2.morphologyEx(self.img, cv2.MORPH_CLOSE, kernel)

    #
    # Processers

    def AddGoodFeatureToTrack(self, max_corners, quality, eucl_distance):
        """
        output:
        -Finds an adds corners to self.img.
        -Returns the x,y coordinates in a list.

        input:
        - self.img should be grayscaled before this feature
        - max_corners = maximum numbers of corners to find. more is better. it doesn't filter on the 'best' corners.
        - quality = specifies the quality of the corner. Float between 0 - 1.
        - eucl_distance = minimum eucledian distance between points.
        """
        if self.channels > 1:
            print(f"Input image({self.name} / {self.file}) is not an grayscale image, instead is has {self.channels} channels")
            return

        corners = cv2.goodFeaturesToTrack(self.img, max_corners, quality, eucl_distance)
        corners = np.int0(corners)
        corner_list = []
        for corner in corners:
            x, y = corner.ravel()
            cv2.circle(self.img, (x, y), 5, (0, 255, 0), 1)
            corner_list.append([x, y])

        self.GetSize()
        self.filter_list.append("good_feature_to_track")
        self.filter_dict["good_feature_to_track"] = [max_corners, quality, eucl_distance]
        return corner_list

    def AddHoughP(self, ImageToProjectTo, minLineLengthPerc, maxLineGapPerc):
        """
        minLineLengthPerc = minimum percentage of the width of the image
        maxLineGapPerc = percentage of the minLineLengthPx
        """
        # minLineLength = minimum number of pixels making up a line
        # maxLineGap = maximum gap in pixels between connectable line segments
        rho = 1  # distance resolution in pixels of the Hough grid
        theta = np.pi / 180  # angular resolution in radians of the Hough grid
        threshold = 100  # minimum number of votes (intersections in Hough grid cell)

        minLineLengthPx = int((self.width*minLineLengthPerc)/100)
        maxLineGapPx = int((minLineLengthPx*maxLineGapPerc)/100)
        print(f"minLineLenghtPx:{minLineLengthPx}, maxLineGapPx:{maxLineGapPx}")

        lines = cv2.HoughLinesP(self.img, rho, theta, threshold, minLineLength=minLineLengthPx, maxLineGap=maxLineGapPx)
        line_image = np.copy(ImageToProjectTo.img) * 0  # creating a blank to draw lines on
        # print (lines)
        # print (lines[0])

        if lines is not None:
            for line in lines:
                # x1 = line[0][0] / #y1 = line[0][1] / #x2 = line[0][2] / #y2 = line[0][3]
                x1, y1, x2, y2 = line[0]    # line[0].ravel()

                # 2Do: add slope detection. to get a good ratio it has to be based on the slope of the line
                LineScreenRatio = GetDistance(x1, y1, x2, y2) / self.width
                print(GetAngle(x1, y1, x2, y2), "degree")
                if LineScreenRatio > 0.95:
                    LineThickness = 1
                else:
                    LineThickness = 3
                # print ('x', x[0][0])
                # print (f"xx1:{x1}, x2:{x2}, y1:{y1}, y2:{y2}, ")
                cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), LineThickness)
            # Draw the lines on the  image
            ImageToProjectTo.img = cv2.addWeighted(ImageToProjectTo.img, 1, line_image, 0.8, 0)
        return lines

    def AddHough(self, ImageToProjectTo, Threshold):

        # https://stackoverflow.com/questions/45322630/how-to-detect-lines-in-opencv
        rho = 1  # distance resolution in pixels of the Hough grid
        theta = np.pi / 180  # angular resolution in radians of the Hough grid
        lines = cv2.HoughLines(self.img, rho, theta, Threshold)
        line_image = np.copy(ImageToProjectTo.img) * 0  # creating a blank to draw lines on
        # print (lines)
        if lines is not None:
            for line in lines:
                # for rho,theta in lines[0]:
                rho, theta = line[0]    # line.ravel()
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a*rho
                y0 = b*rho
                x1 = int(x0 + 1000*(-b))
                y1 = int(y0 + 1000*(a))
                x2 = int(x0 - 1000*(-b))
                y2 = int(y0 - 1000*(a))

                cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            # Draw the lines on the  image
            ImageToProjectTo.img = cv2.addWeighted(ImageToProjectTo.img, 1, line_image, 0.8, 0)
        return lines

    def AddContours(self, ImageToProjectFrom):
        """
        input image needs to be binary or grayscaled
        """
        # contour retrieval mode: https://docs.opencv.org/3.4.2/d9/d8b/tutorial_py_contours_hierarchy.html
        # Contour Approximation Method: https://docs.opencv.org/3.0.0/d4/d73/tutorial_py_contours_begin.html
        bin = ImageToProjectFrom.img.copy()  # findContours modifies the source image
        contours, hierarchy = cv2.findContours(bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]  # get largest five contour area
        rects = []

        """
        # draws rectangulair boundingbox
        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            x, y, w, h = cv2.boundingRect(approx)
            if h >= 15:
                # if height is enough
                # create rectangle for bounding
                rect = (x, y, w, h)
                rects.append(rect)
                cv2.rectangle(self.img, (x, y), (x+w, y+h), (255, 0, 0), 2)     #BLUE
        """

        # loop over the contours
        for c in contours:
            # approximate the contour
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.1 * peri, True)

            # if our approximated contour has four points, then we
            # can assume that we have found our screen
            area = cv2.contourArea(c)
            print(f"area: {area}px, points:{len(approx)}")

            if len(approx) > 2:
                thickness = 3
            else:
                thickness = 1
            cv2.drawContours(self.img, [approx], 0, (0, 255, 0), thickness)            # GREEN

            """
            if len(approx) == 4:
                screenCnt = approx
                print (screenCnt)
                break
            """
            # get convex hull
            hull = cv2.convexHull(c)
            if len(hull) > 2:
                thickness = 3
            else:
                thickness = 1
            # draw it in red color
            cv2.drawContours(self.img, [hull], -1, (0, 255, 255), thickness)           # YELLOW

            """
            # rotated angle (follows not the excact contours, but a rotated rectangulair bounding box)
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(self.img,[box],0,(0,0,255),2)                     #RED
            """
        return

    # Trackbar opperators

    def AddTrackbar(self, trackbar_name, trackbar_pointer, max, startValue=-1):
        """
        when startValue=-1, than the init value will be between min and max
        trackbar_pointer:
        - canny_max
        - canny_min
        - threshold_binary
        """
        double_pointer = ["canny_min", "canny_max"]
        if startValue == -1:
            # startValue will be in the middle
            startValue = int(max/2)
        # cv2.createTrackbar(trackbar_name, self.name, min, max, self.TrackbarCallback)
        cv2.createTrackbar(trackbar_name, self.name, startValue, max, lambda v,
                           x=trackbar_pointer: self.TrackbarCallback(v, x))

        # preset the trackbar-slider at the given startValue
        # cv2.setTrackbarPos(trackbar_name, self.name, startValue)

        # 2do check if not all the pointer (single or double) could be in the trackbar_dict
        # if trackbar_pointer in double_pointer:
        #    self.trackbar_dict[trackbar_pointer]=startValue             # min
        self.trackbar_dict[trackbar_pointer] = startValue  # standaard alle pointers in dict
        self.filter_list.append("trackbar")
        self.filter_dict["trackbar"] = [trackbar_pointer, trackbar_name, min, max]

    def TrackbarCallback(self, value, trackbar_pointer):

        self.trackbar_dict[trackbar_pointer] = value      # put value in dict

        if trackbar_pointer.lower() == "threshold_binary":
            #_, self.img = cv2.threshold(self.orgimg, value, 255, cv2.THRESH_BINARY)
            self.AddFilterThreshold_binary(self.trackbar_dict["threshold_binary"])
            # print ("didit", self.name)
        elif trackbar_pointer.lower() == "canny_max":
            # sets the canny_max value in the trackbar_dict
            # self.trackbar_dict["canny_max"]=value
            # retrieve the current min and max values and apply a canny filter via self.addFilterCanny()
            self.AddFilterCanny(self.trackbar_dict["canny_min"], self.trackbar_dict["canny_max"])
        elif trackbar_pointer.lower() == "canny_min":
            # sets the canny_min value in the trackbar_dict
            # self.trackbar_dict["canny_min"]=value
            # retrieve the current min and max values and apply a canny filter via self.addFilterCanny()
            self.AddFilterCanny(self.trackbar_dict["canny_min"], self.trackbar_dict["canny_max"])

        print(f"value: {value}, trackbar_pointer:{trackbar_pointer}")
        print(self.trackbar_dict)


# Functions

def GetDistance(x1, y1, x2, y2):
    """
    returns the distance of 2 2D points in px.
    """
    a = np.array((x1, y1))
    b = np.array((x2, y2))
    dist = np.linalg.norm(a-b)
    return dist


def GetAngle(x1, y1, x2, y2):
    angle = int(math.atan((y1-y2)/(x2-x1))*180/math.pi)
    return angle
