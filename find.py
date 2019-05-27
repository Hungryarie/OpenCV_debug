import cv2
import numpy as np
from openCV_debug import OCVWindow


def nothing(x):
    # print(f"val:{x}")
    pass


def main():
    nw_image = OCVWindow("input image")
    nw_image.AddCam("movies/test.mp4", 10)
    # nw_image.AddImgFile("test (1).jpg")
    nw_image.imgResize(500)

    nw_gray = OCVWindow("nw_gray : gray")

    nw_edges = OCVWindow("nw_edges : edges from gray")
    nw_edges.AddTrackbar("max", "canny_max", 600, 30)
    nw_edges.AddTrackbar("min", "canny_min", 255, 20)

    nw_2edges = OCVWindow("nw_2edges: edges from dilated")
    nw_2edges.AddTrackbar("max", "canny_max", 600, 70)
    nw_2edges.AddTrackbar("min", "canny_min", 255, 45)

    nw_hough = OCVWindow("nw_hough : hough")
    nw_hough.AddTrackbar("minLength", "houghp-minlinelength", 100, 60)  # percentage of width
    nw_hough.AddTrackbar("maxLineGap", "houghp-maxlinegap", 100, 6)    # percentage of minlinelenght
    nw_hough.AddTrackbar("Threshold", "hough-threshold", 300, 120)

    nw_contour = OCVWindow("nw_contour : contour")

    nw_contrast = OCVWindow("nw_contrast : enhanched contrast")

    nw_test = OCVWindow("nw_test : ...")

    while(True):
        # Get input frame from cam
        #nw_image.GetFrame()
        nw_image.ShowImg(True)
        # Do stuff

        nw_gray.CopyFrom(nw_image)
        nw_gray.AddFilterBGR2Gray()
        nw_gray.ShowImg()

        nw_test.CopyFrom(nw_image)
        nw_test.AddFilterBGR2HSV()
        # nw_test.ShowImg()

        nw_contrast.CopyFrom(nw_image)
        nw_contrast.AddContrast()
        nw_contrast.AddFilterBGR2Gray()
        nw_contrast.AddClosing(5)                   # to filter out small lines (text)
        nw_contrast.AddFilterBlurBilateral(5, 20)    # optional
        nw_contrast.AddFilterBlurGaussian(5)
        nw_contrast.ShowImg()

        # new namedWindow (copy frome gray)
        nw_edges.CopyFrom(nw_gray)
        nw_edges.AddFilterBlurGaussian(5)
        nw_edges.SetAsOrgImg()
        nw_edges.AddFilterCanny(nw_edges.trackbar_dict["canny_min"], nw_edges.trackbar_dict["canny_max"])
        nw_edges.ShowImg()

        # new namedWindow (copy frome gray)
        nw_2edges.CopyFrom(nw_contrast)
        nw_2edges.SetAsOrgImg()
        nw_2edges.AddFilterCanny(nw_2edges.trackbar_dict["canny_min"], nw_2edges.trackbar_dict["canny_max"])  # 200,50)
        nw_2edges.ShowImg()

        nw_hough.CopyFrom(nw_2edges)
        nw_hough.AddFilterGray2BGR()
        nw_2edges.AddHoughP(nw_hough, nw_hough.trackbar_dict["houghp-minlinelength"],
                            nw_hough.trackbar_dict["houghp-maxlinegap"])    # green
        nw_2edges.AddHough(
            nw_hough, nw_hough.trackbar_dict["hough-threshold"])        # red
        nw_hough.ShowImg()

        nw_hough.CopyFrom(nw_image, True)
        nw_2edges.AddHoughP(nw_hough, nw_hough.trackbar_dict["houghp-minlinelength"],
                            nw_hough.trackbar_dict["houghp-maxlinegap"])    # green
        nw_2edges.AddHough(nw_hough, nw_hough.trackbar_dict["hough-threshold"])        # red
        nw_hough.AddFilterBGR2Gray()
        nw_hough.AddFilterThreshold_binary(50)

        nw_contour.CopyFrom(nw_image)
        # nw_contour.AddContours(nw_edges)
        nw_contour.AddContours(nw_hough)
        nw_contour.ShowImg()

        # end of Do stuff
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
    """
    #new namedWindow for input image
    nw_image = OCVWindow("image")
    #nw_image.AddImgFile("test (2).jpg")
    nw_image.CopyFrom(nw_cam)

    print ("original sizes:")
    nw_image.PrintSize()
    nw_image.imgResize(600)
    print ("new sizes:")
    nw_image.PrintSize()
    #nw_image.ShowImg()

    # new namedWindow (copy frome image)
    nw_gray= OCVWindow("gray")
    nw_gray.CopyFrom(nw_image)
    nw_gray.AddFilterBGR2Gray()
    #nw_gray.AddGoodFeatureToTrack(100,0.01,100)
    nw_gray.ShowImg()

    # this is to recognize white on white
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
    dilated = cv2.dilate(nw_gray.img, kernel)

    #nw_gray.img = dilated
    cv2.imshow("dilated", dilated)
    #cv2.waitKey(0)

    # new namedWindow (copy frome gray)
    nw_th = OCVWindow("threshold")
    nw_th.CopyFrom(nw_gray)
    nw_th.AddFilterThreshold_binary(100)
    nw_th.AddTrackbar("split","threshold_binary",0,255)
    nw_th.ShowImg()

    # new namedWindow (copy frome gray)
    nw_edges = OCVWindow("2edges")
    nw_edges.CopyFrom(nw_gray)
    nw_2edges.img = cl# dilated
    nw_edges.AddFilterBlurGaussian(5)
    nw_edges.SetAsOrgImg()
    nw_edges.AddFilterCanny(300,1)
    nw_edges.AddTrackbar("max","canny_max",1,600)
    nw_edges.AddTrackbar("min","canny_min",1,255)
    #nw_edges.AddGoodFeatureToTrack(100,0.01,100)
    nw_edges.ShowImg(True)

    nw_hough= OCVWindow("hough")
    nw_hough.CopyFrom(nw_image)
    nw_edges.AddHoughP(nw_hough, 100,10)
    nw_edges.AddHough(nw_hough, 110)
    nw_hough.ShowImg()

    nw_contour= OCVWindow("nw_contour : contour")
    nw_contour.CopyFrom(nw_image)
    nw_contour.AddContours(nw_edges)
    nw_contour.ShowImg()
    """

    cv2.waitKey(0)
    cv2.destroyAllWindows


if __name__ == '__main__':
    main()
