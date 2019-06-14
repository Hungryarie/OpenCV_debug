import cv2
import numpy as np
from openCV_debug import OCVWindow


def nothing(x):
    # print(f"val:{x}")
    pass


def main():
    nw_image = OCVWindow("input image")
    #nw_image.AddCam("movies/test.mp4", 10)
    #nw_image.AddImgFile("IMG_3832_small.jpg")
    #nw_image.imgResize(500)
    nw_image.AddImgBatch("IMG_3832.JPG")
    nw_image.AddImgBatch("IMG_3833.JPG")
    nw_image.AddImgBatch("IMG_3834_zoom.JPG")
    nw_image.CycleImg(delay=0)
    

    nw_gray = OCVWindow("nw_gray : gray")

    """
    nw_edges = OCVWindow("nw_edges : edges from gray")
    nw_edges.AddTrackbar("max", "canny_max", 600, 300)
    nw_edges.AddTrackbar("min", "canny_min", 255, 100)

    nw_2edges = OCVWindow("nw_2edges: edges from dilated")
    nw_2edges.AddTrackbar("max", "canny_max", 600, 300)
    nw_2edges.AddTrackbar("min", "canny_min", 255, 100)
    """
    
    #nw_hough = OCVWindow("nw_hough : hough")
    #nw_hough.AddTrackbar("minLength", "houghp-minlinelength", 100, 60)  # percentage of width
    #nw_hough.AddTrackbar("maxLineGap", "houghp-maxlinegap", 100, 6)    # percentage of minlinelenght
    #nw_hough.AddTrackbar("Threshold", "hough-threshold", 300, 120)

    # nw_contour = OCVWindow("nw_contour : contour")

    # nw_contrast = OCVWindow("nw_contrast : enhanched contrast")

    nw_test = OCVWindow("nw_test : ...")
    nw_test.AddTrackbar("treshold", "threshold_binary", 255, 210)

    nw_test2 = OCVWindow("nw_test2 : ...")
    nw_test2.AddTrackbar("treshold", "threshold_binary", 255, 210)


    while(True):
        # Get input frame from cam
        #nw_image.GetFrame()
        nw_image.CycleImg(delay=50)
        nw_image.imgResize(500)
        nw_image.ShowImg(waitFurtherProcessing=False)
        # Do stuff

        nw_gray.CopyFrom(nw_image)
        nw_gray.AddFilterBGR2Gray()
        nw_gray.ShowImg()

        nw_test.CopyFrom(nw_gray)
        nw_test.AddFilterBlurGaussian(5)
        #nw_test.AddContrast()
        nw_test.AddFilterThreshold_binary(nw_test.trackbar_dict["threshold_binary"])
        #nw_test.AddErosion(1)
        #nw_test.AddOpening(3)
        nw_test.ShowImg()

        nw_test2.CopyFrom(nw_test)
        nw_test2.AddFilterBlurGaussian(5)
        nw_test2.SetAsOrgImg()
        nw_test2.AddFilterThreshold_binary(nw_test2.trackbar_dict["threshold_binary"])
        #nw_test2.AddErosion(3)
        nw_test2.AddOpening(1)
        nw_test2.ShowImg()
    

        """
        nw_contrast.CopyFrom(nw_image)
        nw_contrast.AddContrast()
        nw_contrast.AddFilterBGR2Gray()
        nw_contrast.AddClosing(5)                   # to filter out small lines (text)
        nw_contrast.AddFilterBlurBilateral(5, 20)    # optional
        nw_contrast.AddFilterBlurGaussian(5)
        nw_contrast.ShowImg()
        """

        """
        # new namedWindow (copy frome gray)
        nw_edges.CopyFrom(nw_gray)
        nw_edges.AddFilterBlurGaussian(5)
        nw_edges.SetAsOrgImg()
        nw_edges.AddFilterCanny(nw_edges.trackbar_dict["canny_min"], nw_edges.trackbar_dict["canny_max"])
        nw_edges.ShowImg()

        # new namedWindow (copy frome gray)
        nw_2edges.CopyFrom(nw_gray)
        nw_2edges.AddFilterBlurGaussian(5)
        nw_2edges.AddClosing(5)
        nw_2edges.SetAsOrgImg()
        nw_2edges.AddFilterCanny(nw_2edges.trackbar_dict["canny_min"], nw_2edges.trackbar_dict["canny_max"])  # 200,50)
        nw_2edges.ShowImg()
        """

        """
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
        """


        #nw_contour.CopyFrom(nw_image)
        ## nw_contour.AddContours(nw_edges)
        #nw_contour.AddContours(nw_hough)
        #nw_contour.ShowImg()

        # end of Do stuff
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

    cv2.waitKey(0)
    cv2.destroyAllWindows


if __name__ == '__main__':
    main()
