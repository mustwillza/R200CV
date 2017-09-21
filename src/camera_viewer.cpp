/*
R200 Camera Interface Converted to OpenCV with CV tools



#pragma comment declare for use console or native windows app
mainCRTStartup => calls main(), the entrypoint for console mode apps
wmainCRTStartup => calls wmain(), as above but the Unicode version
WinMainCRTStartup => calls WinMain(), the entrypoint for native Windows apps
wWinMainCRTStartup => calls wWinMain(), as above but the Unicode version
*/

//Using native Windows app with Unicode version
//#pragma comment(linker, "/SUBSYSTEM:CONSOLE /ENTRY:wWinMainCRTStartup")

#define MAIN_WINDOW_NAME "Assembly Project Monitoring System : FIBO HIT-UTAS SANWA"


// include the librealsense C++ header file
#include <librealsense/rs.hpp>

// include CV UI header file
#include "cvui.h"

//include main Resoruce
#include "resource.h"
//include Core function file
#include "MCore.h"
// include OpenCV header file
#include <opencv2/opencv.hpp>
#include "opencv2/features2d.hpp"
using namespace std;
using namespace cv;

RNG rng(12345);
Mat processed;
Mat raw_depth;

void clickToRead(int event, int x, int y, int, void*)
{
	if (event != CV_EVENT_LBUTTONDOWN)
		return;
	Point pt = Point(x, y);
	cout << "x=" << pt.x << "\t y=" << pt.y << "\t value=" << (int)raw_depth.at<uchar>(y, x) << "\n";
}



int main() {
	//Create GUI 
	cv::Mat frame = cv::Mat(650, 1000, CV_8UC3);
	cv::namedWindow(MAIN_WINDOW_NAME);
	cvui::init(MAIN_WINDOW_NAME);

	cout << "Initialization..." << endl;
	rs::context ctx;
	rs::device * dev = ctx.get_device(0);
	if (!RS200_Initialize(dev)) {
		cout << "	Using device 0, an " << dev->get_name() << endl;
		cout << "	Serial number: " << dev->get_serial() << endl;
		cout << "	Firmware version: " << dev->get_firmware_version() << endl;
		cout << "Initialize Complete" << endl;
	}

	//Click to read depth value
	cv::namedWindow("Raw Depth Image");
	setMouseCallback("Raw Depth Image", clickToRead, 0);

	//Start Update System (Loop)
	for (;;) {


		//Wait for next frame ready!
		dev->wait_for_frames();

		//Get Image data from R200 Camera
		Mat color(Size(frameWidth, frameHeight), CV_8UC3, (void*)dev->get_frame_data(rs::stream::color), Mat::AUTO_STEP);
		Mat frameDepth = getDepthImage(dev);

		//-- CROP
		Rect roi(x_roi, y_roi, width_roi, height_roi);
		Mat color_replace = color(roi).clone();
		processed = frameDepth(roi).clone();
		Mat preprocess = frameDepth(roi).clone();
		raw_depth = frameDepth.clone();

		applyColorMap(frameDepth, frameDepth, COLORMAP_JET);

		//Draw Crop Area
		rectangle(color, cv::Point2f(x_roi, y_roi), cv::Point2f(width_roi + x_roi, height_roi + y_roi), cv::Scalar(255, 0, 0));
		rectangle(frameDepth, cv::Point2f(x_roi, y_roi), cv::Point2f(width_roi + x_roi, height_roi + y_roi), cv::Scalar(255, 0, 0));
		imshow("Raw Depth Image", frameDepth);



		///////////////////////////////////////////
		////////      Pre Process here     ////////
		///////////////////////////////////////////

		//Gaussian Blur
		if (en_gaussian) {
			GaussianBlur(processed, processed,en_gaussian_kernel,1);
		}

		//Sharpen (Add Weighted)
		if (en_sharpen) {
			addWeighted(processed, en_sharpen_alpha, processed, en_sharpen_beta, en_sharpen_gamma, processed);
		}

		//Dilasion
		if (en_dilation) {
			processed = Erosion(processed, 3, 2, "MORPH CROSS");
		}

		//Erosion
		if (en_erosion) {
			processed = Dilation(processed, 3, 2, "MORPH CROSS");
		}
		
		//THRESHOLDING
		if (en_threshold) {
			threshold(processed, processed, 180, 250, CV_THRESH_BINARY);
		}
		//Temporary Drawing
		Mat drawing;
		cvtColor(processed, drawing, cv::COLOR_GRAY2BGR);

		///////////////////////////////////////////
		////////     Feature Extraction    ////////
		///////////////////////////////////////////

		//Contour
		if (en_contour) {
			/////////////////////CANNY EDGE DETECTOR/////////////
			Mat canny_output;
			vector<vector<Point> > contours;
			vector<Vec4i> hierarchy;
			/// Detect edges using canny
			Canny(processed.clone(), canny_output, 100, 100 * 2, 3);
			/// Find contours
			findContours(canny_output, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));

			/// Draw contours
			drawing = Mat::zeros(canny_output.size(), CV_8UC3);
			//cout << "Contours : " << contours.size() << endl;
			int counter = 0;
			for (int i = 0; i < contours.size(); i++)
			{
				approxPolyDP(Mat(contours[i]), contours[i], 3, true);
				double area = contourArea(contours[i]);
				if (area < 0) {
					continue;
				}
				Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
				drawContours(drawing, contours, i, 124, 2, 8, hierarchy, 0, Point());
			}
		}

		//Blob Detector
		if (en_blob) {
			/// BLOB Detection
			// Setup SimpleBlobDetector parameters.
			SimpleBlobDetector::Params params;

			// Change thresholds
			params.minThreshold = 180;
			params.maxThreshold = 250;

			// Filter by Area.
			params.filterByArea = true;
			params.minArea = 10;

			// Filter by Circularity
			params.filterByCircularity = true;
			params.minCircularity = 0.1;

			// Filter by Convexity
			params.filterByConvexity = true;
			params.minConvexity = 0.2;

			// Filter by Inertia
			params.filterByInertia = true;
			params.minInertiaRatio = 0.01;
			Ptr<SimpleBlobDetector> b_detector = SimpleBlobDetector::create(params);
			// Storage for blobs
			vector<KeyPoint> keypoints;

			// Detect blobs
			b_detector->detect(preprocess, keypoints);

			// Draw detected blobs as red circles.
			// DrawMatchesFlags::DRAW_RICH_KEYPOINTS flag ensures
			// the size of the circle corresponds to the size of blob

			Mat im_with_keypoints;
			drawKeypoints(drawing, keypoints, drawing, Scalar(255, 0, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
			cout << "////////////Blob Detection/////////////" << endl;
			for (std::vector<cv::KeyPoint>::iterator blobIterator = keypoints.begin(); blobIterator != keypoints.end(); blobIterator++) {
				std::cout << "size of blob is: " << blobIterator->size << std::endl;
				std::cout << "point is at: " << blobIterator->pt.x << " " << blobIterator->pt.y << std::endl;
			}
			cout << "////////End Blob Detection/////////////" << endl << endl;

		}
		


		//GUI Window
		frame = cv::Scalar(49, 52, 49);
		cvui::window(frame, 15, 10, 180, 600, "Controller");
		cvui::checkbox(frame, 30, 40, "Gaussian Blur", &en_gaussian);
		cvui::checkbox(frame, 30, 60, "Add Weighted", &en_sharpen);
		cvui::checkbox(frame, 30, 80, "Erosion", &en_erosion);
		cvui::checkbox(frame, 30, 100, "Dilation", &en_dilation);
		cvui::checkbox(frame, 30, 120, "Threshold", &en_threshold);
		cvui::checkbox(frame, 30, 140, "Contour", &en_contour);
		cvui::checkbox(frame, 30, 160, "Blob", &en_blob);
		cvui::checkbox(frame, 30, 300, "Step (Press key to show image)", &step_look);

		//Display Multi Image in single window
		try
		{
			Mat Color_display;
			resize(color_replace, Color_display, cv::Size(), 0.8, 0.8);
			Rect roi(Rect(200, 15, Color_display.cols, Color_display.rows));
			Color_display.copyTo(frame(roi));

			Mat processed_display;
			resize(processed, processed_display, cv::Size(), 0.8, 0.8);
			Rect roi2(Rect(200+Color_display.cols+10, 15, processed_display.cols, processed_display.rows));
			cvtColor(processed_display, processed_display, cv::COLOR_GRAY2BGR);
			processed_display.copyTo(frame(roi2));

			Mat drawing_display;
			resize(drawing, drawing_display, cv::Size(), 0.8, 0.8);
			Rect roi3(Rect(200 + Color_display.cols + processed_display.cols + 20, 15, processed_display.cols, processed_display.rows));
			drawing_display.copyTo(frame(roi3));
		}
		catch (cv::Exception& e)
		{
			const char* err_msg = e.what();
			std::cout << "exception caught: " << err_msg << std::endl;
		}


		imshow(MAIN_WINDOW_NAME, frame);
		cvui::update();

		//Awaiting key input to escape || Step Look
		if (waitKey(1) == 27) break;
		if (waitKey(1) == 's')step_look = !step_look;
		if (step_look)waitKey(0);

	}

	return 0;

}