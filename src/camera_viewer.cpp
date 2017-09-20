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


// include the librealsense C++ header file
#include <librealsense/rs.hpp>

#include "resource.h"
// include OpenCV header file
#include <opencv2/opencv.hpp>
#include "opencv2/features2d.hpp"
using namespace std;
using namespace cv;

int threshold_value = 120;
int threshold_type = 3;;
int const max_value = 255;
int const max_type = 4;
int const max_BINARY_value = 255;
RNG rng(12345);
int x_roi = 340;
int y_roi = 50;
int width_roi = 290;
int height_roi = 400;
bool step_look = false;
Mat processed;

Mat Dilation(Mat src, int dilation_size, int dilation_elem, void*);
Mat Erosion(Mat src, int erosion_size, int erosion_elem, void*);



int main() {
	// Setting for realsense device
	const int frameHeight = 480;
	const int frameWidth = 640;

	cout << "Initialization..." << endl;
	rs::context ctx;
	rs::device * dev = ctx.get_device(0);
	cout << "	Using device 0, an " <<  dev->get_name() <<endl;
	cout << "	Serial number: " << dev->get_serial() << endl;
	cout << "	Firmware version: " << dev->get_firmware_version() << endl;
	// Configure Infrared stream to run at VGA resolution at 30 frames per second
	//dev->enable_stream(rs::stream::infrared, 320, 240, rs::format::y8, 30);

	// We must also configure depth stream in order to IR stream run properly
	//dev->enable_stream(rs::stream::depth, 480, 360, rs::format::z16, 30);

		try {
			dev->enable_stream(rs::stream::color, frameWidth, frameHeight, rs::format::bgr8, 30);
			dev->enable_stream(rs::stream::depth, frameWidth, frameHeight, rs::format::z16, 30);
			dev->enable_stream(rs::stream::infrared2, frameWidth, frameHeight, rs::format::y8, 30);
			dev->enable_stream(rs::stream::infrared, frameWidth, frameHeight, rs::format::y8, 30);

			dev->start();

		}
		catch (const rs::error & e) {
			cout << e.get_failed_function().c_str() << e.get_failed_args().c_str() << endl << e.what() << endl;
			return EXIT_FAILURE;
		}
	// Start streaming

	// Camera warmup - Dropped frames to allow stabilization
	for (int i = 0; i < 30; i++)
		dev->wait_for_frames();

	// Creating OpenCV matrix from depth image
	namedWindow("Color Image", WINDOW_AUTOSIZE);

	Mat tmp;
	if (!dev->is_stream_enabled(rs::stream::color)) {
		cout << "[ ERROR ] Disabled color stream" << endl;
		return 1;
	}
	if (!dev->is_stream_enabled(rs::stream::depth)) {
		cout << "[ ERROR ] Disabled depth stream" << endl;
		return 1;
	}
	if (!dev->is_stream_enabled(rs::stream::infrared)) {
		cout << "[ ERROR ] Disabled infrared stream" << endl;
		return 1;
	}
	if (!dev->is_stream_enabled(rs::stream::infrared2)) {
		cout << "[ ERROR ] Disabled infrared2 stream" << endl;
		return 1;
	}
	/// Create Trackbar to choose type of Threshold



	for (;;) {
		dev->wait_for_frames();
		//Mat cam_depth(Size(frameWidth, frameHeight), CV_16UC1, (void*)dev->get_frame_data(rs::stream::depth), Mat::AUTO_STEP);
		//Mat depth;
		//Mat cam_frameDepth;
		//cam_depth.convertTo(cam_frameDepth, CV_8UC1);
		//equalizeHist(frameDepth, frameDepth);
		//applyColorMap(frameDepth, frameDepth, COLORMAP_JET);

		Mat ir1(Size(frameWidth, frameHeight), CV_8UC1, (void*)dev->get_frame_data(rs::stream::infrared), Mat::AUTO_STEP);
		Mat ir2(Size(frameWidth, frameHeight), CV_8UC1, (void*)dev->get_frame_data(rs::stream::infrared2), Mat::AUTO_STEP);

		Mat color(Size(frameWidth, frameHeight), CV_8UC3, (void*)dev->get_frame_data(rs::stream::color), Mat::AUTO_STEP);
		
		//Depth Compute
		Ptr<StereoBM> sbm = StereoBM::create(16*4, 13);

		Mat depth(Size(frameWidth,frameHeight),CV_16UC1),frameDepth(Size(frameWidth, frameHeight), CV_8UC1);
		
		sbm->compute(ir1, ir2, depth);
		double minVal; double maxVal;

		minMaxLoc(depth, &minVal, &maxVal);

		//-- 4. Display it as a CV_8UC1 image
		depth.convertTo(frameDepth, CV_8UC1, 255 / (maxVal - minVal));

		rectangle(color, cv::Point2f(x_roi, y_roi), cv::Point2f(width_roi + x_roi, height_roi + y_roi), cv::Scalar(255, 0, 0));

		rectangle(frameDepth, cv::Point2f(x_roi, y_roi), cv::Point2f(width_roi + x_roi, height_roi + y_roi), cv::Scalar(255, 0, 0));
		imshow("Raw Depth Image", frameDepth);
		//imshow("Cam Depth Image", cam_frameDepth);

		//-- CROP
		Rect roi(x_roi, y_roi, width_roi, height_roi);
		processed = frameDepth(roi);
		//Draw Crop Area
		
		imshow("Color Image", color);


		///////////////////////////////////////////
		////////      Pre Process here     ////////
		///////////////////////////////////////////

		for (int row = 0; row < processed.rows; row++) {
			for (int col = 0; col < processed.cols; col++) {
				//cout << "row :" <<row <<",col:"<<col <<",val:" << (int)ir.at<uchar>(row, col) << endl;
//				if ((int)depth.at<uchar>(row, col) < 0) {
					//processed.at<uchar>(row, col) = 255;
	//			}
			}
		}
		GaussianBlur(processed, processed, Size(3,3),5);
		addWeighted(processed, 1.5, processed, -0.5, 0, processed);
		Mat dilated;
		Mat eroded;
		dilated = Dilation(processed, 3, 1, "MORPH CROSS");
		eroded = Erosion(dilated,3,1,"MORPH CROSS");
		processed = dilated;
		
		imshow("Depth Image [Processed]", processed);

		/////////////////////CANNY EDGE DETECTOR/////////////
		Mat canny_output;
		vector<vector<Point> > contours;
		vector<Vec4i> hierarchy;
		/// Detect edges using canny
		Canny(processed, canny_output, 100, 100 * 2, 3);
		/// Find contours
		findContours(canny_output, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point(0, 0));

		/// Draw contours
		Mat drawing = Mat::zeros(canny_output.size(), CV_8UC3);
		cout << "Contours : " << contours.size() << endl;
		int counter = 0;
		for (int i = 0; i< contours.size(); i++)
		{
			approxPolyDP(Mat(contours[i]), contours[i], 3, true);
			double area = contourArea(contours[i]);
			if (area < 0) {
				continue;
			}
			cout << "#Contour : " << i << "	Area : " << area << endl;
			Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
			drawContours(drawing, contours, i, 124, 2, 8, hierarchy, 0, Point());
		}
		cout << "Area > 0 : " << counter << endl;
		
		/// Show in a window
		namedWindow("Contours", WINDOW_AUTOSIZE);
		imshow("Contours", drawing);
		
		//Awaiting key input to escape || Step Look
		if (waitKey(1) == 27) break;
		if (waitKey(1) == 's')step_look = !step_look;
		if (step_look)waitKey(0);

	}

	return 0;

}

/**  @function Erosion  */
Mat Erosion(Mat src,int erosion_size,int erosion_elem, void*)
{
	int erosion_type;
	if (erosion_elem == 0) { erosion_type = MORPH_RECT; }
	else if (erosion_elem == 1) { erosion_type = MORPH_CROSS; }
	else if (erosion_elem == 2) { erosion_type = MORPH_ELLIPSE; }
	Mat erosion_dst;
	Mat element = getStructuringElement(erosion_type,
		Size(2 * erosion_size + 1, 2 * erosion_size + 1),
		Point(erosion_size, erosion_size));
	/// Apply the erosion operation
	erode(src, erosion_dst, element);
	return erosion_dst;
}

/** @function Dilation */
Mat Dilation(Mat src,int dilation_size,int dilation_elem, void*)
{
	Mat dilation_dst;
	int dilation_type;
	if (dilation_elem == 0) { dilation_type = MORPH_RECT; }
	else if (dilation_elem == 1) { dilation_type = MORPH_CROSS; }
	else if (dilation_elem == 2) { dilation_type = MORPH_ELLIPSE; }

	Mat element = getStructuringElement(dilation_type,
		Size(2 * dilation_size + 1, 2 * dilation_size + 1),
		Point(dilation_size, dilation_size));
	/// Apply the dilation operation
	dilate(src, dilation_dst, element);
	return dilation_dst;
}