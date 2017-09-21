#pragma once

// include RS200 Librealsense header file
#include <librealsense/rs.hpp>

// include OpenCV header file
#include <opencv2/opencv.hpp>
#include "opencv2/features2d.hpp"
#include "opencv2/videoio.hpp"


using namespace std;
using namespace cv;


//Threshold Parameter
int threshold_value = 120;
int threshold_type = 3;;
int const max_value = 255;
int const max_type = 4;
int const max_BINARY_value = 255;


//Variable Control
//Crop Factor
int x_roi = 340;
int y_roi = 50;
int width_roi = 290;
int height_roi = 400;


//Enable Filter/Feature
bool en_gaussian = false;
Size en_gaussian_kernel = Size(3, 3);

bool en_sharpen = false;
double en_sharpen_alpha = 1.5; double en_sharpen_beta = -0.5; double en_sharpen_gamma = 0;

bool en_erosion = false;
bool en_dilation = false;
bool en_threshold = false;
bool en_contour = false;
bool en_blob = false;
bool step_look = false;
bool record = false;
bool playback = false;
bool run_cam = false;

// Setting for realsense device
const int frameHeight = 480;
const int frameWidth = 640;
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

int RS200_Initialize(rs::device * dev) {
	//Initialize Stream Config
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
	if (!dev->is_stream_enabled(rs::stream::color)) {
		cout << "[ ERROR ] Disabled color stream" << endl;
		return EXIT_FAILURE;
	}
	if (!dev->is_stream_enabled(rs::stream::depth)) {
		cout << "[ ERROR ] Disabled depth stream" << endl;
		return EXIT_FAILURE;
	}
	if (!dev->is_stream_enabled(rs::stream::infrared)) {
		cout << "[ ERROR ] Disabled infrared stream" << endl;
		return EXIT_FAILURE;
	}
	if (!dev->is_stream_enabled(rs::stream::infrared2)) {
		cout << "[ ERROR ] Disabled infrared2 stream" << endl;
		return EXIT_FAILURE;
	}
	return EXIT_SUCCESS;
}

Mat getDepthImage(rs::device * dev) {
	Mat ir1(Size(frameWidth, frameHeight), CV_8UC1, (void*)dev->get_frame_data(rs::stream::infrared), Mat::AUTO_STEP);
	Mat ir2(Size(frameWidth, frameHeight), CV_8UC1, (void*)dev->get_frame_data(rs::stream::infrared2), Mat::AUTO_STEP);
	// Creating OpenCV matrix from infrared image (Depth Image Generate)
	Ptr<StereoBM> sbm = StereoBM::create(16 * 4, 13);

	Mat depth(Size(frameWidth, frameHeight), CV_16UC1), frameDepth(Size(frameWidth, frameHeight), CV_8UC1);

	sbm->compute(ir1, ir2, depth);
	double minVal; double maxVal;

	minMaxLoc(depth, &minVal, &maxVal);

	//-- 4. Display it as a CV_8UC1 image
	depth.convertTo(frameDepth, CV_8UC1, 255 / (maxVal - minVal));

	return frameDepth;
}

