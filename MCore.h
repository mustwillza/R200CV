#pragma once

// include RS200 Librealsense header file
#include <librealsense/rs.hpp>

// include OpenCV header file
#include <opencv2/opencv.hpp>
#include "opencv2/features2d.hpp"
#include "opencv2/videoio.hpp"


using namespace std;

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

const int BEEP_AF_SECONDS = 1;

//Enable Filter/Feature
bool en_gaussian = false;
cv::Size en_gaussian_kernel = cv::Size(3, 3);

bool en_sharpen = false;
double en_sharpen_alpha = 1.5; double en_sharpen_beta = -0.5; double en_sharpen_gamma = 0;

bool en_erosion = false;
bool en_dilation = false;
bool en_threshold = false;
bool en_contour = false;
bool en_blob = false;
bool step_look = false;
bool en_subtract = false;
bool en_invert = false;
bool en_addition = false;
bool record = false;
bool playback = false;
bool run_cam = false;
bool en_handpress = false;
bool pos_item_check = false;
bool en_pos_item_train = false;
bool en_pos_item_class = false;
bool en_dilation_af_threshold = false;
bool en_black_pixel_calc = false;
bool en_pos_hand = false;

// Setting for realsense device
const int frameHeight = 480;
const int frameWidth = 640;
/**  @function Erosion  */
cv::Mat Erosion(cv::Mat src,int erosion_size,int erosion_elem, void*)
{
	int erosion_type;
	if (erosion_elem == 0) { erosion_type = cv::MORPH_RECT; }
	else if (erosion_elem == 1) { erosion_type = cv::MORPH_CROSS; }
	else if (erosion_elem == 2) { erosion_type = cv::MORPH_ELLIPSE; }
	cv::Mat erosion_dst;
	cv::Mat element = getStructuringElement(erosion_type,
		cv::Size(2 * erosion_size + 1, 2 * erosion_size + 1),
		cv::Point(erosion_size, erosion_size));
	/// Apply the erosion operation
	erode(src, erosion_dst, element);
	return erosion_dst;
}

/** @function Dilation */
cv::Mat Dilation(cv::Mat src,int dilation_size,int dilation_elem, void*)
{
	cv::Mat dilation_dst;
	int dilation_type;
	if (dilation_elem == 0) { dilation_type = cv::MORPH_RECT; }
	else if (dilation_elem == 1) { dilation_type = cv::MORPH_CROSS; }
	else if (dilation_elem == 2) { dilation_type = cv::MORPH_ELLIPSE; }

	cv::Mat element = getStructuringElement(dilation_type,
		cv::Size(2 * dilation_size + 1, 2 * dilation_size + 1),
		cv::Point(dilation_size, dilation_size));
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

cv::Mat getDepthImage(rs::device * dev) {
	cv::Mat ir1(cv::Size(frameWidth, frameHeight),CV_8UC1, (void*)dev->get_frame_data(rs::stream::infrared), cv::Mat::AUTO_STEP);
	cv::Mat ir2(cv::Size(frameWidth, frameHeight),CV_8UC1, (void*)dev->get_frame_data(rs::stream::infrared2), cv::Mat::AUTO_STEP);
	//cv::Mat ir1(Size(frameWidth, frameHeight), CV_8UC1);
	//cv::Mat ir2(Size(frameWidth, frameHeight), CV_8UC1);

	// Creating OpenCV cv::Matrix from infrared image (Depth Image Generate)

	cv::Ptr<cv::StereoBM> sbm = cv::StereoBM::create(64, 13);
	cv::Mat depth(cv::Size(frameWidth, frameHeight), CV_16UC1), frameDepth(cv::Size(frameWidth, frameHeight), CV_8UC1);

	sbm->compute(ir1, ir2, depth);
	double minVal; double maxVal;

	minMaxLoc(depth, &minVal, &maxVal);

	//-- 4. Display it as a CV_8UC1 image
	depth.convertTo(frameDepth, CV_8UC1, 255 / (maxVal - minVal));

	return frameDepth;
}

int BlobDetect(cv::Mat depth_original_crop,cv::Mat* drawing,float min_thresh,float max_thresh,float min_area) {
	// Setup SimpleBlobDetector parameters.
	cv::SimpleBlobDetector::Params params;

	// Change thresholds
	params.minThreshold = min_thresh;
	params.maxThreshold = max_thresh;

	// Filter by Area.
	params.filterByArea = true;
	params.minArea = min_area;

	// Filter by Circularity
	params.filterByCircularity = true;
	params.minCircularity = (float)0.1;

	// Filter by Convexity
	params.filterByConvexity = true;
	params.minConvexity = (float)0.1;

	// Filter by Inertia
	params.filterByInertia = true;
	params.minInertiaRatio = (float)0.01;
	cv::Ptr<cv::SimpleBlobDetector> b_detector = cv::SimpleBlobDetector::create(params);
	// Storage for blobs
	std::vector<cv::KeyPoint> keypoints;

	// Detect blobs
	b_detector->detect(depth_original_crop, keypoints);

	// Draw detected blobs as red circles.
	// Drawcv::MatchesFlags::DRAW_RICH_KEYPOINTS flag ensures
	// the size of the circle corresponds to the size of blob

	cv::Mat im_with_keypoints;
	drawKeypoints(*drawing, keypoints, *drawing, cv::Scalar(255, 0, 255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	//cout << "////////////Blob Detection/////////////" << endl;
	short blob_counter = 0;
	for (std::vector<cv::KeyPoint>::iterator blobIterator = keypoints.begin(); blobIterator != keypoints.end(); blobIterator++) {
		//std::cout << "size of blob is: " << blobIterator->size << std::endl;
		//std::cout << "point is at: " << blobIterator->pt.x << " " << blobIterator->pt.y << std::endl;
		blob_counter++;
	}
	//cout << "////////End Blob Detection/////////////" << endl << endl;
	return blob_counter;
}

int depth_pixel_counter(cv::Mat tmp_d, int thres_min,int thres_max) {
	int px_counter = 0;
	for (int x = 0; x < tmp_d.rows; x++) {
		for (int y = 0; y < tmp_d.cols; y++) {
			if (tmp_d.at<uchar>(cv::Point(x, y)) > thres_min,tmp_d.at<uchar>(cv::Point(x,y)) < thres_max) {
				px_counter++;
			}
		}
	}
	return px_counter;
}

void tick_timer(void *Args) {

}