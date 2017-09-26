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

#include <algorithm>
#include <time.h>
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

VideoWriter rgbWrite; //("ColorVid.mkv", VideoWriter::fourcc('F', 'F', 'V', '1'), 30, Size(frameWidth, frameHeight), true);
VideoWriter depthWrite;//("depthVid.mkv", VideoWriter::fourcc('F', 'F', 'V', '1'), 30, Size(frameWidth, frameHeight), false);
VideoCapture rgbRead;
VideoCapture depthRead;
int main() {
	//Create GUI 
	clock_t this_time = clock();
	clock_t last_time = this_time;
	double time_counter = 0;
	char* record_label = "Record";
	char* playback_label = "Playback";
	char* r200_label = "R200 Connect";
	cv::Mat frame = cv::Mat(650, 1000, CV_8UC3);
	cv::namedWindow(MAIN_WINDOW_NAME);
	cvui::init(MAIN_WINDOW_NAME);

	cout << "Initialization..." << endl;
	rs::device * dev;
	//Click to read depth value
	cv::namedWindow("Raw Depth Image");
	setMouseCallback("Raw Depth Image", clickToRead, 0);
	//Start Update System (Loop)
	for (;;) {
		this_time = clock();
		time_counter += (double)(this_time - last_time);

		int hand_area = 0;
		//Wait for next frame ready!
		Mat color(Size(frameWidth, frameHeight), CV_8UC3,Mat::AUTO_STEP);

		Mat frameDepth(Size(frameWidth, frameHeight),CV_8UC1);
		//Get Image data from R200 Camera
		if (!playback && run_cam) {
			dev->wait_for_frames();
			color.data = (uchar*)dev->get_frame_data(rs::stream::color);
			frameDepth = getDepthImage(dev);
		}
		else 
		if(playback){
			if (rgbRead.isOpened()) {
				try {
					rgbRead.read(color);
					if (color.empty()) {
						cout << "End of video" << endl;
						playback = false;
						playback_label = "Playback";
						continue;
					}
				}
				catch (cv::Exception& e)
				{
					const char* err_msg = e.what();
					std::cout << "exception caught: " << err_msg << std::endl;
				}
			}
			else {
				cout << "RGB playback failed" <<endl;
				playback = false;
			}
			if (depthRead.isOpened()) {
				try {
					Mat tmp;
					depthRead.read(tmp);
					cvtColor(tmp, frameDepth, cv::COLOR_RGB2GRAY);
				}
				catch (cv::Exception& e)
				{
					const char* err_msg = e.what();
					std::cout << "exception caught: " << err_msg << std::endl;
				}
			}
			else {
				cout << "Depth playback failed"<<endl;
				playback = false;
			}
		}
		//-- CROP
		Rect roi(x_roi-32, y_roi-6, width_roi, height_roi);
		Rect roi_depth(x_roi, y_roi, width_roi, height_roi);

		Mat color_replace = color(roi).clone();
		Mat color_original_crop = color(roi).clone();
		processed = frameDepth(roi_depth).clone();
		Mat depth_original_crop = frameDepth(roi_depth).clone();
		raw_depth = frameDepth.clone();

		applyColorMap(frameDepth, frameDepth, COLORMAP_JET);

		//Draw Crop Area
		Mat color2 = color.clone();
		rectangle(color2, cv::Point2f(x_roi, y_roi), cv::Point2f(width_roi + x_roi, height_roi + y_roi), cv::Scalar(255, 0, 0));
		rectangle(frameDepth, cv::Point2f(x_roi, y_roi), cv::Point2f(width_roi + x_roi, height_roi + y_roi), cv::Scalar(255, 0, 0));



		///////////////////////////////////////////
		////////      Pre Process here     ////////
		///////////////////////////////////////////
		if (en_invert) {
			bitwise_not(frameDepth, frameDepth);
			bitwise_not(processed, processed);

		}

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
			processed = Dilation(processed, 3, 2, "MORPH CROSS");
		}

		//Erosion
		if (en_erosion) {
			processed = Erosion(processed, 3, 2, "MORPH CROSS");
		}

		
		//THRESHOLDING
		if (en_threshold) {
			threshold(processed, processed, 180, 254, CV_THRESH_BINARY);
		}

		//Dilation After Thrreshold
		if (en_dilation_af_threshold) {
			processed = Dilation(processed, 5, 2, "MORPH CROSS");
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
			findContours(canny_output, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point(0, 0));

			/// Draw contours
			drawing = Mat::zeros(canny_output.size(), CV_8UC3);
			//cout << "Contours : " << contours.size() << endl;
			int counter = 0;

			vector<Rect> boundRect(contours.size());
			double area;
			vector<Point2f>center(contours.size());
			vector<float>radius(contours.size());
			double min=1000, max=0;
			short num_max;
			for (int i = 0; i < contours.size(); i++) {
				area = contourArea(contours[i]);
				if (area > max) {
					max = area;
					num_max = i;
				}
				if (area < min) {
					min = area;
				}
			}
			//cout << "Max Area : " << max << "\t Num Max : " << num_max << endl;
			for (int i = 0; i < contours.size(); i++)
			{
				approxPolyDP(Mat(contours[i]), contours[i], 3, true);
				area = contourArea(contours[i]);
				if (i == num_max) {
					boundRect[i] = boundingRect(Mat(contours[i]));
					minEnclosingCircle((Mat)contours[i], center[i], radius[i]);
				}
				Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
				rectangle(drawing, boundRect[i].tl(), boundRect[i].br(), color, 2, 8, 0);
				drawContours(drawing, contours, i, 124, 2, 8, hierarchy, 0, Point());

				//drawContours(color_replace, contours, i, 124, 2, 8, hierarchy, 0, Point());

			}
		}

		//Blob Detector
		if (en_blob) {
			/// BLOB Detection
			BlobDetect(depth_original_crop, &drawing,180,250,10);
		}
		
		//Subtraction color image with threshold
		if (en_subtract) {
			if (en_threshold) {
				for (int y = 0; y < color_replace.cols; y++) {
					for (int x = 0; x < color_replace.rows; x++) {
						if (en_addition) {
							if (processed.at<uchar>(Point(y, x)) >= 100) {
								hand_area++;
							}
							else {
								color_replace.at<Vec3b>(Point(y, x)) = 0;
							}
						}else if (processed.at<uchar>(Point(y, x)) >= 100) {
							color_replace.at<Vec3b>(Point(y, x)) = 0;
							hand_area++;
						}
					}
				}
			}
		}

		//Micro Position Item Checking
		if (pos_item_check) {
			//Crop each item segmentation

			//Wheel counter
			Rect crop(0,180, 80, 140);
			Mat tmp = color_original_crop(crop).clone();
			Mat display = tmp.clone();

			Rect roi_pos(Rect(200,300, tmp.cols,tmp.rows));
			display.copyTo(frame(roi_pos));

			rectangle(color_replace, cv::Point2f(0, 180), cv::Point2f(80, 320), cv::Scalar(255, 100, 100));

			if (en_pos_item_train) {
				cvtColor(tmp, tmp, cv::COLOR_BGR2GRAY);
				cout <<"Blob Size ::" << BlobDetect(tmp, &tmp,100,255,12.5) << endl;
			}
			imshow("Object Segment Image",tmp);

		}
		
		//GUI Window
		frame = cv::Scalar(49, 52, 49);
		cvui::window(frame, 15, 10, 180, 600, "Controller");
		cvui::checkbox(frame, 30, 40, "Gaussian Blur", &en_gaussian);
		cvui::checkbox(frame, 30, 60, "Invert Depth", &en_invert);
		cvui::checkbox(frame, 30, 80, "Add Weighted", &en_sharpen);
		cvui::checkbox(frame, 30, 100, "Erosion", &en_erosion);
		cvui::checkbox(frame, 30, 120, "Dilation", &en_dilation);
		cvui::checkbox(frame, 30, 140, "Threshold", &en_threshold);
		cvui::checkbox(frame, 30, 160, "Contour", &en_contour);
		cvui::checkbox(frame, 30, 180, "Blob", &en_blob);
		cvui::checkbox(frame, 30, 300, "Step (Press key to show image)", &step_look);
		cvui::checkbox(frame, 30, 200, "Color - Contour", &en_subtract);
		cvui::checkbox(frame, 30, 220, "Invert(Color-Contour)", &en_addition);
		cvui::checkbox(frame, 30, 240, "Dilate Af/Thresh", &en_dilation_af_threshold);

		
		cvui::checkbox(frame, 30, 520, "Beep after hand press", &en_handpress);
		cvui::checkbox(frame, 30, 540, "Position Item Enable (Mode)", &pos_item_check);
		cvui::checkbox(frame, 30, 560, "Position Item Train", &en_pos_item_train);
		cvui::checkbox(frame, 30, 580, "Position Item Classify", &en_pos_item_class);
		
		
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

		if (cvui::button(frame, 50, 350, record_label)) {
			record = !record;
			if (record == true) {
				std::cout << "Record! : " << record << std::endl;
				rgbWrite.open("ColorVid.mkv", VideoWriter::fourcc('F', 'F', 'V', '1'), 30, Size(frameWidth, frameHeight), true);
				depthWrite.open("depthVid.mkv", VideoWriter::fourcc('F', 'F', 'V', '1'), 30, Size(frameWidth, frameHeight), false);
				if (rgbWrite.isOpened()) {
					cout << "ColorVid.mkv Ready!" << endl;
				}
				else {
					cout << "Cannot open ColorVid to rec." << endl;

				}
				if (depthWrite.isOpened()) {
					record_label = "Stop Record";
					cout << "depthVid.mkv Ready!" << endl;
				}
				else {
					cout << "Cannot open depthVid to rec." << endl;
				}

			}
			else {
				std::cout << "Stop Record! : " << record << std::endl;
				rgbWrite.release();
				depthWrite.release();
				record_label = "Record";
			}

		}

		if (record) {
			if (rgbWrite.isOpened()) {
				rgbWrite.write(color);
			}
			else {
				cout << "RGB Record failed " << endl;
			}
			if (depthWrite.isOpened()) {
				depthWrite.write(raw_depth);
			}
			else {
				cout << "Depth Record failed " << endl;
			}
		}
		
		if (cvui::button(frame, 50, 400, playback_label)) {
			playback = !playback;
			if (playback) {
				playback_label = "Stop Playback";
				cout << "Video Playback!" << endl;
				if (rgbRead.open("ColorVid.mkv")) {
					cout << "Playback ColorVid Ready" << endl;
				}

				if (depthRead.open("depthVid.mkv")) {
					cout << "Playback depthVid Ready" << endl;
				}
			}
			else {
				playback_label = "Playback";
				cout << "Playback stopped" << endl;
				rgbRead.release();
				depthRead.release();
			}
		}
		rs::log_to_console(rs::log_severity::warn);

		if (cvui::button(frame, 50, 450, r200_label)) {
			if (!run_cam) {
				static rs::context ctx;

				if (ctx.get_device_count() == 0) { cout << "No Camera Found" << endl; }
				else {
					dev = ctx.get_device(0);
					if (dev == NULL) {
						cout << "R200 Camera not found!" << endl;
					}
					else if (!RS200_Initialize(dev)) {
						cout << "	Using device 0, an " << dev->get_name() << endl;
						cout << "	Serial number: " << dev->get_serial() << endl;
						cout << "	Firmware version: " << dev->get_firmware_version() << endl;
						cout << "Initialize Complete" << endl;
						run_cam = true;
						r200_label = "R200 Disconnect";
					}
				}
			}
			else {
				r200_label = "R200 Connect";
				cout << "Camera stopped" << endl;
				dev->stop();
				run_cam = false;
			}
		}
		cvui::update();
		imshow("Raw Depth Image", frameDepth);
		imshow(MAIN_WINDOW_NAME, frame);

		if (en_handpress) {
			if (raw_depth.at<uchar>(Point(609, 430)) >= 167 && raw_depth.at<uchar>(Point(609, 430)) <= 169)
			{
				if (time_counter > (double)(BEEP_AF_SECONDS * CLOCKS_PER_SEC)) {
					cout << '\a';
					last_time = this_time;
				}
			}
		}


		//Awaiting key input to escape || Step Look
		if (waitKey(1) == 27) break;
		if (waitKey(1) == 's')step_look = !step_look;
		if (waitKey(1) == 'm')cout<<" Hand area : " << hand_area <<endl;

		if (step_look)waitKey(0);

	}

	return 0;

}