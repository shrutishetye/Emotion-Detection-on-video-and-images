// minimal.cpp: Display the landmarks of a face in an image.
//              This demonstrates stasm_search_single.
/*Header files*/
#include <stdio.h>
#include <stdlib.h>
#include "opencv/highgui.h"
#include "stasm_lib.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv/cv.h>
#include <sstream>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <float.h>
#include <limits.h>
#include <time.h>
#include <ctype.h>
#include <math.h>
#include <iterator>
#include "stasm_lib.h"
#include "svmClassifier.h"
/*Namespaces*/
using namespace cv; 
using namespace std;

/*Global variables*/
String face_cascade_name = "../data/haarcascade_frontalface_alt.xml"; //haar for stasm
String eyes_cascade_name = "../data/haarcascade_eye_tree_eyeglasses.xml";
CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;
String window_name = "Capture - Face detection";
int initCount = 0;
float ini_x = 0;
float ini_y = 0;
int pixelChangeValue = 5;
int overallFeatureChange = 5;
float initialFeatureVetor[154];
float currentFeatureVetor[154];
SvmClassifier svmclassifier;

///** Function Headers */
void detectAndDisplay( Mat frame );
void getfeaturesASM(const char* filename);


int main()
{
	/*Capture Video*/
	VideoCapture capture("../video/1.avi");
	//VideoCapture capture(0);
	svmclassifier.svmTrain();
	cv::Mat_<unsigned char> frame;

	/*1. Load the cascades*/
	if( !face_cascade.load( face_cascade_name ) ){
		printf("--(!)Error loading face cascade\n");
		return -1;
	}
	if( !eyes_cascade.load( eyes_cascade_name ) ){ 
		printf("--(!)Error loading eyes cascade\n"); 
		return -1; 
	}

	/*2. Read the video stream*/
	//capture.open(-1);
	if ( ! capture.isOpened() ){
		printf("--(!)Error opening video capture\n"); 
		return -1; 
	}

	int i=0;
	int counter =0;
	while (capture.grab())
	{
		if(counter % 30 ==0) {

			capture >> frame;
			if( frame.empty() )
			{
				printf(" --(!) No captured frame -- Break!");
				break;
			}

			char filename[80];
			sprintf(filename,"../data/test_%d.jpg",i);
			cv::imwrite(filename, frame);
			i++;

			/*Detect face and eyes*/
			detectAndDisplay( frame );
			//*Get the feature points by Active Shape Modelling*/
			getfeaturesASM(filename);

		}
		counter++;
		int c = waitKey(10);
		if( (char)c == 27 ) { break; } // escape
	}
}

void getfeaturesASM(const char* filename){

	bool neutral = false;
	float currentx = 0;
	float currenty = 0;
	int *pointer;
	/*float* initialFeatureVetor= (float*) malloc(sizeof(154));
	for (int i=0; i<154; i++) {initialFeatureVetor[i]= 0;}
	float* currentFeatureVetor= (float*) malloc(sizeof(154));
	for (int i=0; i<154; i++) {currentFeatureVetor[i]= 0;}*/
	//float initialFeatureVetor[154];
	//float currentFeatureVetor[154];
	

	int changed = 0;
	float landmarks[2 * stasm_NLANDMARKS];
	if (initCount == 6 || (ini_x == 0 && ini_y == 0))
		neutral = true;

	Mat_<unsigned char> frame=imread(filename,CV_LOAD_IMAGE_GRAYSCALE);
	int foundface;
	if (!stasm_search_single(&foundface, landmarks,
		(char*)frame.data, frame.cols, frame.rows,filename, "../data"))
	{
		printf("Error in stasm_search_single: %s\n", stasm_lasterr());
		exit(1);
	}
	if (!foundface)
		printf("No face found");
	else
	{ 
		/*memset(initialFeatureVetor, 0, sizeof(initialFeatureVetor));
		memset(currentFeatureVetor, 0, sizeof(currentFeatureVetor));*/

		int index = 0;
		std::cout<<sizeof(initialFeatureVetor);
		float flowPreviousXPoint = 0;
		float flowPreviousYPoint = 0;

		stasm_force_points_into_image(landmarks, frame.cols, frame.rows);
		for (int i = 0; i < stasm_NLANDMARKS; i++) {
			frame(cvRound(landmarks[i*2+1]), cvRound(landmarks[i*2])) = 255;
			if (neutral == true) {
				std::fill_n(initialFeatureVetor, 100, -1);
				initialFeatureVetor [index ++] = cvRound(landmarks[i*2+1]);
				initialFeatureVetor [index ++] = cvRound(landmarks[i*2]);

				ini_x += cvRound(landmarks[i*2+1]);
				ini_y += cvRound(landmarks[i*2]);


				if (flowPreviousXPoint == 0 && flowPreviousYPoint ==0) {
					flowPreviousXPoint = cvRound(landmarks[i*2+1]);
					flowPreviousYPoint = cvRound(landmarks[i*2]);
				}

			}
			else {
				bool neutral = false;
				int change = initialFeatureVetor [index] - cvRound(landmarks[i*2+1]);
				currentFeatureVetor[index ++] = change;

				change = initialFeatureVetor [index] - cvRound(landmarks[i*2]);
				currentFeatureVetor[index ++] = change;

				currentx += cvRound(landmarks[i*2+1]);
				currenty += cvRound(landmarks[i*2]);
				//cout<<"xInterim"<<xInterim;

				if (flowPreviousXPoint == 0 && flowPreviousYPoint ==0) {
					flowPreviousXPoint = cvRound(landmarks[i*2+1]);
					flowPreviousYPoint = cvRound(landmarks[i*2]);
				} else {
					line(frame, Point(flowPreviousYPoint, flowPreviousXPoint), Point(cvRound(landmarks[i*2]), cvRound(landmarks[i*2+1])), Scalar(0, 0, 255));

					flowPreviousXPoint = cvRound(landmarks[i*2+1]);
					flowPreviousYPoint = cvRound(landmarks[i*2]);
				}
			}
		}
		
		/*float temp[154];
		for (int idx = 0; idx <154; idx ++) {
			temp[idx] = currentFeatureVetor[idx];
			cout << "currentFeatureVetor[idx] "  << idx << " " << currentFeatureVetor[idx] << endl;
		}*/
		cv::Mat queryFeature = cv::Mat(1, 154, CV_32F, currentFeatureVetor);
		int j=0;
	
		float outputHappy = svmclassifier.predictionHappy(queryFeature);
		float outputSurprised = svmclassifier.predictionSurprised(queryFeature);
		float outputDisgust = svmclassifier.predictionDisgust(queryFeature);
		float outputAnger = svmclassifier.predictionAnger(queryFeature);


		ostringstream predictionInformation;
		predictionInformation <<  " PREDICTED : " ;
		if (outputHappy != 0)
		predictionInformation << " HAPPY ";
		if (outputSurprised != 0)
		predictionInformation << " SURPRISE ";
		if (outputDisgust != 0)
		predictionInformation << " DISGUST ";
		if (outputAnger != 0)
		predictionInformation << " ANGER ";

		cv::putText(frame, predictionInformation.str().c_str(), Point(150, 100), CV_FONT_NORMAL, 1, Scalar(255,0,0), 2);


		float diff = (fabs (ini_x - currentx) + fabs (ini_y - currenty));

		std::cout << "Initial x : " << ini_x << " , Initial y : " << ini_y << endl;
		std::cout << "currentx : " << currenty << " , currenty : " << currenty << endl;
		std::cout << "  = diff : " << diff << endl;
		//cout<<"changed"<<changed<<endl;
		ostringstream diffValue;
		diffValue <<  diff << " , " << changed;

	/*if (changed <overallFeatureChange) {
		if (changed !=0) {
		diffValue <<  "RESULT :: EMOTION";
		} else {
		diffValue <<  "RESULT :: NEUTRAL";
		}

		//putText(frame, diffValue.str().c_str(), Point(100,250), CV_FONT_NORMAL, 1, Scalar(255,0,0),1,1);
		//putText(frame, diffValue.str().c_str(), Point(100, 100), CV_FONT_HERSHEY_TRIPLEX, 0.5, Scalar(255,255,255), 2);
		//cout<<"landmarks"<<stasm_NLANDMARKS;
	
	}*/
}

	cv::imwrite("minimal.bmp", frame);

	cv::resize(frame, frame, Size(800, 800), 0, 0, INTER_CUBIC);

	cvNamedWindow( "Output", 1 );
	cv::imshow("Output", frame);
}

void detectAndDisplay( Mat frame )
{
	std::vector<Rect> faces;
	Mat frame_gray;

	cvtColor( frame, frame_gray, COLOR_BGR2GRAY );
	equalizeHist( frame_gray, frame_gray );

	//-- Detect faces
	face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CASCADE_SCALE_IMAGE, Size(30, 30) );

	for( size_t i = 0; i < faces.size(); i++ )
	{
		Point center( faces[i].x + faces[i].width/2, faces[i].y + faces[i].height/2 );
		ellipse( frame, center, Size( faces[i].width/2, faces[i].height/2), 0, 0, 360, Scalar( 255, 0, 255 ), 4, 8, 0 );

		Mat faceROI = frame_gray( faces[i] );
		std::vector<Rect> eyes;

		//-- In each face, detect eyes
		eyes_cascade.detectMultiScale( faceROI, eyes, 1.1, 2, 0 |CASCADE_SCALE_IMAGE, Size(30, 30) );

		for( size_t j = 0; j < eyes.size(); j++ )
		{
			Point eye_center( faces[i].x + eyes[j].x + eyes[j].width/2, faces[i].y + eyes[j].y + eyes[j].height/2 );
			int radius = cvRound( (eyes[j].width + eyes[j].height)*0.25 );
			circle( frame, eye_center, radius, Scalar( 255, 0, 0 ), 4, 8, 0 );
		}
	}
	//-- Show what you got
	cv::imshow( window_name, frame );

}