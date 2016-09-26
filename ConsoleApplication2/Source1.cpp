//#include <opencv2/core/core.hpp>
//#include <opencv2/highgui/highgui.hpp>
//#include <opencv/cv.h>
//#include <opencv2/objdetect/objdetect.hpp>
//#include <iostream>
//#include "stasm_lib.h"
//#include <opencv/dirent.h>
//#include <string.h>
//#include <fstream>
//#include <sstream>
//#include <opencv2/ml/ml.hpp>
////
//using namespace cv;
//using namespace std;
//
//int initCount = 0;
//
//float xInitial = 0;
//float yInitial = 0;
//int pixelChangeValue = 5;
//int overallFeatureChange = 5;
//float initialFeatureVetor[154];
//float currentFeatureVetor[154];
//
////
/////** Global variables */
//String face_cascade_name = "../data/haarcascades/haarcascade_frontalface_alt.xml";
//String eyes_cascade_name  = "../data/haarcascades/haarcascade_eye.xml";
//CascadeClassifier face_cascade;
//CascadeClassifier eyes_cascade;
//String window_name = "Capture - Face detection";
////
/////** Function Headers */
//void detectAndDisplay( Mat frame );
//void ASMpoints(const char* path);
//void readTrainingData();
//void svmTraining();
//
////** @function main */
//int main(){
//
//	//VideoCapture capture("1.avi");
//	VideoCapture capture(0);
//
//	cv::Mat_<unsigned char> frame;
//	//IplImage *frame, *frame_copy = 0;
//	
//	//-- 1. Load the cascades
//	if( !face_cascade.load( face_cascade_name ) ){
//		printf("--(!)Error loading face cascade\n");
//		return -1;
//	};
//	if( !eyes_cascade.load( eyes_cascade_name ) ){ printf("--(!)Error loading eyes cascade\n"); return -1; };
//
//	//-- 2. Read the video stream
//	//capture.open(-1);
//	//if ( ! capture.isOpened() ) { printf("--(!)Error opening video capture\n"); return -1; }
//	int i=0;
//	int counter =0;
//	//  readTrainingData();
//	while (  capture.grab() )
//	{
//		if(counter % 30 ==0) {
//
//			capture >> frame;
//			if( frame.empty() )
//			{
//				printf(" --(!) No captured frame -- Break!");
//				break;
//			}
//			char filename[80];
//
//			sprintf(filename,"/Users/shobhikapanda/Documents/data/test_%d.jpg",i);
//			imwrite(filename, frame);
//			i++;
//
//			//-- 3. Apply the classifier to the frame
//			//readTrainingData();
//
//			//detect ("");
//
//			detectAndDisplay( frame );
//
//			ASMpoints(filename);
//			//svmTraining();
//
//			// cout<<"landmarks"<<stasm_NLANDMARKS;
//		}
//		counter++;
//		int c = waitKey(10);
//		if( (char)c == 27 ) { break; } // escape
//	}
//
//	// cv::Mat_<unsigned char> img=cv::imread(frame, CV_LOAD_IMAGE_GRAYSCALE);
//
//
//	return 0;
//}
//
//
//
//
//
//void ASMpoints(const char* filename){
//	bool storeInitial = false;
//
//	float xInterim = 0;
//	float yInterim = 0;
//
//	int changed = 0;
//	float landmarks[2 * stasm_NLANDMARKS];
//	if (initCount == 6 || (xInitial == 0 && yInitial == 0))
//		storeInitial = true;
//	// draw the landmarks on the image as white dots
//	// path = "/Users/shobhikapanda/Documents/data/testface.jpg";
//	Mat_<unsigned char> frame=imread(filename,CV_LOAD_IMAGE_GRAYSCALE);
//	int foundface;
//	String inputfileName="/Users/shobhikapanda/Downloads/inputimages/inputFile_landmarks1.txt";
//	//sprintf(filename,"/Users/shobhikapanda/Downloads/inputimages/inputFile_landmarks_%d.txt",i);
//	std::ofstream outputfile;
//	outputfile.open(inputfileName);
//	if (!stasm_search_single(&foundface, landmarks,
//		(char*)frame.data, frame.cols, frame.rows,filename, "/Users/shobhikapanda/Documents/data"))
//	{
//		printf("Error in stasm_search_single: %s\n", stasm_lasterr());
//		exit(1);
//	}
//	if (!foundface)
//		printf("No face found in");
//	else
//	{
//		int index = 0;
//		// draw the landmarks on the image as white dots (image is monochrome)
//		stasm_force_points_into_image(landmarks, frame.cols, frame.rows);
//		for (int i = 0; i < stasm_NLANDMARKS; i++) {
//			frame(cvRound(landmarks[i*2+1]), cvRound(landmarks[i*2])) = 255;
//
//			if (storeInitial == true) {
//				initialFeatureVetor [index ++] = cvRound(landmarks[i*2+1]);
//				initialFeatureVetor [index ++] = cvRound(landmarks[i*2]);
//
//				xInitial += cvRound(landmarks[i*2+1]);
//				yInitial += cvRound(landmarks[i*2]);
//			}
//			else {
//				bool set = false;
//				int delta1 = initialFeatureVetor [index] - cvRound(landmarks[i*2+1]);
//				currentFeatureVetor[index ++] = delta1;
//
//				if (delta1 > pixelChangeValue) {
//					set = true;
//					changed ++;
//				}
//
//				delta1 = initialFeatureVetor [index] - cvRound(landmarks[i*2]);
//				currentFeatureVetor[index ++] = delta1;
//
//				if (set == false && delta1 > pixelChangeValue) {
//					changed ++;
//				}
//
//				xInterim += cvRound(landmarks[i*2+1]);
//				yInterim += cvRound(landmarks[i*2]);
//			}
//		}
//
//		float diff = (fabs (xInitial - xInterim) + fabs (yInitial - yInterim));
//
//		std::cout << "xInitial : " << xInitial << " , yInitial : " << yInitial << endl;
//		std::cout << "xInterim : " << xInterim << " , yInterim : " << yInterim << endl;
//		std::cout << "  = diff : " << diff << endl;
//		cout<<"changed"<<changed<<endl;
//		ostringstream diffValue;
//		diffValue <<  diff << " , " << changed;
//
//		if (changed > overallFeatureChange) {
//			diffValue <<  "RESULT :: EMOTION";
//		} else {
//			diffValue <<  "RESULT :: NEUTRAL";
//		}
//
//		putText(frame, diffValue.str().c_str(), Point(100,250), CV_FONT_NORMAL, 1, Scalar(255,255,255),1,1);
//		//putText(frame, diffValue.str().c_str(), Point(100, 100), CV_FONT_HERSHEY_TRIPLEX, 0.5, Scalar(255,255,255), 2);
//
//		//cout<<"landmarks"<<stasm_NLANDMARKS;
//	}
//
//	imshow( "window", frame);
//}
//void svmTraining(){
//	int width = 512, height = 512;
//	Mat image = Mat::zeros(height, width, CV_8UC3);
//	CvMLData mlTrainData;
//	mlTrainData.read_csv("/Users/shobhikapanda/Downloads/77trainfeat.csv");
//	CvMLData mlLabelsData;
//	mlLabelsData.read_csv("/Users/shobhikapanda/Downloads/77traindataclass.csv");
//	const CvMat* tmp1 = mlTrainData.get_values();
//	const CvMat* tmp2 = mlLabelsData.get_values();
//
//
//	cout<<"temp1"<<mlTrainData.get_values();
//	//setLabel(tmp2,1);
//	cout<<"temp2"<<mlLabelsData.get_values();
//	// float labels[4] = {1.0, -1.0, -1.0, -1.0};
//	//Mat labelsMat(4, 1, CV_32FC1, labels);
//
//	// float trainingData[4][2] = { {501, 10}, {255, 10}, {501, 255}, {10, 501} };
//	//Mat trainingDataMat(4, 2, CV_32FC1, trainingData);
//
//	// Set up SVM's parameters
//	CvSVMParams params;
//	params.svm_type    = CvSVM::C_SVC;
//	params.kernel_type = CvSVM::LINEAR;
//	params.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);
//
//	// Train the SVM
//	CvSVM SVM;
//	SVM.train(tmp1, tmp2, Mat(), Mat(), params);
//
//	Vec3b green(0,255,0), blue (255,0,0);
//	for (int i = 0; i < image.rows; ++i)
//		for (int j = 0; j < image.cols; ++j)
//		{
//			Mat sampleMat = (Mat_<float>(1,2) << j,i);
//			float response = SVM.predict(sampleMat);
//
//			if (response == 1)
//				image.at<Vec3b>(i,j)  = green;
//			else if (response == -1)
//				image.at<Vec3b>(i,j)  = blue;
//		}
//
//		// Show the training data
//		int thickness = -1;
//		int lineType = 8;
//		circle( image, Point(501,  10), 5, Scalar(  0,   0,   0), thickness, lineType);
//		circle( image, Point(255,  10), 5, Scalar(255, 255, 255), thickness, lineType);
//		circle( image, Point(501, 255), 5, Scalar(255, 255, 255), thickness, lineType);
//		circle( image, Point( 10, 501), 5, Scalar(255, 255, 255), thickness, lineType);
//
//		// Show support vectors
//		thickness = 2;
//		lineType  = 8;
//		int c = SVM.get_support_vector_count();
//
//		for (int i = 0; i < c; ++i)
//		{
//			const float* v = SVM.get_support_vector(i);
//			circle( image,  Point( (int) v[0], (int) v[1]),   6,  Scalar(128, 128, 128), thickness, lineType);
//		}
//
//		imwrite("result.png", image);        // save the image
//
//		imshow("SVM Simple Example", image); // show it to the user
//		waitKey(0);
//}
//void readTrainingData(){
//
//	String filename="/Users/shobhikapanda/Downloads/outputimages/";
//
//	struct dirent *ent;
//	DIR *dir;
//	dir = opendir(filename.c_str());
//
//	while ((ent = readdir (dir)) != NULL) {
//		String fname (ent->d_name);
//		if(fname.at(0)!='.'){
//			//cout<<"--->"<<ent->d_name;
//			string imgPath(filename + ent->d_name);
//			// cout<<imgPath;
//			//char const * fpath;
//			Mat_<unsigned char> frame=imread(imgPath,CV_LOAD_IMAGE_GRAYSCALE);
//			char *fpath = const_cast<char*>(imgPath.c_str());
//			char * pch = strtok (ent->d_name,".");
//			cout<<pch;
//
//			string str=pch;
//			str=str.append("_landmarks.txt");
//			cout<<"Hello   "<<str;
//			std::ofstream outputfile;
//			outputfile.open("/Users/shobhikapanda/Downloads/trainData/"+str);
//			float landmarks[2 * stasm_NLANDMARKS];
//			int foundface;
//			if (!stasm_search_single(&foundface, landmarks,
//				(char*)frame.data, frame.cols, frame.rows,fpath, "/Users/shobhikapanda/Documents/data"))
//			{
//				printf("Error in stasm_search_single: %s\n", stasm_lasterr());
//				exit(1);
//			}
//			if (!foundface)
//				printf("No face found in");
//			else
//			{
//				// draw the landmarks on the image as white dots
//				stasm_force_points_into_image(landmarks, frame.cols, frame.rows);
//				//cvtColor( frame, frame, COLOR_BGR2GRAY );
//				for (int i = 0; i < stasm_NLANDMARKS; i++){
//					//cout<<"landmarks"<<landmarks[i]<<" ";
//					//cout << landmarks[i*2+1] << "," << landmarks[i*2] << endl;
//
//
//
//					//fout << format(landmarks[i*2+1]+" "+landmarks[i*2],Formatter::FMT_CSV);
//					outputfile << landmarks[i*2+1]<<" "<<landmarks[i*2]<< endl;
//					frame(cvRound(landmarks[i*2+1]), cvRound(landmarks[i*2])) = 255;
//				}
//				//cout<<"landmarks"<<stasm_NLANDMARKS;
//			}
//
//			//const char* fpath=filename + ent->d_name;
//			// cout<<fpath;
//		}
//		//cout<<imgPath;
//	}
//	// string filePath = @"C:\Files\";
//
//	//Mat img = imread(imgPath);
//	//cvtColor(img,img,CV_BGR2GRAY);
//
//}
//void detectAndDisplay( Mat frame )
//{
//	std::vector<Rect> faces;
//	Mat frame_gray;
//
//	cvtColor( frame, frame, COLOR_BGR2GRAY );
//	equalizeHist( frame, frame );
//
//	//-- Detect faces
//	face_cascade.detectMultiScale( frame, faces, 1.1, 2, 0|CASCADE_SCALE_IMAGE, Size(30, 30) );
//
//	for( size_t i = 0; i < faces.size(); i++ )
//	{
//		Point center( faces[i].x + faces[i].width/2, faces[i].y + faces[i].height/2 );
//		ellipse( frame, center, Size( faces[i].width/2, faces[i].height/2), 0, 0, 360, Scalar( 255, 0, 255 ), 4, 8, 0 );
//
//		Mat faceROI = frame( faces[i] );
//		std::vector<Rect> eyes;
//
//		//-- In each face, detect eyes
//		eyes_cascade.detectMultiScale( faceROI, eyes, 1.1, 2, 0 |CASCADE_SCALE_IMAGE, Size(30, 30) );
//
//		for( size_t j = 0; j < eyes.size(); j++ )
//		{
//			Point eye_center( faces[i].x + eyes[j].x + eyes[j].width/2, faces[i].y + eyes[j].y + eyes[j].height/2 );
//			int radius = cvRound( (eyes[j].width + eyes[j].height)*0.25 );
//			circle( frame, eye_center, radius, Scalar( 255, 0, 0 ), 4, 8, 0 );
//		}
//	}
//	//-- Show what you got
//	imshow( "window_name", frame );
//
//}
