//#include "opencv/cv.h"
//#include <opencv2/core/core.hpp>
//#include <opencv2/highgui/highgui.hpp>
//
//#include <stdio.h>
//#include <sstream>
//#include <stdlib.h>
//#include <string.h>
//#include <assert.h>
//#include <math.h>
//#include <float.h>
//#include <limits.h>
//#include <time.h>
//#include <ctype.h>
//#include <math.h>
//#include <iterator>
//#include "stasm_lib.h"
//
//using namespace cv;
//using namespace std;
//
//// Create a new Haar classifier
//CascadeClassifier cascade = 0;
//
//// Function prototype for detecting and drawing an object from an image
//void detect_and_draw( IplImage* image );
//
//// Create a string that contains the cascade name
//const char* cascade_name = "../data/haarcascade_frontalface_alt2.xml";
//
//const char* path = "../out/temp/temp_image.jpeg";
//
//const char* initalImagePath = "../out/temp/inital_image.jpeg";
//
////onst char* datadir = "/Users/Heman/Documents/workstation/Developement_Studio/Xcode_Laboratory/EmotionRecognition/EmotionRecognition/EmotionRecognition/data/sample";
//
//int initCount = 0;
//
//float xInitial = 0;
//float yInitial = 0;
//
//float initialFeatureVetor[154];
//float currentFeatureVetor[154];
//
//void computeAsm () {
//    bool storeInitial = false;
//    
//    float xInterim = 0;
//    float yInterim = 0;
//    
//    int changed = 0;
//
//    cv::Mat_<unsigned char> img(cv::imread(path, CV_LOAD_IMAGE_GRAYSCALE));
//    
//    if (initCount == 6 || (xInitial == 0 && yInitial == 0))
//        storeInitial = true;
//    
//    if (!img.data)
//    {
//        printf("Cannot load %s\n", path);
//        exit(1);
//    }
//    
//    int foundface;
//    float landmarks[2 * stasm_NLANDMARKS]; // x,y coords (note the 2)
//    
//    if (!stasm_search_single(&foundface, landmarks,
//                             (const char*) img.data, img.cols, img.rows, path, "/data"))
//    {
//        printf("Error in stasm_search_single: %s\n", stasm_lasterr());
//        exit(1);
//    }
//    
//    if (!foundface) {
//        printf("No face found in %s\n", path);
//    }
//    else
//    {
//        int index = 0;
//        // draw the landmarks on the image as white dots (image is monochrome)
//        stasm_force_points_into_image(landmarks, img.cols, img.rows);
//        for (int i = 0; i < stasm_NLANDMARKS; i++) {
//            img(cvRound(landmarks[i*2+1]), cvRound(landmarks[i*2])) = 255;
//            
//            if (storeInitial == true) {
//                initialFeatureVetor [index ++] = cvRound(landmarks[i*2+1]);
//                initialFeatureVetor [index ++] = cvRound(landmarks[i*2]);
//                
//                xInitial += cvRound(landmarks[i*2+1]);
//                yInitial += cvRound(landmarks[i*2]);
//            }
//            else {
//                bool set = false;
//                int delta1 = initialFeatureVetor [index] - cvRound(landmarks[i*2+1]);
//                currentFeatureVetor[index ++] = delta1;
//                
//                if (delta1 > 10) {
//                    set = true;
//                    changed ++;
//                }
//                    
//                delta1 = initialFeatureVetor [index] - cvRound(landmarks[i*2]);
//                currentFeatureVetor[index ++] = delta1;
//                
//                if (set == false && delta1 > 10) {
//                    changed ++;
//                }
//                
//                xInterim += cvRound(landmarks[i*2+1]);
//                yInterim += cvRound(landmarks[i*2]);
//            }
//        }
//        
//        float diff = (fabs (xInitial - xInterim) + fabs (yInitial - yInterim));
//
//        std::cout << "xInitial : " << xInitial << " , yInitial : " << yInitial << endl;
//        std::cout << "xInterim : " << xInterim << " , yInterim : " << yInterim << endl;
//        std::cout << "  = diff : " << diff << endl;
//        
//        ostringstream diffValue;
//        diffValue <<  diff << " , " << changed;
//        if (changed > 50) {
//            diffValue <<  "  REACHING >> PEAK !";
//        } else {
//            diffValue <<  " REACHING >> NORMAL !";
//        }
//        
//        
//        putText(img, diffValue.str().c_str(), Point(100, 100), CV_FONT_HERSHEY_TRIPLEX, 0.5, Scalar(255,255,255), 2);
//        
//    }
//    cout << " here" << endl;
//    
//    cv::imwrite("minimal.bmp", img);
//    cv::imshow("stasm minimal", img);
//}
//
//bool detect(std::string filename) {
//    
//    bool result = true;
//    
//    // Load the [[HaarClassifierCascade]]
//    cascade.load( cascade_name );
//    
//    // Check whether the cascade has loaded successfully. Else report and error and quit
//    if( !cascade.load( cascade_name ))
//    {
//        fprintf( stderr, "ERROR: Could not load classifier cascade\n" );
//        return -1;
//    }
//    
//    // Create a new named window with title: result
//    cvNamedWindow( "result", 1 );
//
//    VideoCapture capture;
//    
//    // Load the image from that filename
//    if (filename.empty() && filename.length() == 0) {
//        capture.open( -1 );
//    } else {
//        capture.open( filename );
//    }
//
//    Mat frame;
//    
//    //-- 2. Read the video stream
//    if( capture.isOpened() )
//    {
//            for(;;)
//            {
//                long loop = 0;
//              
//                if (loop++ % 30 == 0) {
//                    initCount ++;
//                    
//                    capture >> frame;
//                    
//                    // draw inital image after 3 secs.
//                    if (initCount == 6) {
//                        
//                        // delete the old file.
//                        std::remove(initalImagePath);
//                        
//                        imwrite( initalImagePath, frame );
//                    }
//                    
//                    //-- 3. Apply the classifier to the frame
//                    if( !frame.empty() )
//                    {
//                        IplImage* image = cvCreateImage(cvSize(frame.cols, frame.rows), 8, 3);
//                        IplImage ipltemp = frame;
//                        cvCopy(&ipltemp, image);
//                    
//                        detect_and_draw(image);
//                    }
//                    else
//                    {
//                        printf(" --(!) No captured frame -- Break!"); break;
//                    }
//                
//                    int c = cvWaitKey(1);
//                
//                    if (char(c) == 27)
//                        break;
//                }
//            }
//    }
//    
//    // Destroy the window previously created with filename: "result"
//    cvDestroyWindow("result");
//    
//    return result;
//}
//
//
//// Function to detect and draw any faces that is present in an image
//void detect_and_draw( IplImage* img )
//{
//    int scale = 1;
//    
//    // Create a new image based on the input image
//    IplImage* temp = cvCreateImage ( cvSize(img -> width/scale,img->height/scale), 8, 3 );
//    
//    // Create two points to represent the face locations
//    CvPoint pt1, pt2;
//    
//    int i;
//    
//    CvMemStorage* storage = cvCreateMemStorage(0);
//    
//    // Clear the memory storage which was used before
//    cvClearMemStorage( storage );
//    
//    // Find whether the cascade is loaded, to find the faces. If yes, then:
//    if( cascade.load( cascade_name ) )
//    {
//        
//        // There can be more than one face in an image. So create a growable sequence of faces.
//        // Detect the objects and store them in the sequence
//        CvSeq* faces = cvHaarDetectObjects( img, cascade, storage, 1.1, 2, CV_HAAR_DO_CANNY_PRUNING, cvSize(40, 40) );
//        
//        // Loop the number of faces found.
//        for( i = 0; i < (faces ? faces->total : 0); i++ )
//        {
//            // Create a new rectangle for drawing the face
//            CvRect* r = (CvRect*) cvGetSeqElem( faces, i );
//            
//            // Find the dimensions of the face,and scale it if necessary
//            pt1.x = r->x*scale;
//            pt2.x = (r->x+r->width)*scale;
//            pt1.y = r->y*scale;
//            pt2.y = (r->y+r->height)*scale;
//            
//            // Draw the rectangle in the input image
//            cvRectangle( img, pt1, pt2, CV_RGB(255,0,0), 3, 4, 0 );
//        }
//    }
//    
//    // --- stasm code
//    Mat imageIplMat = img;
//
//    imwrite( path, imageIplMat );
//    
//    computeAsm ();
//}
//
//int main()
//{
//   // std::cout << "------ start ------" << endl;
//    
//    //detect ("");
//    
//    detect("\video\1.avi");
//    
//    return 0;
//}