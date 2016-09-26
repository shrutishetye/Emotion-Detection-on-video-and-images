// Example : Support Vector Machine (SVM) learning
// usage: prog training_data_file testing_data_file
// For use with test / training datasets : speech_ex
// Author : Toby Breckon, toby.breckon@cranfield.ac.uk
// Version : 0.2
// Copyright (c) 2011 School of Engineering, Cranfield University
// License : LGPL - http://www.gnu.org/licenses/lgpl.html


#include "svmClassifier.h"

using namespace cv;
using namespace std; // OpenCV API is in the C++ "cv" namespace

/******************************************************************************/
// use SVM "grid search" for kernel parameters
#define USE_OPENCV_GRID_SEARCH_AUTOTRAIN 0 // set to 0 to set SVM parameters manually

/******************************************************************************/

#define NUMBER_OF_TRAINING_SAMPLES 327
#define ATTRIBUTES_PER_SAMPLE 154
#define NUMBER_OF_TESTING_SAMPLES 1
#define NUMBER_OF_CLASSES 2

// N.B. classes are spoken alphabetric letters A-Z labelled 1 -> 26



/******************************************************************************/


float SvmClassifier::predictionHappy(Mat mat)  {
    
	float result = svmHappy->predict(mat);
    return result;
    
}

float SvmClassifier::predictionSurprised(Mat mat)  {
    
    float result = svmSurprised->predict(mat);
    return result;
    
}

float SvmClassifier::predictionDisgust(Mat mat)  {
    
    float result = svmDisgust->predict(mat);
    return result;
    
}

float SvmClassifier::predictionAnger(Mat mat)  {
    
    float result = svmAnger->predict(mat);
    return result;
    
}

// loads the sample database from file (which is a CSV text file)
int SvmClassifier::read_data_from_csv(const char* filename, Mat data, Mat classes, int n_samples ) {
    
    float tmp;
    

    // if we can't read the input file then return 0
    
    FILE* f = fopen( filename, "r" );
    
    if( !f ) {
        
        printf("ERROR: cannot read file %s\n",  filename);
        return 0; // all not OK
        
    }
    
    // for each sample in the file
    for(int line = 0; line < n_samples; line++) {

        for(int attribute = 0; attribute < (ATTRIBUTES_PER_SAMPLE + 1); attribute++) {
            
            if (attribute < ATTRIBUTES_PER_SAMPLE) {
                
                fscanf(f, "%f,", &tmp);
                data.at<float>(line, attribute) = tmp;

            }
            
            else if (attribute == ATTRIBUTES_PER_SAMPLE) {
                
                int classtmp=0;
                fscanf(f, "%d,", &classtmp);
                std::cout <<"\n" << classtmp << " " << line << "\n";
                classes.at<float>(line, 0) = classtmp;
            }
        }
    }
    
    fclose(f);
    
    return 1; // all OK
}

int SvmClassifier::read_testdata_from_csv(const char* filename, Mat data, Mat classes, int n_samples ) {
    float tmp;
    
    FILE* f = fopen( filename, "r" );
    
    if ( !f ) {
        printf("ERROR: cannot read file %s\n",  filename);
        return 0; // all not OK
    }
    
    for(int line = 0; line < n_samples; line++) {
        
        for(int attribute = 0; attribute < ATTRIBUTES_PER_SAMPLE; attribute++) {
            
            if (attribute < ATTRIBUTES_PER_SAMPLE) {
                // first 617 elements (0-616) in each line are the attributes
                fscanf(f, "%f,", &tmp);
                data.at<float>(line, attribute) = tmp;
            }
        }
    }
    
    fclose(f);
    
    return 1; // all OK
}

/******************************************************************************/

int SvmClassifier::svmTrain(const char* trainfile, int classNumber) {
    printf ("OpenCV version %s (%d.%d.%d)\n", CV_VERSION, CV_MAJOR_VERSION, CV_MINOR_VERSION, CV_SUBMINOR_VERSION);
    
    // define training data storage matrices (one for attribute examples, one
    Mat training_data = Mat(NUMBER_OF_TRAINING_SAMPLES, ATTRIBUTES_PER_SAMPLE, CV_32FC1);
    Mat training_classifications = Mat(NUMBER_OF_TRAINING_SAMPLES, 1, CV_32FC1);

    //define testing data storage matrices
    Mat testing_data = Mat(NUMBER_OF_TESTING_SAMPLES, ATTRIBUTES_PER_SAMPLE, CV_32FC1);
    Mat testing_classifications = Mat(NUMBER_OF_TESTING_SAMPLES, 1, CV_32FC1);
    
    // load training and testing data sets
    if (read_data_from_csv(trainfile, training_data, training_classifications, NUMBER_OF_TRAINING_SAMPLES)) {
        
        // define the parameters for training the SVM (kernel + SVMtype type used for auto-training, other parameters for manual only)
        CvSVMParams params = CvSVMParams (CvSVM::C_SVC,   // Type of SVM, here N classes (see manual)
                                         CvSVM::LINEAR,  // kernel type (see manual)
                                         0.0,	// kernel parameter (degree) for poly kernel only
                                         0.0,	// kernel parameter (gamma) for poly/rbf kernel only
                                         0.0,	// kernel parameter (coef0) for poly/sigmoid kernel only
                                         10,	// SVM optimization parameter C
                                         0,	// SVM optimization parameter nu (not used for N classe SVM)
                                         0,	// SVM optimization parameter p (not used for N classe SVM)
                                         NULL,	// class wieghts (or priors)
                                         
                                         // Optional weights, assigned to particular classes.
                                         // They are multiplied by C and thus affect the misclassification
                                         // penalty for different classes. The larger weight, the larger penalty
                                         // on misclassification of data from the corresponding class.
                                         // termination criteria for learning algorithm
                                         cvTermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 1000, 0.000001)
                                         );
        
        // train SVM classifier (using training data)
        CvSVM* svm = new CvSVM;
        
        if (classNumber == 1) {
            svmHappy = svm;
        }
        else if (classNumber == 2) {
            svmSurprised = svm;
        }
        else if (classNumber == 3) {
            svmAnger = svm;
        }
        else if (classNumber == 4) {
            svmDisgust = svm;
        }
        
        printf( "\nTraining the SVM (in progress) ..... ");
        fflush(NULL);
        
        #if (USE_OPENCV_GRID_SEARCH_AUTOTRAIN)
        
        printf( "(SVM 'grid search' => may take some time!)");
        fflush(NULL);
        
        // train using auto training parameter grid search if it is available
        // (i.e. OpenCV 2.x) with 10 fold cross valdiation
        // N.B. this does not search kernel choice
        
        svm->train_auto(training_data, training_classifications, Mat(), Mat(), params, 10);
        params = svm->get_params();
        
        printf( "\nUsing optimal parameters degree %f, gamma %f, ceof0 %f\n\t C %f, nu %f, p %f\n Training ..",
        params.degree, params.gamma, params.coef0, params.C, params.nu, params.p);
        
#else
        // otherwise use regular training and use parameters manually specified above
        svm->train(training_data, training_classifications, Mat(), Mat(), params);
#endif
        printf( ".... Done\n");
        // get the number of support vectors used to define the SVM decision boundary

        printf("Number of support vectors for trained SVM = %i\n", svm->get_support_vector_count());
        return 0;

    }

    return -1;
}

int SvmClassifier::svmTrain() {

    const char* happyTrainfile="C:/happydata.csv";
    svmTrain(happyTrainfile, 1);
    
    const char* surprisedTrainfile="C:/surprisedata.csv";
    svmTrain(surprisedTrainfile, 2);
    
    const char* angerTrainingFile="C:/angerdata.csv";
    svmTrain(angerTrainingFile, 3);
    
    const char* disgustTrainingFile="C:/disgustdata.csv";
    svmTrain(disgustTrainingFile, 4);
    
    return 0;
    
}

/******************************************************************************/