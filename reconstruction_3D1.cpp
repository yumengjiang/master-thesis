//this function is used to match feature points between two images
//first, check the labels of two points, choose the feature points with the same label
//afterwards, calculate the residuals, choose the point with the least residuals which is the feature point
//featureLast is the central point of the cone, labelflag is the label of the corresponding cone
//featureLast size: N X 3 currentimage size: M x 3
#include <iostream>
#include <string>
#include "math.h"
#include "limits.h"
#include "eigen3/Eigen/Core"
#include "eigen3/Eigen/Dense"
#include <opencv2/features2d.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/opencv.hpp>
#include "opencv2/xfeatures2d.hpp"
#include <opencv2/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"

using namespace std;

float threshold = 40;

float computeResiduals(cv::Point2f pt1, cv::Point2f pt2){
  return pow((pow ((pt1.x-pt2.x),2) + pow ((pt1.y-pt2.y),2)),0.5);
}

void matchFeatures(int imageId, vector<cv::KeyPoint> featureLast, 
                  vector<cv::KeyPoint> featureNext, vector<cv::DMatch> &matched){
  float res, minRes;
  int featureLastRow = featureLast.size();
  int featureNextRow = featureNext.size();
  int index;

  for(int i = 0; i < featureNextRow; i++){
    minRes = threshold;
    for(int j = 0; j < featureLastRow; j++){
      if(featureNext[i].class_id == featureLast[j].class_id){
        res = computeResiduals(featureNext[i].pt, featureLast[j].pt);
        if(res < minRes){
          minRes = res;
          index = j;
        }
      }
    }
    if(minRes < threshold){
      matched.push_back(cv::DMatch(index,i,imageId,minRes));
      cout << index << " " << i << " " << imageId << " " << minRes << endl;
    } 
  }
}

int main( int argc, char** argv )
{
  vector<cv::KeyPoint> featureLast, featureNext;

  ifstream fileLast ( "result/1.csv" );
  string line, x, y, label;  
  while (getline(fileLast, line)) {  
      stringstream liness(line);  
      getline(liness, x, ',');  
      getline(liness, y, ','); 
      getline(liness, label);
      featureLast.push_back(cv::KeyPoint(stof(x),stof(y),3,-1,0,0,stof(label)));
      cout << x << " " << y << " " << label << endl;
  } 

  ifstream fileNext ( "result/2.csv" ); 
  while (getline(fileNext, line)) {  
      stringstream liness(line);  
      getline(liness, x, ',');  
      getline(liness, y, ','); 
      getline(liness, label);
      featureNext.push_back(cv::KeyPoint(stof(x),stof(y),3,-1,0,0,stof(label)));
      cout << x << " " << y << " " << label << endl;
  }  

  vector<cv::DMatch> matched;
  matchFeatures(1, featureLast, featureNext, matched);
  cv::Mat imgLast = cv::imread("result/1.png");
  cv::Mat imgNext = cv::imread("result/2.png");
  cv::Mat outImg;
  cv::drawMatches(imgLast, featureLast, imgNext, featureNext, matched, outImg);
  cv::namedWindow("MatchSIFT", cv::WINDOW_NORMAL);
  cv::imshow("MatchSIFT",outImg);
  cv::waitKey(0);
}
