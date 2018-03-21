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

float threshold = 50;

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
        res = computeResiduals(featureNext[i].pt, featureLast[j].pt);//check residuals, find the smallest one, save it
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

  featureLast.push_back(cv::KeyPoint(299,198,3,-1,0,0,1));
  featureLast.push_back(cv::KeyPoint(688,199,3,-1,0,0,1));
  featureLast.push_back(cv::KeyPoint(143,201,3,-1,0,0,1));
  featureLast.push_back(cv::KeyPoint(455,187,3,-1,0,0,1));
  featureLast.push_back(cv::KeyPoint(455,195,3,-1,0,0,2));
  featureLast.push_back(cv::KeyPoint(276,180,3,-1,0,0,2));
  featureLast.push_back(cv::KeyPoint(161,183,3,-1,0,0,2));
  featureLast.push_back(cv::KeyPoint(612,208,3,-1,0,0,2));
  featureLast.push_back(cv::KeyPoint(234,211,3,-1,0,0,3));
  featureLast.push_back(cv::KeyPoint(510,223,3,-1,0,0,3));

  featureNext.push_back(cv::KeyPoint(280,209,3,-1,0,0,1));
  featureNext.push_back(cv::KeyPoint(624,207,3,-1,0,0,1));
  featureNext.push_back(cv::KeyPoint(457,194,3,-1,0,0,1));
  featureNext.push_back(cv::KeyPoint(113,214,3,-1,0,0,1));
  featureNext.push_back(cv::KeyPoint(268,188,3,-1,0,0,2));
  featureNext.push_back(cv::KeyPoint(462,204,3,-1,0,0,2));
  featureNext.push_back(cv::KeyPoint(142,192,3,-1,0,0,2));
  featureNext.push_back(cv::KeyPoint(324,181,3,-1,0,0,2));
  featureNext.push_back(cv::KeyPoint(538,241,3,-1,0,0,3));
  featureNext.push_back(cv::KeyPoint(209,228,3,-1,0,0,3));

  vector<cv::DMatch> matched;
  matchFeatures(1, featureLast, featureNext, matched);

}
