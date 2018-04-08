#include <iostream>
#include <string>
#include <vector>
#include <utility>
#include <typeinfo>
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
#include <opencv2/features2d.hpp>
#include "opencv2/core/core.hpp"
// #include "opencv2/imgproc/imgproc.hpp"
// #include "opencv2/highgui/highgui.hpp"
// #include <opencv2/opencv.hpp>
// #include "opencv2/xfeatures2d.hpp"
// #include <opencv2/highgui.hpp>
// #include "opencv2/imgproc/imgproc.hpp"
// #include <stdio.h>
// #include <stdlib.h>
// //#include <tinydir.h>
//#include "bal_problem.h"
#include "gflags/gflags.h"
#include "glog/logging.h"
#include <ceres/ceres.h>
// #include <Eigen/Core>
// #include <iostream>
// #include <utility>
#include <string>
// #include <vector>
#include <ceres/rotation.h>
#include <ceres/problem.h>
// #include <fstream>
// #include <algorithm>
// #include <cmath>
// #include <cstdio>
// #include <cstdlib>

using namespace cv;
using namespace std;

struct m_keypoint
{
	Point2d pt;
	int class_id;
	Point3d U;
};

float mThreshold = 1;

float computeResiduals(Point3d pt1, Point3d pt2){
  return pow((pow ((pt1.x-pt2.x),2) + pow ((pt1.y-pt2.y),2) + pow ((pt1.z-pt2.z),2)),0.5);
}

void matchFeatures(int imageId, vector<m_keypoint> featureLast, 
                  vector<m_keypoint> featureNext, vector<DMatch> &matched){
  float res, minRes;
  int featureLastRow = featureLast.size();
  int featureNextRow = featureNext.size();
  int index;

  for(int i = 0; i < featureNextRow; i++){
    minRes = mThreshold;
    for(int j = 0; j < featureLastRow; j++){
      if(featureNext[i].class_id == featureLast[j].class_id){
        res = computeResiduals(featureNext[i].U, featureLast[j].U);//check residuals, find the smallest one, save it
        if(res < minRes){
          minRes = res;
          index = j;
        }
      }
    }
    if(minRes < mThreshold){
      matched.push_back(DMatch(index,i,imageId,minRes));
      // cout << index << " " << i << " " << imageId << " " << minRes << endl;
    } 
  	// minRes = computeResiduals(featureNext[i].pt, featureLast[i].pt);
   //  matched.push_back(DMatch(i,i,imageId,minRes));
   //  cout << i << " " << imageId << " " << minRes << endl;
  }

 //cout<<matched[0].trainIdx<<endl;
}

void get_matched_points(//根据matches,返回匹配时两张图分别的坐标
	vector<m_keypoint>& p1,
	vector<m_keypoint>& p2,
	vector<DMatch> matches,
	vector<Point3d>& out_p1,
	vector<Point3d>& out_p2
	)
{
	out_p1.clear();
	out_p2.clear();
	for (int i = 0; i < matches.size(); ++i)
	{
		out_p1.push_back(p1[matches[i].queryIdx].U);
		out_p2.push_back(p2[matches[i].trainIdx].U);
	}
}



int main( int argc, char** argv )
{
	int start = stoi(argv[1]);
	int end = stoi(argv[2]);
	Mat K(Matx33d(
		350.6847, 0, 332.4661,
		0, 350.0606, 163.7461,
		0, 0, 1));
	Vec3b blue(255,0,0);
	Vec3b yellow(0,255,255);
	Vec3b orange(0,0,255);

	int resultSize = 1000;
	double resultResize = 100;
	

	vector<vector<m_keypoint> > key_points_for_all;
	//cout<<"typebefore"<<typeid(key_points_for_all).name()<<endl;
	vector<vector<Vec3b> > colors_for_all;
	vector<vector<DMatch> > matches_for_all;
	
	vector<Mat> camera_cor;
	Mat camera_3d(Matx31d(0,0,0));
	camera_cor.push_back(camera_3d);

	int imgId = 0;
	for(int i = start; i <= end; i++)
	{
		vector<m_keypoint> feature;
		vector<Vec3b> colors;
		ifstream csvPath ( "result/"+to_string(i)+"_stereo.csv" );
		string line, x, y, label, X, Y, Z; 
		int labelId;
		Mat imgLast, imgNext, outImg;
	    // Mat img = imread("result/"+to_string(i)+".png"); 

	    Mat result = Mat::zeros(resultSize, resultSize, CV_8UC3);
	    while (getline(csvPath, line)) 
	    {  
	        stringstream liness(line);  
	        getline(liness, x, ',');  
	        getline(liness, y, ','); 
	        getline(liness, label, ',');
	        getline(liness, X, ','); 
	        getline(liness, Y, ','); 
	        getline(liness, Z, ','); 
	        
	        // circle(img, Point (stoi(x),stoi(y)), 3, Scalar (0,0,0), CV_FILLED);
	        if(label == "blue"){
	          labelId = 0;
	          colors.push_back(blue);
	        }
	        if(label == "yellow"){
	          labelId = 1;
	          colors.push_back(yellow);
	        }
	        if(label == "orange"){
	          labelId = 2;
	          colors.push_back(orange);
	        }
            Point2d pt(stod(x),stod(y));
            Point3d U(stod(X), stod(Y), stod(Z));
        	m_keypoint m_keypoint1={pt,labelId,U};
        	feature.push_back(m_keypoint1);

        	int x = int(U.x * resultResize+resultSize/2);
			int y = int(U.z * resultResize+resultSize/2);
			if (x >= 0 && x <= resultSize && y>= 0 && y <= resultSize){
				circle(result, Point (x,y), 10, Scalar(colors.back()), CV_FILLED);
			}
    	}
    	flip(result, result, 0);
		namedWindow("img", WINDOW_NORMAL);
		imshow("img", result);
		waitKey(0);
		key_points_for_all.push_back(feature);
		colors_for_all.push_back(colors);
		// cout<<"type"<<typeid(key_points_for_all).name()<<endl;
		if (imgId > 0){
			vector<DMatch> matched;
			matchFeatures(imgId, key_points_for_all[imgId-1], key_points_for_all[imgId], matched);
			matches_for_all.push_back(matched);

			// imgLast = imread("result/"+to_string(imgId-1)+".png");
			// imgNext = imread("result/"+to_string(imgId)+".png");
			// resize(imgLast, imgLast, Size(320, 180));
			// resize(imgNext, imgNext, Size(320, 180));
			// drawMatches(imgLast, key_points_for_all[imgId-1], imgNext, key_points_for_all[imgId], matched, outImg);
			// namedWindow("MatchSIFT", WINDOW_NORMAL);
			// imshow("MatchSIFT",outImg);
			// waitKey(0);

			vector<Point3d> P1, P2;
			get_matched_points(key_points_for_all[imgId-1], key_points_for_all[imgId], matches_for_all[imgId-1], P1, P2);

			Mat affinetrans, mask;
		    estimateAffine3D(P1, P2, affinetrans, mask, 1, 0.99);
		    if (affinetrans.rows == 0){
		    	cout << "Fail to estimate affine transformation" << endl;
		    	return 0;
		    }
		    Mat R = affinetrans.colRange(0,3);
			// Mat T = -R.inv()*affinetrans.colRange(3,4);
			Mat T = affinetrans.colRange(3,4);
			
			camera_cor.push_back(R*camera_cor.back()+T);
		}
		imgId++;
	}

	vector<Point2f> path;
	Mat result = Mat::zeros(resultSize, resultSize, CV_8UC3);
	for(int i=0; i<camera_cor.size(); i++){
		cout<<camera_cor[i]<<endl;
		int x = int(camera_cor[i].at<double>(0,0) * resultResize+resultSize/2);
		int y = int(camera_cor[i].at<double>(2,0) * resultResize+resultSize/2);
		if (x >= 0 && x <= resultSize && y>= 0 && y <= resultSize){
			circle(result, Point (x,y), 10, Scalar (255,255,255), CV_FILLED);
		}
		path.push_back(Point2f(x,y));
	}
	for(int i=0; i<path.size()-1; i++){
		line(result, path[i], path[i+1], Scalar (255,255,255), 10, 8, 0);
	}
	flip(result, result, 0);
	namedWindow("result", WINDOW_NORMAL);
	imshow("result", result);
	waitKey(0);
	
}