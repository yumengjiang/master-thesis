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
#include "gflags/gflags.h"
#include "glog/logging.h"
#include <ceres/ceres.h>
#include <string>
#include <ceres/rotation.h>
#include <ceres/problem.h>

using namespace cv;
using namespace std;

struct KP
{
	Point2d pt;
	int id;
};

float mThreshold = 0.1;

float computeResiduals(Point2d pt1, Point2d pt2){
  return pow((pow ((pt1.x-pt2.x),2) + pow ((pt1.y-pt2.y),2)),0.5);
}

void matchFeatures(int imageId, vector<KP> featureLast, 
                  vector<KP> featureNext, vector<DMatch> &matched){
  float res, minRes;
  int featureLastRow = featureLast.size();
  int featureNextRow = featureNext.size();
  int index;

  for(int i = 0; i < featureNextRow; i++){
    minRes = mThreshold;
    for(int j = 0; j < featureLastRow; j++){
      if(featureNext[i].id == featureLast[j].id){
        res = computeResiduals(featureNext[i].pt, featureLast[j].pt);//check residuals, find the smallest one, save it
        if(res < minRes){
          minRes = res;
          index = j;
        }
      }
    }
    if(minRes < mThreshold){
      matched.push_back(DMatch(index,i,imageId,minRes));
      cout << index << " " << i << " " << imageId << " " << minRes << endl;
    } 
  	// minRes = computeResiduals(featureNext[i].pt, featureLast[i].pt);
   //  matched.push_back(DMatch(i,i,imageId,minRes));
   //  cout << i << " " << imageId << " " << minRes << endl;
  }

 //cout<<matched[0].trainIdx<<endl;
}

void get_matched_points(//根据matches,返回匹配时两张图分别的坐标
	vector<KP>& p1,
	vector<KP>& p2,
	vector<Vec3b>& c1,
	vector<Vec3b>& c2,
	vector<DMatch>& matches,
	vector<Point2d>& out_p1,
	vector<Point2d>& out_p2,
	vector<Vec3b>& out_c1,
	vector<Vec3b>& out_c2
	)
{
	out_p1.clear();
	out_p2.clear();
	out_c1.clear();
	out_c2.clear();
	for (int i = 0; i < matches.size(); ++i)
	{
		out_p1.push_back(p1[matches[i].queryIdx].pt);
		out_p2.push_back(p2[matches[i].trainIdx].pt);
		out_c1.push_back(c1[matches[i].queryIdx]);
		out_c2.push_back(c2[matches[i].trainIdx]);
	}
}

void maskout_points(vector<Point2d>& p, Mat& mask)
{
	vector<Point2d> p_copy = p;
	p.clear();

	for (int i = 0; i < mask.rows; ++i)
	{
		if (mask.at<uchar>(i) > 0)
			p.push_back(p_copy[i]);
	}
}

void maskout_colors(vector<Vec3b>& p, Mat& mask)
{
	vector<Vec3b> p_copy = p;
	p.clear();

	for (int i = 0; i < mask.rows; ++i)
	{
		if (mask.at<uchar>(i) > 0)
			p.push_back(p_copy[i]);
	}
}

void init_structure(//初始化
	vector<vector<KP>>& key_points_for_all,
	vector<vector<Vec3b>>& colors_for_all,
	vector<vector<DMatch>>& matches_for_all,
	vector<Point2d>& structure,
	vector<Vec3b>& colors,
	vector<vector<int>>& correspond_struct_idx,
	vector<Mat>& affines
	)
{
	vector<Point2d> p1, p2;
	vector<Vec3b> c2;
	get_matched_points(key_points_for_all[0], key_points_for_all[1], colors_for_all[0], colors_for_all[1], matches_for_all[0], p1, p2, colors, c2);

	Mat affine(Matx33d(0,0,0,0,0,0,0,0,1)), affine_tmp;
	Mat mask;
	affine_tmp = estimateAffine2D(p1, p2, mask, RANSAC, 0.05);
	if (affine_tmp.rows == 0){
    	cout << "Fail to estimate affine transformation" << endl;
    	return;
    }
    affine_tmp.convertTo(affine.rowRange(0,2),CV_64FC1);
    affine = affine*affines.back();

	affines.push_back(affine);

    maskout_points(p1, mask);
	maskout_points(p2, mask);
	maskout_colors(colors, mask);

	for(int i = 0; i < p2.size(); i++){
		Mat pt(Matx31d(p2[i].x,p2[i].y,1));
		pt = affine.inv() * pt;
		structure.push_back(Point2d(pt.rowRange(0,2)));
	}

	

	// for(int j = 0; j < p2.size(); j++){
 //    	int x = int(p2[j].x * resultResize + resultSize/2);
	// 	int y = int(p2[j].y * resultResize + resultSize/2);
	// 	if (x >= 0 && x <= resultSize && y>= 0 && y <= resultSize){
	// 		circle(result, Point (x,y), 3, Scalar(255,0,0), CV_FILLED);
	// 	}
 //    }
  //   for(int i = 0; i < p1.size(); i++){
  //   	int x = int(p1[i].x * resultResize + resultSize/2);
		// int y = int(p1[i].y * resultResize + resultSize/2);
		// if (x >= 0 && x <= resultSize && y>= 0 && y <= resultSize){
		// 	circle(result, Point (x,y), 3, c1[j], CV_FILLED);
		// }

		// // Mat hpt(Matx31d(0,0,1));
		// // hpt.at<double>(0,0) = p1[j].x;
		// // hpt.at<double>(1,0) = p1[j].y;
  // //   	hpt = affine*hpt;
  // //   	x = int(hpt.at<double>(0,0) * resultResize + resultSize/2);
		// // y = int(hpt.at<double>(1,0) * resultResize + resultSize/2);
		// // if (x >= 0 && x <= resultSize && y>= 0 && y <= resultSize){
		// // 	circle(result, Point (x,y), 3, Scalar(0,0,255), CV_FILLED);
		// // }
  //   }

    correspond_struct_idx.clear();
	correspond_struct_idx.resize(key_points_for_all.size());
	for (int i = 0; i < key_points_for_all.size(); ++i)
	{
		correspond_struct_idx[i].resize(key_points_for_all[i].size(), -1);//与key points的size保持一致。为什么要加-1？
	}

	//��дͷ����ͼ���Ľṹ����
	int idx = 0;
	vector<DMatch>& matches = matches_for_all[0];//将matches_for_all的第一个值赋给matches
	for (int i = 0; i < matches.size(); ++i)
	{
		if (mask.at<uchar>(i) == 0)//如果这一对match是outlier
			continue;//如果mask的值等于0的话，继续从头开始循环，否则往下继续

		correspond_struct_idx[0][matches[i].queryIdx] = idx;//将第idx个match,即idx，存入correspond_struct_idx中，存的位置为：如果是前一张，则是第一行，后一张则是第二行，纵坐标对应点在图重视第几个feature point
		correspond_struct_idx[1][matches[i].trainIdx] = idx;
		++idx;
	}	
}

void get_objpoints_and_imgpoints(
	vector<DMatch>& matches,//从第二张图开始，每两张图的matches，其中包含很多个match_features
	vector<int>& struct_indices,
	vector<Point2d>& structure,
	vector<KP>& key_points,//后一张图的key points
	vector<Point2d>& object_points,
	vector<Point2d>& image_points)
{
	object_points.clear();
	image_points.clear();

	for (int i = 0; i < matches.size(); ++i)
	{
		int query_idx = matches[i].queryIdx;//在相应的图中是第几个feature point
		int train_idx = matches[i].trainIdx;

		int struct_idx = struct_indices[query_idx];//访问对应列，输出为第几个有用的match，一个有用的match能组成一个U
		if (struct_idx < 0) continue;

		object_points.push_back(structure[struct_idx]);//输出该次匹配有用的match的3D坐标
		image_points.push_back(key_points[train_idx].pt);//输出有用的match在最新的图上match的坐标
	}
}

void fusion_structure(
	vector<DMatch>& matches,
	vector<int>& struct_indices,
	vector<int>& next_struct_indices,
	vector<Point2d>& structure,
	vector<Point2d>& next_structure,
	vector<Vec3b>& colors,
	vector<Vec3b>& next_colors
	)
{
	for (int i = 0; i < matches.size(); ++i)
	{
		int query_idx = matches[i].queryIdx;
		int train_idx = matches[i].trainIdx;

		int struct_idx = struct_indices[query_idx];//如果两者匹配，假如前一张图的匹配点能构成3D点，后一张图匹配点应该属于同一个3D点
		if (struct_idx >= 0)
		{
			next_struct_indices[train_idx] = struct_idx;
			continue;
		}
		structure.push_back(next_structure[i]);
		colors.push_back(next_colors[i]);
		struct_indices[query_idx] = next_struct_indices[train_idx] = structure.size() - 1;
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
	
	vector<vector<KP>> key_points_for_all;
	vector<vector<Vec3b>> colors_for_all;
	vector<vector<DMatch>> matches_for_all;
	vector<Mat> affines;
	Mat affine = Mat::eye(3,3,CV_64FC1);
	affines.push_back(affine);

	int imgId = 0;
	for(int i = start; i <= end; i++)
	{
		vector<KP> feature;
		vector<Vec3b> colors;
		ifstream csvPath ( "results_3d/"+to_string(i)+".csv" );
		string line, x, y, label, X, Y, Z; 
		int id;
		// Mat imgLast, imgNext, outImg;
	    // Mat img = imread("result/"+to_string(i)+".png"); 
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
	          id = 0;
	          colors.push_back(blue);
	        }
	        if(label == "yellow"){
	          id = 1;
	          colors.push_back(yellow);
	        }
	        if(label == "orange"){
	          id = 2;
	          colors.push_back(orange);
	        }
            Point2d pt(stod(X),stod(Z));
        	KP keypoint = {pt, id};
        	feature.push_back(keypoint);
    	}
		key_points_for_all.push_back(feature);
		colors_for_all.push_back(colors);
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
		}
		imgId++;
	}

	vector<Point2d> structure;
	vector<Vec3b> colors;
	vector<vector<int>> correspond_struct_idx;

	init_structure(//此时已做完第一张图和第二张图的重建，rotations和motions里放了两张图的R和T
		key_points_for_all,
		colors_for_all,
		matches_for_all,
		structure,
		colors,
		correspond_struct_idx,
		affines
		);

	for (int i = 1; i < matches_for_all.size(); ++i)//遍历，从第二张图和第三张图开始，每次两张图的match
	{
		vector<Point2d> object_points; 
		vector<Point2d> image_points;
		
		get_objpoints_and_imgpoints(
			matches_for_all[i],
			correspond_struct_idx[i],
			structure,
			key_points_for_all[i+1],
			object_points,
			image_points
			);

		vector<Point2d> p1, p2;
		vector<Vec3b> c1, c2;
		get_matched_points(key_points_for_all[i], key_points_for_all[i + 1], colors_for_all[i], colors_for_all[i+1], matches_for_all[i], p1, p2, c1, c2);//返回p1,p2，即两张图在匹配时的对应的坐标
		
		Mat affine(Matx33d(0,0,0,0,0,0,0,0,1)), affine_tmp;
		Mat mask;
		affine_tmp = estimateAffine2D(p1, p2, mask, RANSAC, 0.02);
		if (affine_tmp.rows == 0){
	    	cout << "Fail to estimate affine transformation" << endl;
	    	return 0;
	    }
	    affine_tmp.convertTo(affine.rowRange(0,2),CV_64FC1);
	    affine = affine*affines.back();

		affines.push_back(affine);


		maskout_points(p1, mask);
		maskout_points(p2, mask);
		maskout_colors(c1, mask);

		vector<Point2d> next_structure;
		for(int j = 0; j < p2.size(); j++){
			Mat pt(Matx31d(p2[j].x,p2[j].y,1));
			pt = affine.inv() * pt;
			next_structure.push_back(Point2d(pt.rowRange(0,2)));
		}

		fusion_structure(
			matches_for_all[i],
			correspond_struct_idx[i],
			correspond_struct_idx[i+1],
			structure,
			next_structure,
			colors,
			c1
			);
	}
	
	// for (int i = 0; i < correspond_struct_idx.size(); ++i){
	// 	for (int j = 0; j < correspond_struct_idx[i].size(); ++j){
	// 		cout << correspond_struct_idx[i][j] << " ";
	// 	}
	// 	cout << "\n";
	// }

	int resultSize = 1000;
	double resultResize = 100;
	Mat result = Mat::zeros(resultSize, resultSize, CV_8UC3);
	vector<Point2d> path;
	for(int i = 0; i < structure.size(); i++){
		// cout << structure[i] << colors[i] << endl;
		int x = int(structure[i].x * resultResize + resultSize/2);
		int y = int(structure[i].y * resultResize + resultSize/2);
		if (x >= 0 && x <= resultSize && y >= 0 && y <= resultSize){
			circle(result, Point (x,y), 3, colors[i], CV_FILLED);
		}
	}

	for(int i = 0; i < affines.size(); i++){
		Mat camera_cor(Matx31d(0,0,1));
		camera_cor = affines[i].inv() * camera_cor;
		// cout << camera_cor << endl;
		int x = int(camera_cor.at<double>(0,0) * resultResize + resultSize/2);
		int y = int(camera_cor.at<double>(1,0) * resultResize + resultSize/2);
		if (x >= 0 && x <= resultSize && y >= 0 && y <= resultSize){
			circle(result, Point (x,y), 3, Scalar (255,255,255), CV_FILLED);
		}
		path.push_back(Point2d(x,y));
	}
	for(int i=0; i<path.size()-1; i++){
		line(result, path[i], path[i+1], Scalar (255,255,255), 1, 1, 0);
	}
	flip(result, result, 0);
    namedWindow("result", WINDOW_NORMAL);
	imshow("result", result);
	waitKey(0);
	
}