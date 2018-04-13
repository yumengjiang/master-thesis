        /*
                           _ooOoo_
                          o8888888o
                          88" . "88
                          (| -_- |)
                          O\  =  /O
                       ____/`---'\____
                     .'  \\|     |//  `.
                    /  \\|||  :  |||//  \
                   /  _||||| -:- |||||-  \
                   |   | \\\  -  /// |   |
                   | \_|  ''\---/''  |   |
                   \  .-\__  `-`  ___/-. /
                 ___`. .'  /--.--\  `. . __
              ."" '<  `.___\_<|>_/___.'  >'"".
             | | :  `- \`.;`\ _ /`;.`/ - ` : | |
             \  \ `-.   \_ __\ /__ _/   .-` /  /
        ======`-.____`-.___\_____/___.-`____.-'======
                           `=---='
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                 佛祖保佑        结果正确
        */

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
	Point3d pt;
	int id;
};



float computeResidual(Point3d pt1, Point3d pt2){
	return pow((pow ((pt1.x-pt2.x),2) + pow ((pt1.y-pt2.y),2)),0.5);
}

void matchFeatures(vector<KP> featureLast, 
                  vector<KP> featureNext, vector<DMatch> &matched){
	float mThreshold = 0.2;
	float res, minRes;
	int featureLastRow = featureLast.size();
	int featureNextRow = featureNext.size();
	int index;

	for(int i = 0; i < featureNextRow; i++){
		minRes = mThreshold;
		for(int j = 0; j < featureLastRow; j++){
		    if(featureNext[i].id == featureLast[j].id){
		        res = computeResidual(featureNext[i].pt, featureLast[j].pt);//check residuals, find the smallest one, save it
				if(res < minRes){
				    minRes = res;
				    index = j;
		    	}
			}
		}
		if(minRes < mThreshold){
			matched.push_back(DMatch(index,i,minRes));
			// cout << index << " " << i << " " << imageId << " " << minRes << endl;
		} 
	}
}

void matchFeaturesAffine(Mat affine, vector<KP> featureLast, 
                  vector<KP> featureNext, vector<DMatch> &matched){
	float mThreshold = 0.1;
	float res, minRes;
	int featureLastRow = featureLast.size();
	int featureNextRow = featureNext.size();
	int index;
	matched.clear();

	for(int i = 0; i < featureNextRow; i++){
		minRes = mThreshold;
		for(int j = 0; j < featureLastRow; j++){
		    if(featureNext[i].id == featureLast[j].id){
		  	    Point3d affine_pt(Mat(affine*Mat(featureLast[j].pt))); 
		        res = computeResidual(featureNext[i].pt, affine_pt);//check residuals, find the smallest one, save it
		        if(res < minRes){
		            minRes = res;
		            index = j;
		        }
		    }
		}
		if(minRes < mThreshold){
		    matched.push_back(DMatch(index,i,minRes));
		    // cout << index << " " << i << " " << imageId << " " << minRes << endl;
		} 
	}
}

void get_matched_points(//根据matches,返回匹配时两张图分别的坐标
	vector<KP>& last_keypoints,
	vector<KP>& next_keypoints,
	vector<Vec3b>& last_colors,
	vector<Vec3b>& next_colors,
	vector<DMatch>& matches,
	vector<Point3d>& p1,
	vector<Point3d>& p2,
	vector<Vec3b>& c1,
	vector<Vec3b>& c2
	)
{
	p1.clear();
	p2.clear();
	c1.clear();
	c2.clear();
	for (int i = 0; i < matches.size(); ++i)
	{
		p1.push_back(last_keypoints[matches[i].queryIdx].pt);
		p2.push_back(next_keypoints[matches[i].trainIdx].pt);
		c1.push_back(last_colors[matches[i].queryIdx]);
		c2.push_back(next_colors[matches[i].trainIdx]);
	}
}

void estimateTransform2D(vector<Point2d> p1, vector<Point2d> p2, Mat& affine){
	if(p1.size()<2){
		cout << "Too few points to estimate transformation!" << endl;
		return;
	}
	Mat M(2*p1.size(),4,CV_64F), B(2*p1.size(),1,CV_64F);
	for(int i = 0; i < p1.size(); i++){
		Mat m = (Mat_<double>(2,4) << p1[i].x, -p1[i].y, 1, 0, p1[i].y, p1[i].x, 0, 1);
		m.copyTo(M.rowRange(2*i,2*i+2));
		B.at<double>(2*i,0) = p2[i].x;
		B.at<double>(2*i+1,0) = p2[i].y;
	} 
	Mat theta = (M.t()*M).inv()*(M.t())*B;
	double a = theta.at<double>(0,0), b = theta.at<double>(1,0), tx = theta.at<double>(2,0), ty = theta.at<double>(3,0);
	affine = (Mat_<double>(2,3) << a,-b,tx,b,a,ty);
	// cout << affine << endl;
}

Mat reconstruct(//初始化
	vector<KP>& last_keypoints,
	vector<KP>& next_keypoints,
	vector<Vec3b>& last_colors,
	vector<Vec3b>& next_colors,
	vector<DMatch>& matches,
	vector<Vec3b>& c1,
	vector<Point3d>& p2
	)
{
	vector<Point3d> p1;
	vector<Point2d> p3, p4;
	vector<double> res;
	vector<Vec3b> c2;
	Mat affine_tmp;
	Mat affine(Matx33d(0,0,0,0,0,0,0,0,1));
	get_matched_points(last_keypoints, next_keypoints, last_colors, next_colors, matches, p1, p2, c1, c2);

	// Mat mask;

	for(int i = 0; i < p1.size(); i++){
		p3.push_back(Point2d(p1[i].x,p1[i].y));
		p4.push_back(Point2d(p2[i].x,p2[i].y));
	}
	
	estimateTransform2D(p3, p4, affine_tmp);
	// affine_tmp = estimateRigidTransform(p3, p4, false);
	if (affine_tmp.rows == 0){
    	cout << "Fail to estimate affine transformation, number of points: " << p3.size() << endl;
    }
    // cout << affine_tmp;
    affine_tmp.colRange(0,2) /= pow(pow(affine_tmp.at<double>(0,0),2)+pow(affine_tmp.at<double>(0,1),2),0.5);
    affine_tmp.convertTo(affine.rowRange(0,2),CV_64F);
    // cout << affine_tmp << endl;
 //    for(int i = 0; i < p1.size(); i++){
	// 	last_keypoints[matches[i].queryIdx].pt = (p1[i]+Point3d(Mat(affine.inv()*Mat(p2[i]))))/2;
	// 	// cout << "p1: " << p1[i] << ", after adjustment: " << last_keypoints[matches[i].queryIdx].pt << endl;;
	// }
	return affine;
}

void init_structure(//初始化
	vector<vector<KP>>& keypoints_for_all,
	vector<vector<Vec3b>>& colors_for_all,
	vector<vector<DMatch>>& matches_for_all,
	vector<Point3d>& structure,
	vector<Vec3b>& colors,
	vector<vector<int>>& correspond_struct_idx,
	vector<Mat>& affines
	)
{
	vector<Point3d> p2;
	Mat affine = reconstruct(keypoints_for_all[0], keypoints_for_all[1], colors_for_all[0], colors_for_all[1], matches_for_all[0], colors, p2);

    // matchFeaturesAffine(affine, keypoints_for_all[0], keypoints_for_all[1], matches_for_all[0]);
    // reconstruct(keypoints_for_all[0], keypoints_for_all[1], colors_for_all[0], colors_for_all[1], matches_for_all[0], colors, p2, affine);

    affine = affine*affines.back();
	affines.push_back(affine);

	for(int i = 0; i < p2.size(); i++){
		structure.push_back(Point3d(Mat(affine.inv()*Mat(p2[i]))));
	}

    correspond_struct_idx.clear();
	correspond_struct_idx.resize(keypoints_for_all.size());
	for (int i = 0; i < keypoints_for_all.size(); ++i)
	{
		correspond_struct_idx[i].resize(keypoints_for_all[i].size(), -1);//与key points的size保持一致。为什么要加-1？
	}

	//��дͷ����ͼ���Ľṹ����
	int idx = 0;
	vector<DMatch>& matches = matches_for_all[0];//将matches_for_all的第一个值赋给matches
	for (int i = 0; i < matches.size(); ++i)
	{
		// if (mask.at<uchar>(i) == 0)//如果这一对match是outlier
		// 	continue;//如果mask的值等于0的话，继续从头开始循环，否则往下继续

		correspond_struct_idx[0][matches[i].queryIdx] = idx;//将第idx个match,即idx，存入correspond_struct_idx中，存的位置为：如果是前一张，则是第一行，后一张则是第二行，纵坐标对应点在图重视第几个feature point
		correspond_struct_idx[1][matches[i].trainIdx] = idx;
		++idx;
	}	
}

void fusion_structure(
	vector<DMatch>& matches,
	vector<int>& struct_indices,
	vector<int>& next_struct_indices,
	vector<Point3d>& structure,
	vector<Point3d>& next_structure,
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
			// structure[struct_idx] = next_structure[i];
			// structure[struct_idx] = (structure[struct_idx]+next_structure[i])/2;
			// cout << "structure: " << structure[struct_idx] << ", after adjustment: " << next_structure[i] << endl;;
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
	
	vector<vector<KP>> keypoints_for_all;
	vector<vector<Vec3b>> colors_for_all;
	vector<vector<DMatch>> matches_for_all;
	vector<Mat> affines;
	Mat affine = Mat::eye(3,3,CV_64F);
	affines.push_back(affine);

	int imgId = 0;
	for(int i = start; i <= end; i++)
	{
		vector<KP> keypoints;
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
	        // cout << stod(Z) << endl;
	        if(stod(Z)<2){
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
	            Point3d pt(stod(X),stod(Z),1);
	        	KP keypoint = {pt, id};
	        	keypoints.push_back(keypoint);
	        }
	        
    	}
    	if(keypoints.size()<2){
    		cout << "Too few keypoint!" << endl;
    		return 0;
    	}
		keypoints_for_all.push_back(keypoints);
		colors_for_all.push_back(colors);
		if (imgId > 0){
			vector<DMatch> matched;
			matchFeatures(keypoints_for_all[imgId-1], keypoints_for_all[imgId], matched);
			matches_for_all.push_back(matched);

			// imgLast = imread("result/"+to_string(imgId-1)+".png");
			// imgNext = imread("result/"+to_string(imgId)+".png");
			// resize(imgLast, imgLast, Size(320, 180));
			// resize(imgNext, imgNext, Size(320, 180));
			// drawMatches(imgLast, keypoints_for_all[imgId-1], imgNext, keypoints_for_all[imgId], matched, outImg);
			// namedWindow("MatchSIFT", WINDOW_NORMAL);
			// imshow("MatchSIFT",outImg);
			// waitKey(0);
		}
		imgId++;
	}

	vector<Point3d> structure;
	vector<Vec3b> colors;
	vector<vector<int>> correspond_struct_idx;

	init_structure(//此时已做完第一张图和第二张图的重建，rotations和motions里放了两张图的R和T
		keypoints_for_all,
		colors_for_all,
		matches_for_all,
		structure,
		colors,
		correspond_struct_idx,
		affines
		);

	for (int i = 1; i < matches_for_all.size(); ++i)//遍历，从第二张图和第三张图开始，每次两张图的match
	{
		vector<Point3d> p2;
		vector<Vec3b> c1;
		Mat affine = reconstruct(keypoints_for_all[i], keypoints_for_all[i+1], colors_for_all[i], colors_for_all[i+1], matches_for_all[i], c1, p2);


		// vector<DMatch> matchesd;
		// matchFeaturesAffine(affine*affines[i],keypoints_for_all[i-1], keypoints_for_all[i+1], matched);


	    // matchFeaturesAffine(affine, keypoints_for_all[i], keypoints_for_all[i+1], matches_for_all[i]);
	    // reconstruct(keypoints_for_all[i], keypoints_for_all[i+1], colors_for_all[i], colors_for_all[i+1], matches_for_all[i], c1, p2, affine);

	    affine = affine*affines.back();
		affines.push_back(affine);

		vector<Point3d> next_structure;
		for(int i = 0; i < p2.size(); i++){
			next_structure.push_back(Point3d(Mat(affine.inv()*Mat(p2[i]))));
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
	

	// for (int it = 0; it < 5; ++it)
	// 	for (int i = 0; i < matches_for_all.size(); ++i)//遍历，从第二张图和第三张图开始，每次两张图的match
	// 	{
	// 		vector<Point3d> p2;
	// 		vector<Vec3b> c1;
	// 		Mat affine = reconstruct(keypoints_for_all[i], keypoints_for_all[i+1], colors_for_all[i], colors_for_all[i+1], matches_for_all[i], c1, p2);

	// 		affines[i+1] = affine * affines[i];

	// 		// vector<Point3d> next_structure;
	// 		// for(int i = 0; i < p2.size(); i++){
	// 		// 	next_structure.push_back(Point3d(Mat(affines[i+1].inv()*Mat(p2[i]))));
	// 		// }

	// 		// fusion_structure(
	// 		// 	matches_for_all[i],
	// 		// 	correspond_struct_idx[i],
	// 		// 	correspond_struct_idx[i+1],
	// 		// 	structure,
	// 		// 	next_structure,
	// 		// 	colors,
	// 		// 	c1
	// 		// 	);

	// 	}



	vector<int> count_same_structure;
	count_same_structure.resize(structure.size());
	for (int i = 0; i < correspond_struct_idx.size(); ++i){
		for (int j = 0; j < correspond_struct_idx[i].size(); ++j){
			// cout << correspond_struct_idx[i][j] << " ";
			for(int k = 0; k < structure.size(); k++){
				if(correspond_struct_idx[i][j] == k)
					count_same_structure[k]++;
			}
		}
		// cout << "\n";
	}

	int resultSize = 1000;
	double resultResize = 100;
	Mat result = Mat::zeros(resultSize, resultSize, CV_8UC3);
	vector<Point2d> path;
	
	int count = 0;
	for(int i = 0; i < structure.size(); i++){
		// cout << count_same_structure[i] << endl;
		// cout << structure[i] << colors[i] << endl;
		if(count_same_structure[i] > 2){
			count++;
			int x = int(structure[i].x * resultResize + resultSize/4);
			int y = int(structure[i].y * resultResize + resultSize/4);
			if (x >= 0 && x <= resultSize && y >= 0 && y <= resultSize){
				circle(result, Point (x,y), 3, colors[i], CV_FILLED);
			}
		}
	}
	cout << "Number of structure: " << count << endl;

	for(int i = 0; i < affines.size(); i++){
		Mat camera_cor(Matx31d(0,0,1));
		camera_cor = affines[i].inv() * camera_cor;
		int x = int(camera_cor.at<double>(0,0) * resultResize + resultSize/4);
		int y = int(camera_cor.at<double>(1,0) * resultResize + resultSize/4);
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