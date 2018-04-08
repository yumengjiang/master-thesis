

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

using namespace std;

double computeResiduals(cv::Point2d pt1, cv::Point2d pt2)
{
	return pow((pow((pt1.x - pt2.x), 2) + pow((pt1.y - pt2.y), 2)), 0.5);
}

void matchFeatures(int imageId, vector<cv::KeyPoint> featureLast,
				   vector<cv::KeyPoint> featureNext, vector<cv::DMatch> &matched)
{
	double mThreshold = 50;
	double res, minRes;
	int featureLastRow = featureLast.size();
	int featureNextRow = featureNext.size();
	int index;

	for (int i = 0; i < featureNextRow; i++)
	{
		minRes = mThreshold;
		for (int j = 0; j < featureLastRow; j++)
		{
			if (featureNext[i].class_id == featureLast[j].class_id)
			{
				res = computeResiduals(featureNext[i].pt, featureLast[j].pt); //check residuals, find the smallest one, save it
				if (res < minRes)
				{
					minRes = res;
					index = j;
				}
			}
		}
		if (minRes < mThreshold)
		{
			matched.push_back(cv::DMatch(index, i, imageId, minRes));
			// cout << index << " " << i << " " << imageId << " " << minRes << endl;
		}
	}
}

void get_matched_points_and_colors(//根据matches,返回匹配时两张图分别的坐标
	vector<cv::KeyPoint>& p1,
	vector<cv::KeyPoint>& p2,
	vector<cv::Vec3b>& c1,
	vector<cv::Vec3b>& c2,
	vector<cv::DMatch> matches,
	vector<cv::Point2d>& out_p1,
	vector<cv::Point2d>& out_p2,
	vector<cv::Vec3b>& out_c1,
	vector<cv::Vec3b>& out_c2
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


bool find_transform(cv::Mat& K, vector<cv::Point2d>& p1, vector<cv::Point2d>& p2, cv::Mat& R, cv::Mat& T, cv::Mat& mask)
{
	//�����ڲξ�����ȡ�����Ľ����͹������꣨�������꣩
	double focal_length = 0.5*(K.at<double>(0) + K.at<double>(4));
	cv::Point2d principle_point(K.at<double>(2), K.at<double>(5));

	//����ƥ������ȡ����������ʹ��RANSAC����һ���ų�ʧ����
	cv::Mat E = cv::findEssentialMat(p1, p2, focal_length, principle_point, cv::RANSAC, 0.5, 1.0, mask);
	if (E.empty()) {
		cout << "Essential matrix not found!" << endl;
		return false;
	}

	double feasible_count = cv::countNonZero(mask);
	cout << (int)feasible_count << " -in- " << p1.size() << endl;
	//����RANSAC���ԣ�outlier��������50%ʱ�������ǲ��ɿ���
	// if (feasible_count <= 15 || (feasible_count / p1.size()) < 0.6)
	// 	return false;

	//�ֽⱾ�����󣬻�ȡ���Ա任
	double pass_count = cv::recoverPose(E, p1, p2, R, T, focal_length, principle_point, mask);

	//ͬʱλ����������ǰ���ĵ�������Ҫ�㹻��
	// if (pass_count / feasible_count < 0.7)
	// 	return false;

	return true;
}

void maskout_points(vector<cv::Point2d>& p, cv::Mat& mask)
{
	vector<cv::Point2d> p_copy = p;
	p.clear();

	for (int i = 0; i < mask.rows; ++i)
	{
		if (mask.at<uchar>(i) > 0)
			p.push_back(p_copy[i]);
	}
}

void maskout_colors(vector<cv::Vec3b>& c, cv::Mat& mask)
{
	vector<cv::Vec3b> c_copy = c;
	c.clear();

	for (int i = 0; i < mask.rows; ++i)
	{
		if (mask.at<uchar>(i) > 0)
			c.push_back(c_copy[i]);
	}
}

void reconstruct(cv::Mat& K, cv::Mat& R1, cv::Mat& T1, cv::Mat& R2, cv::Mat& T2, vector<cv::Point2d>& p1, vector<cv::Point2d>& p2, vector<cv::Point3d>& structure)//与双目的不同，有修改
{
	//����������ͶӰ����[R T]��triangulatePointsֻ֧��double��
	cv::Mat proj1(3, 4, CV_64FC1);
	cv::Mat proj2(3, 4, CV_64FC1);

	proj1.colRange(0,3) = R1;//将R1的值赋值给proj1的前三列前三行
	proj1.col(3) = T1;
	proj2.colRange(0,3) = R2;//将R2的值赋值给proj2的前三列前三行
	proj2.col(3) = T2;

	proj1 = K*proj1;
	proj2 = K*proj2;

	//�����ؽ�
	cv::Mat s;
	cv::triangulatePoints(proj1, proj2, p1, p2, s);//输出三维点：s

	structure.clear();
	structure.reserve(s.cols);//Requests that the vector capacity be at least enough to contain s.cols elements.
	for (int i = 0; i < s.cols; ++i)
	{
		cv::Mat_<double> col = s.col(i);
		col /= col(3);	///齐次坐标，需要除以最后一个元素才是真正的坐标值
		structure.push_back(cv::Point3d(col(0), col(1), col(2)));//将三维坐标放在最后
	}
}

void init_structure(//初始化
	cv::Mat K,
	vector<vector<cv::KeyPoint>>& key_points_for_all,
	vector<vector<cv::Vec3b>>& colors_for_all,
	vector<vector<cv::DMatch>>& matches_for_all,
	vector<cv::Point3d>& structure,
	vector<vector<int>>& correspond_struct_idx,
	vector<cv::Vec3b>& c1,
	vector<cv::Mat>& rotations,
	vector<cv::Mat>& motions,
	vector<cv::Mat>& camera_cor
	)
{
	//����ͷ����ͼ��֮���ı任����
	vector<cv::Point2d> p1, p2;
	vector<cv::Vec3b> c2;
	cv::Mat R, T;	//��ת������ƽ������
	cv::Mat mask;	//mask�д������ĵ�����ƥ���㣬����������ʧ����
	get_matched_points_and_colors(key_points_for_all[0], key_points_for_all[1], colors_for_all[0], colors_for_all[1], matches_for_all[0], p1, p2, c1, c2);
	find_transform(K, p1, p2, R, T, mask);//求得第二张图的P
	rotations.push_back(R);
	motions.push_back(T);
	camera_cor.push_back(R*camera_cor.back()+T);

	//��ͷ����ͼ��������ά�ؽ�
	maskout_points(p1, mask);
	maskout_points(p2, mask);
	maskout_colors(c1, mask);
	// maskout_colors(c2, mask);

	reconstruct(K, rotations.back(), motions.back(), R, T, p1, p2, structure);
	
	//将correspond_struct_idx变为key_points_for_all的形式
	correspond_struct_idx.clear();
	correspond_struct_idx.resize(key_points_for_all.size());
	for (int i = 0; i < key_points_for_all.size(); ++i)
	{
		correspond_struct_idx[i].resize(key_points_for_all[i].size(), -1);//与key points的size保持一致。为什么要加-1？
	}

	//��дͷ����ͼ���Ľṹ����
	int idx = 0;
	vector<cv::DMatch>& matches = matches_for_all[0];//将matches_for_all的第一个值赋给matches
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
	vector<cv::DMatch>& matches,//从第二张图开始，每两张图的matches，其中包含很多个match_features
	vector<int>& struct_indices,
	vector<cv::Point3d>& structure,
	vector<cv::KeyPoint>& key_points,//后一张图的key points
	vector<cv::Point3d>& object_points,
	vector<cv::Point2d>& image_points)
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
	vector<cv::DMatch>& matches,
	vector<int>& struct_indices,
	vector<int>& next_struct_indices,
	vector<cv::Point3d>& structure,
	vector<cv::Point3d>& next_structure,
	vector<cv::Vec3b>& colors,
	vector<cv::Vec3b>& next_colors
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
	cv::Mat K(cv::Matx33d(
		350.6847, 0, 364.4661,
		0, 350.0606, 163.7461,
		0, 0, 1));
	cv::Vec3b blue(255,0,0);
	cv::Vec3b yellow(0,255,255);
	cv::Vec3b orange(0,0,255);

	vector<vector<cv::KeyPoint> > key_points_for_all;
	//cout<<"typebefore"<<typeid(key_points_for_all).name()<<endl;
	vector<vector<cv::Vec3b> > colors_for_all;
	vector<vector<cv::DMatch> > matches_for_all;

	int imgId = 0;
	for(int i = start; i <= end; i++)
	{
		vector<cv::KeyPoint> feature;
		vector<cv::Vec3b> colors;
		ifstream csvPath ( "result1/"+to_string(i)+".csv" );
		string line, x, y, label, X, Y, Z; 
		int labelId;
		cv::Mat imgLast, imgNext, outImg;
	    // cv::Mat img = cv::imread("result/"+to_string(i)+".png"); 
	    while (getline(csvPath, line)) 
	    {  
	        stringstream liness(line);  
	        getline(liness, x, ',');  
	        getline(liness, y, ','); 
	        getline(liness, label, ','); 
	        
	        // cv::circle(img, Point (stoi(x),stoi(y)), 3, cv::Scalar (0,0,0), CV_FILLED);
	        if(label == "1"){
	          labelId = 0;
	          colors.push_back(blue);
	        }
	        if(label == "0"){
	          labelId = 1;
	          colors.push_back(yellow);
	        }
	        if(label == "2"){
	          labelId = 2;
	          colors.push_back(orange);
	        }
        	feature.push_back(cv::KeyPoint(stod(x),stod(y),3,-1,0,0,labelId));
    	}
		// cv::namedWindow("img", cv::WINDOW_NORMAL);
		// cv::imshow("img", img);
		// cv::waitKey(0);
		key_points_for_all.push_back(feature);
		colors_for_all.push_back(colors);
		// cout<<"type"<<typeid(key_points_for_all).name()<<endl;
		if (imgId > 0){
			vector<cv::DMatch> matched;
			matchFeatures(imgId, key_points_for_all[imgId-1], key_points_for_all[imgId], matched);
			matches_for_all.push_back(matched);

			// imgLast = cv::imread("result1/"+to_string(imgId-1)+".png");
			// imgNext = cv::imread("result1/"+to_string(imgId)+".png");
			// // cv::resize(imgLast, imgLast, cv::Size(320, 180));
			// // cv::resize(imgNext, imgNext, cv::Size(320, 180));
			// cv::drawMatches(imgLast, key_points_for_all[imgId-1], imgNext, key_points_for_all[imgId], matched, outImg);
			// cv::namedWindow("MatchSIFT", cv::WINDOW_NORMAL);
			// cv::imshow("MatchSIFT",outImg);
			// cv::waitKey(0);
		}
		imgId++;
	}

	vector<cv::Point3d> structure;
	vector<vector<int> > correspond_struct_idx;
	vector<cv::Vec3b> colors;
	vector<cv::Mat> rotations;
	vector<cv::Mat> motions;
	vector<cv::Mat> camera_cor;
	rotations.push_back(cv::Mat::eye(3, 3, CV_64FC1));
	motions.push_back(cv::Mat::zeros(3, 1, CV_64FC1));
	camera_cor.push_back(cv::Mat::zeros(3, 1, CV_64FC1));

	init_structure(//此时已做完第一张图和第二张图的重建，rotations和motions里放了两张图的R和T
		K,
		key_points_for_all,
		colors_for_all,
		matches_for_all,
		structure,
		correspond_struct_idx,
		colors,
		rotations,
		motions,
		camera_cor
		);

	for (int i = 1; i < matches_for_all.size(); ++i)//遍历，从第二张图和第三张图开始，每次两张图的match
	{
		vector<cv::Point3d> object_points;//3D points
		vector<cv::Point2d> image_points;
		cv::Mat r, R, T;
		//Mat mask;

		//输出本次遍历中两张图有用的match的3D坐标和新图片上的2D坐标
		get_objpoints_and_imgpoints(
			matches_for_all[i],
			correspond_struct_idx[i],
			structure,
			key_points_for_all[i+1],
			object_points,
			image_points
			);

		//bool solvePnPRansac(InputArray objectPoints, InputArray imagePoints, InputArray cameraMatrix, InputArray distCoeffs, OutputArray rvec, OutputArray tvec,
		// bool useExtrinsicGuess=false, int iterationsCount=100, float reprojectionError=8.0, double confidence=0.99, OutputArray inliers=noArray(), int flags=SOLVEPNP_ITERATIVE )
		//rvec – Output rotation vector (see Rodrigues() ) that, together with tvec , brings points from the model coordinate system to the camera coordinate system.
    //tvec – Output translation vector.
		cv::solvePnPRansac(object_points, image_points, K, cv::noArray(), r, T);
		cv::Rodrigues(r, R);//Converts a rotation matrix to a rotation vector or vice versa.
		//得到最新一张图的R和T
		rotations.push_back(R);
		motions.push_back(T);
		camera_cor.push_back(R*camera_cor.back()+T);

		vector<cv::Point2d> p1, p2;
		vector<cv::Vec3b> c1, c2;
		get_matched_points_and_colors(key_points_for_all[i], key_points_for_all[i + 1], colors_for_all[i], colors_for_all[i + 1], matches_for_all[i], p1, p2, c1, c2);//返回p1,p2，即两张图在匹配时的对应的坐标

		//求3D点
		vector<cv::Point3d> next_structure;
		reconstruct(K, rotations[i], motions[i], R, T, p1, p2, next_structure);//重建新的图
		fusion_structure(
			matches_for_all[i],
			correspond_struct_idx[i],
			correspond_struct_idx[i + 1],
			structure,
			next_structure,
			colors,
			c1
			);
	}

	int resultSize = 1000;
	double resultResize = 10;
	vector<cv::Point2d> path;
	cv::Mat result = cv::Mat::zeros(resultSize, resultSize, CV_8UC3);
	for(int u = 0; u < structure.size(); u++){
		cout << "3D points: " << structure[u] << colors[u] << endl;
		int x = int(structure[u].x * resultResize+resultSize/3);
		int y = int(structure[u].z * resultResize+resultSize/2);
		if (x >= 0 && x <= resultSize && y >= 0 && y <= resultSize){
		cv::circle(result, cv::Point (x,y), 3, cv::Scalar(colors[u]), CV_FILLED);
		}
	}
	for(int i = 0; i < camera_cor.size(); i++){
		cout << camera_cor[i] << endl;
		int x = int(camera_cor[i].at<double>(0,0) * resultResize+resultSize/2);
		int y = int(camera_cor[i].at<double>(2,0) * resultResize+resultSize/2);
		if (x >= 0 && x <= resultSize && y >= 0 && y <= resultSize){
			cv::circle(result, cv::Point (x,y), 10, cv::Scalar (255,255,255), CV_FILLED);
		}
		path.push_back(cv::Point2d(x,y));
	}
	for(int i=0; i<path.size()-1; i++){
		cv::line(result, path[i], path[i+1], cv::Scalar (255,255,255), 10, 8, 0);
	}
	cv::flip(result, result, 0);
	cv::namedWindow("result", cv::WINDOW_NORMAL);
	cv::imshow("result", result);
	cv::waitKey(0);
	
}