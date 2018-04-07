

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
// #include "stdafx.h"
#include <algorithm> //包含通用算法
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
#include <fstream>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>

using std::cin;
using namespace cv;
using namespace std;
using namespace ceres;
using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solve;
using ceres::Solver;

// void extract_features(//存入所有照片的descriptor和key points
// 	vector<string>& image_names,
// 	vector<vector<KeyPoint>>& key_points_for_all,
// 	vector<Mat>& descriptor_for_all,
// 	vector<vector<Vec3b>>& colors_for_all
// 	)
// {
// 	key_points_for_all.clear();
// 	descriptor_for_all.clear();
// 	Mat image;

// 	//��ȡͼ�񣬻�ȡͼ�������㣬������
// 	Ptr<Feature2D> sift = xfeatures2d::SIFT::create(0, 3, 0.04, 10);
// 	for (auto it = image_names.begin(); it != image_names.end(); ++it)
// 	{
// 		image = imread(*it);
// 		if (image.empty()) continue;

// 		cout << "Extracing features: " << *it << endl;

// 		vector<KeyPoint> key_points;
// 		Mat descriptor;
// 		//ż�������ڴ�����ʧ�ܵĴ���
// 		sift->detectAndCompute(image, noArray(), key_points, descriptor);

// 		//���������٣����ų���ͼ��
// 		if (key_points.size() <= 10) continue;

// 		key_points_for_all.push_back(key_points);
// 		descriptor_for_all.push_back(descriptor);

// 		vector<Vec3b> colors(key_points.size());
// 		for (int i = 0; i < key_points.size(); ++i)
// 		{
// 			Point2f& p = key_points[i].pt;
// 			colors[i] = image.at<Vec3b>(p.y, p.x);
// 		}
// 		colors_for_all.push_back(colors);
// 	}
// }

// void match_features(Mat& query, Mat& train, vector<DMatch>& matches)
// {
// 	vector<vector<DMatch>> knn_matches;
// 	BFMatcher matcher(NORM_L2);
// 	matcher.knnMatch(query, train, knn_matches, 2);

// 	//��ȡ����Ratio Test����Сƥ���ľ���
// 	float min_dist = FLT_MAX;
// 	for (int r = 0; r < knn_matches.size(); ++r)
// 	{
// 		//Ratio Test
// 		if (knn_matches[r][0].distance > 0.6*knn_matches[r][1].distance)
// 			continue;

// 		float dist = knn_matches[r][0].distance;
// 		if (dist < min_dist) min_dist = dist;
// 	}

// 	matches.clear();
// 	for (size_t r = 0; r < knn_matches.size(); ++r)
// 	{
// 		//�ų�������Ratio Test�ĵ���ƥ�����������ĵ�
// 		if (
// 			knn_matches[r][0].distance > 0.6*knn_matches[r][1].distance ||
// 			knn_matches[r][0].distance > 5 * max(min_dist, 10.0f)
// 			)
// 			continue;

// 		//����ƥ����
// 		matches.push_back(knn_matches[r][0]);
// 	}
// }

// void match_features(vector<Mat>& descriptor_for_all, vector<vector<DMatch>>& matches_for_all)
// {
// 	matches_for_all.clear();
// 	for (int i = 0; i < descriptor_for_all.size() - 1; ++i)//遍历每个图的descriptor
// 	{
// 		cout << "Matching images " << i << " - " << i + 1 << endl;//每次两张图
// 		vector<DMatch> matches;
// 		match_features(descriptor_for_all[i], descriptor_for_all[i + 1], matches);//将每次匹配的结果作为一个元素放入match_features中
// 		matches_for_all.push_back(matches);
// 	}
// }
struct keypoint
{
	Point2f pt;
	int class_id;
	Point3f Pt;
};
float mThreshold = 50;

float computeResiduals(Point2f pt1, Point2f pt2)
{
	return pow((pow((pt1.x - pt2.x), 2) + pow((pt1.y - pt2.y), 2)), 0.5f);
}
// matchFeatures(imgId, key_points_for_all[imgId-1], key_points_for_all[imgId], matched);
void matchFeatures(int imageId, vector<keypoint> featureLast,
				   vector<keypoint> featureNext, vector<DMatch> &matched)
{
	float res, minRes;
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
			matched.push_back(DMatch(index, i, imageId, minRes));
			//cout << index << " " << i << " " << imageId << " " << minRes << endl;
		}
		// minRes = computeResiduals(featureNext[i].pt, featureLast[i].pt);
		//  matched.push_back(DMatch(i,i,imageId,minRes));
		//  cout << i << " " << imageId << " " << minRes << endl;
	}

	//cout<<matched[0].trainIdx<<endl;
}
//
void matchFeaturesForAll(vector<vector<keypoint>> &key_points_for_all, vector<vector<DMatch>> &matches_for_all)
{
	matches_for_all.clear();
	for (int i = 0; i < key_points_for_all.size() - 1; ++i) //遍历每个图的descriptor
	{
		cout << "Matching images " << i << " - " << i + 1 << endl; //每次两张图
		vector<DMatch> matches;
		int imageId = i + 1;
		matchFeatures(imageId, key_points_for_all[i], key_points_for_all[i + 1], matches); //将每次匹配的结果作为一个元素放入matchFeaturesForAll中
		matches_for_all.push_back(matches);
	}
}
//bool operator()(const T* const intrinsic, const T* const extrinsic, const T* const pos3d, T* residuals) const;

// float findResidualsforRT(Mat K, Mat R, Mat T, Point2f p2, vector<Point3d> Structure)
// {
// 	//Mat K(Matx33d(
// 	// 350.6847, 0, 332.4661,
// 	// 0, 350.0606, 163.7461,
// 	// 0, 0, 1));
// 	Mat pos_proj;
// 	Mat structure = Mat(Structure[0]);
// 	pos_proj = R * structure;
// 	// cout<<pos_proj.size()<<endl;
// 	// Apply the camera translation
// 	pos_proj.at<double>(0, 0) += T.at<double>(0, 0); //平移变化
// 	pos_proj.at<double>(0, 1) += T.at<double>(1, 0);
// 	pos_proj.at<double>(0, 2) += T.at<double>(2, 0);

// 	float x = pos_proj.at<double>(0, 0) / pos_proj.at<double>(0, 2); //求x和y
// 	float y = pos_proj.at<double>(0, 1) / pos_proj.at<double>(0, 2);

// 	float fx = K.at<double>(0, 0); //读取内参矩阵
// 	float fy = K.at<double>(1, 1);
	
// 	float cx = K.at<double>(0, 2);
	
// 	float cy = K.at<double>(1, 2);
	

// 	// Apply intrinsic
// 	float u = fx * x + cx; //这是啥？
// 	float v = fy * y + cy;
// 	return pow((pow((u - p2.x), 2) + pow((v - p2.y), 2)), 0.5f);
// }

void findResidualsforRT(Mat K, Mat R, Mat T, vector<Point2f> p2, vector<Point3d> Structure, vector<double> &residuals)
{
	residuals.clear();

	for(int i = 0; i < Structure.size(); i ++){
		Mat pos_proj = Mat(Structure[i]);
		pos_proj = R * pos_proj + T;
		pos_proj = K * pos_proj;
		double u = pos_proj.at<double>(0, 0) / pos_proj.at<double>(0, 2); //求x和y
		double v = pos_proj.at<double>(0, 1) / pos_proj.at<double>(0, 2);
		double res = pow((pow((u - p2[i].x), 2) + pow((v - p2[i].y), 2)), 0.5f);
		cout << res << endl;
		residuals.push_back(res);
	}
}


// void reconstruct(Mat &K, Mat &R1, Mat &T1, Mat &R2, Mat &T2, Point2f &p1, Point2f &p2, vector<Point3d> &structure) //与双目的不同，有修改
// {
// 	//����������ͶӰ����[R T]��triangulatePointsֻ֧��float��
// 	Mat proj1(3, 4, CV_32FC1);
// 	Mat proj2(3, 4, CV_32FC1);

// 	R1.convertTo(proj1(Range(0, 3), Range(0, 3)), CV_32FC1); //将R1的值赋值给proj1的前三列前三行
// 	T1.convertTo(proj1.col(3), CV_32FC1);					 //将T的值赋值到第四列
// 	R2.convertTo(proj2(Range(0, 3), Range(0, 3)), CV_32FC1); //将R2的值赋值给proj2的前三列前三行
// 	T2.convertTo(proj2.col(3), CV_32FC1);
		
// 	Mat fK;
// 	K.convertTo(fK, CV_32FC1);

// 	proj1 = fK * proj1;
// 	proj2 = fK * proj2;
// 	vector<Point2f> p_1;
// 	vector<Point2f> p_2;
// 	p_1.push_back(p1);
// 	p_2.push_back(p2);
// 	//�����ؽ�
// 	Mat s;
// 	triangulatePoints(proj1, proj2, p_1, p_2, s); //输出三维点：s
// 	structure.clear();
// 	structure.reserve(s.cols); //Requests that the vector capacity be at least enough to contain s.cols elements.
// 	for (int i = 0; i < s.cols; ++i)
// 	{
// 		Mat_<float> col = s.col(i);
// 		col /= col(3);										  ///齐次坐标，需要除以最后一个元素才是真正的坐标值
// 		structure.push_back(Point3f(col(0), col(1), col(2))); //将三维坐标放在最后
// 	}
// }

void reconstruct(Mat& K, Mat& R1, Mat& T1, Mat& R2, Mat& T2, vector<Point2f>& p1, vector<Point2f>& p2, vector<Point3d>& structure)//与双目的不同，有修改
{
  //����������ͶӰ����[R T]��triangulatePointsֻ֧��float��
  cv::Mat proj1(3, 4, CV_32FC1);
  cv::Mat proj2(3, 4, CV_32FC1);

  R1.convertTo(proj1(cv::Range(0, 3), cv::Range(0, 3)), CV_32FC1);//将R1的值赋值给proj1的前三列前三行
  T1.convertTo(proj1.col(3), CV_32FC1);//将T的值赋值到第四列
 

  R2.convertTo(proj2(cv::Range(0, 3),cv::Range(0, 3)), CV_32FC1);//将R2的值赋值给proj2的前三列前三行
  T2.convertTo(proj2.col(3), CV_32FC1);

  cv::Mat fK;
  K.convertTo(fK, CV_32FC1);
  proj1 = fK*proj1;
  proj2 = fK*proj2;

  cv::Mat s;
  triangulatePoints(proj1, proj2, p1, p2, s);//输出三维点：s

  structure.clear();
  structure.reserve(s.cols);//Requests that the vector capacity be at least enough to contain s.cols elements.
  for (int i = 0; i < s.cols; ++i)
   {
     cv::Mat_<float> col = s.col(i);
     col /= col(3);  ///齐次坐标，需要除以最后一个元素才是真正的坐标值
     structure.push_back(cv::Point3f(col(0), col(1), col(2)));//将三维坐标放在最后
   }
}

bool find_transform(Mat &K, vector<Point2f> &p1, vector<Point2f> &p2, Mat &R, Mat &T, Mat &mask)
{

	double focal_length = 0.5 * (K.at<double>(0) + K.at<double>(4));
	Point2d principle_point(K.at<double>(2), K.at<double>(5));

	// Mat affinetrans(3,4,CV_64FC1);
	//    estimateAffine3D(P1, P2, affinetrans, mask, 3, 0.99);

	// double feasible_count = countNonZero(mask);
	// cout << (int)feasible_count << " -in- " << P1.size() << endl;

	// if (feasible_count <= 2 || (feasible_count / P1.size()) < 0.6)
	// 	return false;

	// R = affinetrans.colRange(0,3);
	// T = affinetrans.colRange(3,4);
	Mat E = findEssentialMat(p1, p2, focal_length, principle_point, RANSAC, 0.999, 1.0, mask);
	if (E.empty())
		return false;

	double feasible_count = countNonZero(mask);
	cout << (int)feasible_count << " -in- " << p1.size() << endl;
	//����RANSAC���ԣ�outlier��������50%ʱ�������ǲ��ɿ���
	if (feasible_count <= 2 || (feasible_count / p1.size()) < 0.6)
		return false;

	//�ֽⱾ�����󣬻�ȡ���Ա任
	int pass_count = recoverPose(E, p1, p2, R, T, focal_length, principle_point, mask);

	//ͬʱλ����������ǰ���ĵ�������Ҫ�㹻��
	if (((double)pass_count) / feasible_count < 0.7)
		return false;

	return true;
}

bool find_transform1(Mat &K, vector<Point3f> &P1, vector<Point3f> &P2, Mat &R, Mat &T, Mat &mask)
{

	double focal_length = 0.5 * (K.at<double>(0) + K.at<double>(4));
	Point2d principle_point(K.at<double>(2), K.at<double>(5));

	Mat affinetrans(3, 4, CV_64FC1);
	estimateAffine3D(P1, P2, affinetrans, mask, 3, 0.99);

	double feasible_count = countNonZero(mask);
	cout << (int)feasible_count << " -in- " << P1.size() << endl;

	if (feasible_count <= 2 || (feasible_count / P1.size()) < 0.6)
		return false;

	R = affinetrans.colRange(0, 3);
	T = affinetrans.colRange(3, 4);
	// Mat E = findEssentialMat(p1, p2, focal_length, principle_point, RANSAC, 0.999, 1.0, mask);
	// if (E.empty()) return false;

	// double feasible_count = countNonZero(mask);
	// cout << (int)feasible_count << " -in- " << p1.size() << endl;
	// //����RANSAC���ԣ�outlier��������50%ʱ�������ǲ��ɿ���
	// if (feasible_count <= 2 || (feasible_count / p1.size()) < 0.6)
	// 	return false;

	// //�ֽⱾ�����󣬻�ȡ���Ա任
	// int pass_count = recoverPose(E, p1, p2, R, T, focal_length, principle_point, mask);

	// //ͬʱλ����������ǰ���ĵ�������Ҫ�㹻��
	// if (((double)pass_count) / feasible_count < 0.7)
	// 	return false;

	return true;
}

// bool comp(float n)
// {
// 	float threshold = 5;
// 	cin >> n;
// 	return n < threshold;
// }
bool comp(double & a) { return a < 5; }
// bool find_transform_homography(Mat &K, vector<Point2f> &p1, vector<Point2f> &p2, Mat R1, Mat T1, Mat &best_r, Mat &best_t)
// {
// 	vector<float> errors;
// 	vector<Point3d> Sstructure;
// 	Mat homography_matrix;
// 	homography_matrix = findHomography(p1, p2, RANSAC, 3);
// 	// cout << "homography_matrix is " << endl
// 	// 	 << homography_matrix << endl;
// 	vector<Mat> r, t, n;
// 	decomposeHomographyMat(homography_matrix, K, r, t, n);
// 	// cout << "========Homography========" << endl;
// 	int best_inliers_num = 0;
// 	for (int i = 0; i < r.size(); i++)
// 	{

// 		if (t[i].at<double>(2,0) > 0)
// 		{
// 			errors.clear();
// 		    for (int j=0; j<p2.size(); j++)
// 			{ 
// 				reconstruct(K, R1, T1, r[i], t[i], p1[j], p2[j], Sstructure);

// 				float residuals = findResidualsforRT(K, r[i], t[i], p2[j], Sstructure);

// 				errors.push_back(residuals);

// 			}
// 		    int inliers_num = count_if(errors.begin(), errors.end(), comp);
//             cout<<"inliers for homology "<<i<<" "<<inliers_num<<endl;
// 			if (inliers_num > best_inliers_num)
// 			{
// 				best_r = r[i];
// 				best_t = t[i];
// 				best_inliers_num = inliers_num;
			
// 		     }  
// 		}

// 	//    return true ;
// 	}
// 	// best_t = t[3];
// 	// best_r = r[3];

// }

bool find_transform_homography(Mat &K, vector<Point2f> &p1, vector<Point2f> &p2, Mat R1, Mat T1, Mat &best_r, Mat &best_t)
{
	vector<double> residuals;
	vector<Point3d> Sstructure;
	Mat homography_matrix;
	homography_matrix = findHomography(p1, p2, RANSAC, 3);
	// cout << "homography_matrix is " << endl
	// 	 << homography_matrix << endl;
	vector<Mat> r, t, n;
	decomposeHomographyMat(homography_matrix, K, r, t, n);
	// cout << "========Homography========" << endl;
	int best_inliers_num = 0;
	for (int i = 0; i < r.size(); i++)
	{

		if (t[i].at<double>(2,0) > 0)
		{
			reconstruct(K, R1, T1, r[i], t[i], p1, p2, Sstructure);
			findResidualsforRT(K, r[i], t[i], p2, Sstructure, residuals);

		    int inliers_num = count_if(residuals.begin(), residuals.end(), comp);
            
			if (inliers_num > best_inliers_num)
			{
				best_r = r[i];
				best_t = t[i];
				best_inliers_num = inliers_num;
		    }  
		}
	}
}

//get_matched_points(key_points_for_all[i], key_points_for_all[i + 1], matches_for_all[i], p1, p2);
void get_matched_points( //根据matches,返回匹配时两张图分别的坐标
	vector<keypoint> &p1,
	vector<keypoint> &p2,
	vector<DMatch> matches,
	vector<Point2f> &out_p1,
	vector<Point2f> &out_p2,
	vector<Point3f> &out_p3,
	vector<Point3f> &out_p4)
{
	out_p1.clear();
	out_p2.clear();
	out_p3.clear();
	out_p4.clear();
	for (int i = 0; i < matches.size(); ++i)
	{
		out_p1.push_back(p1[matches[i].queryIdx].pt);
		out_p2.push_back(p2[matches[i].trainIdx].pt);
		out_p3.push_back(p1[matches[i].queryIdx].Pt);
		out_p4.push_back(p2[matches[i].trainIdx].Pt);
	}
}

void get_matched_colors(
	vector<Vec3b> &c1,
	vector<Vec3b> &c2,
	vector<DMatch> matches,
	vector<Vec3b> &out_c1,
	vector<Vec3b> &out_c2)
{
	out_c1.clear();
	out_c2.clear();
	for (int i = 0; i < matches.size(); ++i)
	{
		out_c1.push_back(c1[matches[i].queryIdx]);
		out_c2.push_back(c2[matches[i].trainIdx]);
	}
}

void maskout_points(vector<Point2f> &p1, Mat &mask)
{
	vector<Point2f> p1_copy = p1;
	p1.clear();

	for (int i = 0; i < mask.rows; ++i)
	{
		if (mask.at<uchar>(i) > 0)
			p1.push_back(p1_copy[i]);
	}
}

void maskout_colors(vector<Vec3b> &p1, Mat &mask)
{
	vector<Vec3b> p1_copy = p1;
	p1.clear();

	for (int i = 0; i < mask.rows; ++i)
	{
		if (mask.at<uchar>(i) > 0)
			p1.push_back(p1_copy[i]);
	}
}

void save_structure(string file_name, vector<Mat> &rotations, vector<Mat> &motions, vector<Point3d> &structure, vector<Vec3b> &colors)
{
	int n = (int)rotations.size();

	FileStorage fs(file_name, FileStorage::WRITE);
	fs << "Camera Count" << n;
	fs << "Point Count" << (int)structure.size();

	fs << "Rotations"
	   << "[";
	for (size_t i = 0; i < n; ++i)
	{
		fs << rotations[i];
	}
	fs << "]";

	fs << "Motions"
	   << "[";
	for (size_t i = 0; i < n; ++i)
	{
		fs << motions[i];
	}
	fs << "]";

	fs << "Points"
	   << "[";
	for (size_t i = 0; i < structure.size(); ++i)
	{
		fs << structure[i];
	}
	fs << "]";

	fs << "Colors"
	   << "[";
	for (size_t i = 0; i < colors.size(); ++i)
	{
		fs << colors[i];
	}
	fs << "]";

	fs.release();
}
//get_objpoints_and_imgpoints(
// matches_for_all[i],
// correspond_struct_idx[i],
// structure,
// key_points_for_all[i+1],
// object_points,
// image_points
// );
void get_objpoints_and_imgpoints(
	vector<DMatch> &matches, //从第二张图开始，每两张图的matches，其中包含很多个match_features
	vector<int> &struct_indices,
	vector<Point3d> &structure,
	vector<keypoint> &key_points, //后一张图的key points
	vector<Point3f> &object_points,
	vector<Point3f> &threeD_points,
	vector<Point2f> &image_points)
{
	object_points.clear();
	threeD_points.clear();
	image_points.clear();

	for (int i = 0; i < matches.size(); ++i)
	{
		int query_idx = matches[i].queryIdx; //在相应的图中是第几个feature point
		int train_idx = matches[i].trainIdx;

		int struct_idx = struct_indices[query_idx]; //访问对应列，输出为第几个有用的match，一个有用的match能组成一个U
		if (struct_idx < 0)
			continue;

		object_points.push_back(structure[struct_idx]);	//输出该次匹配有用的match的3D坐标
		image_points.push_back(key_points[train_idx].pt);  //输出有用的match在最新的图上match的坐标
		threeD_points.push_back(key_points[train_idx].Pt); //输出有用的match在最新的图上match的坐标
	}
}
// fusion_structure(
// 	matches_for_all[i],
// 	correspond_struct_idx[i],
// 	correspond_struct_idx[i + 1],
// 	structure,
// 	next_structure,
// 	colors,
// 	c1
// 	);
void fusion_structure(
	vector<DMatch> &matches,
	vector<int> &struct_indices,
	vector<int> &next_struct_indices,
	vector<Point3d> &structure,
	vector<Point3d> &next_structure,
	vector<Vec3b> &colors,
	vector<Vec3b> &next_colors)
{
	for (int i = 0; i < matches.size(); ++i)
	{
		int query_idx = matches[i].queryIdx;
		int train_idx = matches[i].trainIdx;

		int struct_idx = struct_indices[query_idx]; //如果两者匹配，假如前一张图的匹配点能构成3D点，后一张图匹配点应该属于同一个3D点
		if (struct_idx >= 0)
		{
			next_struct_indices[train_idx] = struct_idx;
			// continue;
		}
		structure.push_back(next_structure[i]);
		colors.push_back(next_colors[i]);
		struct_indices[query_idx] = next_struct_indices[train_idx] = structure.size() - 1;
	}
}

void init_structure( //初始化
	Mat K,
	vector<vector<keypoint>> &key_points_for_all,
	vector<vector<Vec3b>> &colors_for_all,
	vector<vector<DMatch>> &matches_for_all,
	vector<Point3d> &structure,
	vector<vector<int>> &correspond_struct_idx,
	vector<Vec3b> &colors,
	vector<Mat> &rotations,
	vector<Mat> &motions,
	vector<Mat> &camera_cor)
{

	vector<cv::Point2f> p1, p2;
	vector<Point3f> P1, P2; //3D points for the two cameras
	vector<Vec3b> c2;
	Mat R, T;
	Mat R0 = Mat::eye(3, 3, CV_64FC1); //第一张图的P
	Mat T0 = Mat::zeros(3, 1, CV_64FC1);
	Mat mask;
	get_matched_points(key_points_for_all[0], key_points_for_all[1], matches_for_all[0], p1, p2, P1, P2);
	get_matched_colors(colors_for_all[0], colors_for_all[1], matches_for_all[0], colors, c2);
 	find_transform_homography(K, p1, p2, R0, T0, R, T);
 	reconstruct(K, R0, T0, R, T, p1, p2, structure);
 	rotations.push_back(R0);
    rotations.push_back(R);
    motions.push_back(T0);
    motions.push_back(T);
    Mat camera_cor01 = (Mat_<double>(3, 1) << 0,0,0);
	Mat camera_cor02 = R*camera_cor01+T;
	camera_cor.push_back(camera_cor01);
  	camera_cor.push_back(camera_cor02);


//将correspond_struct_idx变为key_points_for_all的形式
	correspond_struct_idx.clear();
	correspond_struct_idx.resize(key_points_for_all.size());
	for (int i = 0; i < key_points_for_all.size(); ++i)
	{
	correspond_struct_idx[i].resize(key_points_for_all[i].size(), -1);//与key points的size保持一致。为什么要加-1？
	}	

	int idx = 0;
	vector<DMatch>& matches = matches_for_all[0];//将matches_for_all的第一个值赋给matches
	for (int i = 0; i < matches.size(); ++i)
	{
	// if (mask.at<uchar>(i) == 0)//如果这一对match是outlier
		// continue;//如果mask的值等于0的话，继续从头开始循环，否则往下继续
    correspond_struct_idx[0][matches[i].queryIdx] = idx;//将第idx个match,即idx，存入correspond_struct_idx中，存的位置为：如果是前一张，则是第一行，后一张则是第二行，纵坐标对应点在图重视第几个feature point
    correspond_struct_idx[1][matches[i].trainIdx] = idx;
    ++idx;
	}
}

/////////////////////////////////////////////////////////////////////////
//////////////// Bundle Adjustment-Google ceres//////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

// 定义代价函数
struct ReprojectCost
{
    cv::Point2d observation;

    ReprojectCost(cv::Point2d& observation)
        : observation(observation)
    {
    }
   //使用模板的目的就是能够让程序员编写与类型无关的代码
  //AutoDiffCostFunction<ReprojectCost, 2, 4, 6, 3>
    template <typename T>
    bool operator()(const T* const intrinsic, const T* const extrinsic, const T* const pos3d, T* residuals) const//pos3d:对应的3D点；observation:相应的图片的坐标，
   //通过对3D点来计算2D坐标值与实际值比较，检查R，T的正确性
    {
        const T* r = extrinsic;//外参 R
        const T* t = &extrinsic[3];//T

        T pos_proj[3];//定义一个长度为4的pos_proj

       AngleAxisRotatePoint(r, pos3d, pos_proj);//y=r(angle_axis)pos3d,根据R进行旋转变化

        // Apply the camera translation
        pos_proj[0] += t[0];//平移变化
        pos_proj[1] += t[1];
        pos_proj[2] += t[2];

        const T x = pos_proj[0] / pos_proj[2];//求x和y
        const T y = pos_proj[1] / pos_proj[2];

        const T fx = intrinsic[0];//读取内参矩阵
        const T fy = intrinsic[1];
        const T cx = intrinsic[2];
        const T cy = intrinsic[3];

        // Apply intrinsic
        const T u = fx * x + cx;//这是啥？
        const T v = fy * y + cy;

        residuals[0] = u - T(observation.x);//反向投影误差
        residuals[1] = v - T(observation.y);

        return true;
    }
};
//使用
 //Solver求解BA，其中使用了Ceres提供的Huber函数作为损失函数
void bundle_adjustment(
    cv::Mat& intrinsic,
    vector<cv::Mat>& extrinsics,
    vector<vector<int> >& correspond_struct_idx,
    vector<vector<keypoint> >& key_points_for_all,
    vector<cv::Point3d>& structure
)
{
    Problem problem;

    // load extrinsics (rotations and motions)
    for (size_t i = 0; i < extrinsics.size(); ++i)
    {
        problem.AddParameterBlock(extrinsics[i].ptr<double>(), 6);//Add a parameter block with appropriate size and parameterization to the problem.
       //Repeated calls with the same arguments are ignored. Repeated calls with the same double pointer but a different size results in undefined behavior.
    }
    // fix the first camera.
    problem.SetParameterBlockConstant(extrinsics[0].ptr<double>());//Hold the indicated parameter block constant during optimization.保持第一个外惨矩阵不变

    // load intrinsic
    problem.AddParameterBlock(intrinsic.ptr<double>(), 4); // fx, fy, cx, cy

    // load points
    LossFunction* loss_function = new HuberLoss(4);   // loss function make bundle adjustment robuster.
    for (size_t img_idx = 0; img_idx < correspond_struct_idx.size(); ++img_idx)
    {
        vector<int>& point3d_ids = correspond_struct_idx[img_idx];
        vector<keypoint>& key_points = key_points_for_all[img_idx];
        for (size_t point_idx = 0; point_idx < point3d_ids.size(); ++point_idx)
        {
            int point3d_id = point3d_ids[point_idx];
            if (point3d_id < 0)
                continue;

            cv::Point2d observed = key_points[point_idx].pt;//corresponding 2D points coordinates with feasible 3D point
            // 模板参数中，第一个为代价函数的类型，第二个为代价的维度，剩下三个分别为代价函数第一第二还有第三个参数的维度
            CostFunction* cost_function = new AutoDiffCostFunction<ReprojectCost, 2, 4, 6, 3>(new ReprojectCost(observed));
            //向问题中添加误差项
            problem.AddResidualBlock(//adds a residual block to the problem,implicitly adds the parameter blocks(This causes additional correctness checking) if they are not present
                cost_function,
                loss_function,
                intrinsic.ptr<double>(),            // Intrinsic
                extrinsics[img_idx].ptr<double>(),  // View Rotation and Translation
                &(structure[point3d_id].x)          // Point in 3D space
            );
        }
    }

    // Solve BA
    Solver::Options ceres_config_options;
    ceres_config_options.minimizer_progress_to_stdout = false;
    ceres_config_options.logging_type = SILENT;
    ceres_config_options.num_threads = 1;//Number of threads to be used for evaluating the Jacobian and estimation of covariance.
    ceres_config_options.preconditioner_type = JACOBI;
   ceres_config_options.linear_solver_type = DENSE_SCHUR;
    // ceres_config_options.linear_solver_type = ceres::SPARSE_SCHUR;//ype of linear solver used to compute the solution to the linear least squares problem in each iteration of the Levenberg-Marquardt algorithm
    // ceres_config_options.sparse_linear_algebra_library_type = ceres::EIGEN_SPARSE;

    Solver::Summary summary;
    Solve(ceres_config_options, &problem, &summary);

    if (!summary.IsSolutionUsable())
    {
        std::cout << "Bundle Adjustment failed." << std::endl;
    }
    else
    {
        // Display statistics about the minimization
        std::cout << std::endl
            << "Bundle Adjustment statistics (approximated RMSE):\n"
            << " #views: " << extrinsics.size() << "\n"
            << " #residuals: " << summary.num_residuals << "\n"
            << " Initial RMSE: " << std::sqrt(summary.initial_cost / summary.num_residuals) << "\n"
            << " Final RMSE: " << std::sqrt(summary.final_cost / summary.num_residuals) << "\n"
            << " Time (s): " << summary.total_time_in_seconds << "\n"
            << std::endl;
    }
}

// void get_file_names(string dir_name, vector<string> & names)//读取文件列表的库，最后得到文件列表
// {
// 	names.clear();
// 	tinydir_dir dir;
// 	tinydir_open(&dir, dir_name.c_str());

// 	while (dir.has_next)
// 	{
// 		tinydir_file file;
// 		tinydir_readfile(&dir, &file);
// 		if (!file.is_dir)
// 		{
// 			names.push_back(file.path);
// 		}
// 		tinydir_next(&dir);
// 	}
// 	tinydir_close(&dir);
// }

int main(int argc, char **argv)
{
	// vector<string> img_names;
	// get_file_names("images", img_names);//将images里面的数据读到img_names里面

	// //��������
	// Mat K(Matx33d(
	// 	2759.48, 0, 1520.69,
	// 	0, 2764.16, 1006.81,
	// 	0, 0, 1));

	// vector<vector<KeyPoint>> key_points_for_all;
	// vector<Mat> descriptor_for_all;
	// vector<vector<Vec3b>> colors_for_all;
	// vector<vector<DMatch>> matches_for_all;
	// //��ȡ����ͼ��������
	// extract_features(img_names, key_points_for_all, descriptor_for_all, colors_for_all);//收集到所有图片的key points和descriptors
	// match_features(descriptor_for_all, matches_for_all);//matches_for_all储存所有matches的地方

	// vector<Point3f> structure;
	// vector<vector<int>> correspond_struct_idx;
	// vector<Vec3b> colors;
	// vector<Mat> rotations;
	// vector<Mat> motions;

	//初始化structure

	int start = stoi(argv[1]);
	int end = stoi(argv[2]);
	Mat K(Matx33d(
		350.6847, 0, 332.4661,
		0, 350.0606, 163.7461,
		0, 0, 1));
	// Mat K(Matx33d(
	// 	349.891, 0, 318.852,
	// 	0, 349.891, 180.437,
	// 	0, 0, 1));
	Mat yellow = (Mat_<double>(1, 3) << 0, 255, 255);
	Mat blue = (Mat_<double>(1, 3) << 255, 0, 0);
	Mat orange = (Mat_<double>(1, 3) << 0, 0, 255);

	vector<vector<keypoint>> key_points_for_all;
	//cout<<"typebefore"<<typeid(key_points_for_all).name()<<endl;
	vector<vector<Vec3b>> colors_for_all;
	vector<vector<DMatch>> matches_for_all;
	vector<Mat> camera_cor;

	int imgId = 0;
	for (int i = start; i <= end; i++)
	{
		vector<keypoint> feature;
		vector<cv::Vec3b> colors;
		ifstream csvPath("result/" + to_string(i) + "_stereo.csv");
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
			getline(liness, X, ',');
			getline(liness, Y, ',');
			getline(liness, Z, ',');

			// cv::circle(img, cv::Point (stoi(x),stoi(y)), 3, cv::Scalar (0,0,0), CV_FILLED);
			if (label == "blue")
			{
				labelId = 0;
				colors.push_back(blue);
			}
			if (label == "yellow")
			{
				labelId = 1;
				colors.push_back(yellow);
			}
			if (label == "orange")
			{
				labelId = 2;
				colors.push_back(orange);
			}
			Point2f pt(stof(x), stof(y));
			Point3f Pt(stof(X), stof(Y), stof(Z));
			keypoint keypoint1 = {pt, labelId, Pt};
			feature.push_back(keypoint1);
		}
		// namedWindow("img", WINDOW_NORMAL);
		// imshow("img", img);
		// waitKey(0);
		key_points_for_all.push_back(feature);
		colors_for_all.push_back(colors);
		// cout<<"type"<<typeid(key_points_for_all).name()<<endl;
		if (imgId > 0)
		{
			vector<DMatch> matched;
			matchFeatures(imgId, key_points_for_all[imgId - 1], key_points_for_all[imgId], matched);
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

	// // matchFeaturesForAll(key_points_for_all, matches_for_all);
	// //cout<<matches_for_all[0][0].queryIdx;
	// // vector<Point2f> p1;
	// // vector<Point2f> p2;
	vector<Point3d> structure;
	vector<vector<int>> correspond_struct_idx;
	vector<Vec3b> colors;
	vector<Mat> rotations;
	vector<Mat> motions;
	init_structure( //此时已做完第一张图和第二张图的重建，rotations和motions里放了两张图的R和T
		K,
		key_points_for_all,
		colors_for_all,
		matches_for_all,
		structure,
		correspond_struct_idx,
		colors,
		rotations,
		motions,
		camera_cor);


	for (int i = 1; i < matches_for_all.size(); ++i)//遍历，从第二张图和第三张图开始，每次两张图的match
	{
	
		vector<Point3f> object_points;//3D points
		vector<Point2f> image_points;
		vector<Point3f> threeD_points;
		Mat r, R, T;
		Mat mask;

		//输出本次遍历中两张图有用的match的3D坐标和新图片上的2D坐标
		get_objpoints_and_imgpoints(
			matches_for_all[i],
			correspond_struct_idx[i],
			structure,
			key_points_for_all[i+1],
			object_points,
			threeD_points,
			image_points);
		vector<Point3f> P1, P2;
		vector<Point2f> p1, p2;
		vector<Vec3b> c1, c2;
		get_matched_points(key_points_for_all[i], key_points_for_all[i+1], matches_for_all[i], p1, p2, P1, P2);
		get_matched_colors(colors_for_all[i], colors_for_all[i+1], matches_for_all[i], c1, c2);
	    // find_transform1(K, P1, P2, R, T, mask);//求得第二张图的P
	    find_transform_homography(K, p1, p2, rotations.back(), motions.back(), R, T);


// 		//bool solvePnPRansac(InputArray objectPoints, InputArray imagePoints, InputArray cameraMatrix, InputArray distCoeffs, OutputArray rvec, OutputArray tvec,
// 		// bool useExtrinsicGuess=false, int iterationsCount=100, float reprojectionError=8.0, double confidence=0.99, OutputArray inliers=noArray(), int flags=SOLVEPNP_ITERATIVE )
// 		//rvec – Output rotation vector (see Rodrigues() ) that, together with tvec , brings points from the model coordinate system to the camera coordinate system.
// 		//tvec – Output translation vector.
		// solvePnPRansac(object_points, image_points, K, noArray(), r, T);

		// Rodrigues(r, R);//Converts a rotation matrix to a rotation vector or vice versa.

		Mat camera_cor02 = R*camera_cor.back()+T;
		camera_cor.push_back(camera_cor02);
		// //得到最新一张图的R和T
		rotations.push_back(R);
		motions.push_back(T);

// //求3D点
		vector<Point3d> next_structure;
		reconstruct(K, rotations[i], motions[i], R, T, p1, p2, next_structure);//重建新的图
		fusion_structure(
			matches_for_all[i],
			correspond_struct_idx[i],
			correspond_struct_idx[i + 1],
			structure,
			next_structure,
			colors,
			c1);

	}

    google::InitGoogleLogging(argv[0]);
	cv::Mat intrinsic(cv::Matx41d(K.at<double>(0, 0), K.at<double>(1, 1), K.at<double>(0, 2), K.at<double>(1, 2)));
	vector<cv::Mat> extrinsics;
	for (size_t i = 0; i < rotations.size(); ++i)
	{
	cv::Mat extrinsic(6, 1, CV_64FC1);

	cv::Mat r;
	Rodrigues(rotations[i], r);

	r.copyTo(extrinsic.rowRange(0, 3));
	motions[i].copyTo(extrinsic.rowRange(3, 6));

	 extrinsics.push_back(extrinsic);
	 }
	bundle_adjustment(intrinsic, extrinsics, correspond_struct_idx, key_points_for_all, structure);

int resultSize = 1000;
float resultResize = 100;
Mat result = Mat::zeros(resultSize, resultSize, CV_8UC3);
for(int u=0; u<structure.size(); u++){
	cout<<"3D points"<<structure[u]<<colors[u]<<endl;
	int x = int(structure[u].x * resultResize+resultSize/3);
	int y = int(structure[u].z * resultResize+resultSize/2);
	if (x >= 0 && x <= resultSize && y>= 0 && y <= resultSize){
	circle(result, Point (x,y), 3, Scalar (colors[u]), CV_FILLED);
	}
}
for(int u=0; u<camera_cor.size(); u++){
	cout<<"camera cor"<<""<<camera_cor[u]<<endl;
	int x = int(camera_cor[u].at<double>(0,0) * resultResize+resultSize/2);
	int y = int(camera_cor[u].at<double>(2,0) * resultResize);
	if (x >= 0 && x <= resultSize && y>= 0 && y <= resultSize){
	circle(result, Point (x,y), 5, Scalar (255,255,255), CV_FILLED);
    }
}
flip(result, result, 0);
namedWindow("result", WINDOW_NORMAL);
imshow("result", result);
waitKey(0);
save_structure("structure444.yml", rotations, motions, structure, colors);
cout<<"done"<<endl;
}