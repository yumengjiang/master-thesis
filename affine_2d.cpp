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

#define PI 3.14159265

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
	matched.clear();
	float mThreshold = 1;
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
			cout << index << " " << i << " " << minRes << endl;
		} 
	}
}

void matchFeaturesAffine(Mat affine, vector<KP> featureLast, 
                  vector<KP> featureNext, vector<DMatch> &matched){
	float mThreshold = 0.2;
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
	vector<KP>& p1,
	vector<KP>& p2,
	vector<Vec3b>& c1,
	vector<Vec3b>& c2,
	vector<DMatch>& matches,
	vector<Point3d>& out_p1,
	vector<Point3d>& out_p2,
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

void maskout_points(vector<Point3d>& p, Mat& mask)
{
	vector<Point3d> p_copy = p;
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
void reprojectionErrors(Mat affine_homo, vector<Point2d> p1, vector<Point2d> p2, double& errors){
	vector<Point3d> p2homo;
	convertPointsToHomogeneous(p2, p2homo);

    errors = 0;
	for(int i=0; i<p1.size();i++){
		Mat p1est;
		p1est=affine_homo.inv()*Mat(p2homo[i]);
		errors += pow(pow(p1est.at<double>(0,0)-p1[i].x,2)+pow(p1est.at<double>(1,0)-p1[i].y,2),0.5f);
	}
	
}
bool m_compare(double error)
{
    return error<0.08;
}
void estimateTransform2D(vector<Point2d> p1, vector<Point2d> p2, Mat& best_affine, double& final_error){
	if(p1.size()<2){
		cout << "Too few points to estimate transformation!" << endl;
		return;
	}
	best_affine.release();
	//x1:p1[i].x
	//y1:p1[i].y
	//p1:p2[i].x
	//q1:p2[i].y 
	//x2:p1[i+1].x
	//y2:p1[i+1].y
	//p2"p2[i+1].x
	//Mat M(2*p1.size(),4,CV_64F), B(2*p1.size(),1,CV_64F);
	// double most_inliers=0;
	double min_errors = 100000;
	double alphathreshold = 0.088886;

	for(int i = 0; i < p1.size()-1; i++){
		for(int k = i+1; k<p1.size();k++){
		vector<double> alpha;
		//vector<Mat>affine_forransac-temp;
		double a= pow(p1[k].x-p1[i].x,2)+pow(p1[i].y-p1[k].y,2);
		double b=2*(p2[i].x-p2[k].x)*(p1[i].y-p1[k].y);
		double c= pow(p2[k].x-p2[i].x,2)-pow(p1[k].x-p1[i].x,2);
		alpha.push_back(asin((-b+pow(pow(b,2)-4*a*c,0.5f))/(2*a)));
		alpha.push_back(asin((-b-pow(pow(b,2)-4*a*c,0.5f))/(2*a)));
		for(int j=0; j<2;j++){
			if(alpha[j]>alphathreshold||alpha[j]<-alphathreshold||isnan(alpha[j])){continue;}
			// Mat affine_single;
			double cosa=cos(alpha[j]);
				double errors;
				double tx=p2[i].x-cosa*p1[i].x+sin(alpha[j])*p1[i].y;
			     
				double ty=p2[i].y-sin(alpha[j])*p1[i].x-cosa*p1[i].y;
				 
				Mat affine_single(Matx33d(cosa,-sin(alpha[j]),tx,sin(alpha[j]),cosa,ty,0,0,1));
				// affine_single.at<double>(0,1)=-sina[j];
				// affine_single.at<double>(0,2)=tx;
				// affine_single.at<double>(1,0)=sina[j];
				// affine_single.at<double>(1,1)=cosa;
				// affine_single.at<double>(1,2)=ty;

				//cout<<"starts p"<<i<<"loop"<<j<<endl;
				reprojectionErrors(affine_single, p1, p2, errors);
				// cout<<"done p"<<i<<"loop"<<j<<endl;
				// int inlierscount_if(errors.begin(),errors.end(),m_compare);
				// cout<<"inliers size"<<inliers<<endl;
				// if (inliers>most_inliers){
				// 	most_inliers=inliers;
				// 	best_affine=affine_single;
				// }
				if (errors<min_errors){
					min_errors=errors;
					best_affine=affine_single;
				}
			}
		}
	}
	final_error=min_errors;

}
// best_affine.at<double>(0,2)+=0.5*best_affine.at<double>(1,0);
// best_affine.at<double>(1,2)-=0.5*best_affine.at<double>(0,0);

		
	 



void reconstruct(//初始化
	vector<KP>& last_keypoints,
	vector<KP>& next_keypoints,
	vector<Vec3b>& last_colors,
	vector<Vec3b>& next_colors,
	vector<DMatch>& matches,
	vector<Vec3b>& c1,
	vector<Point3d>& p2,
	Mat& affine_tmp,
	double& final_error
	)
{   
	vector<Point3d> p1;
	vector<Point2d> p3, p4;
	vector<double> res;
	vector<Vec3b> c2;
	get_matched_points(last_keypoints, next_keypoints, last_colors, next_colors, matches, p1, p2, c1, c2);

	// Mat mask;

	for(int i = 0; i < p1.size(); i++){
		p3.push_back(Point2d(p1[i].x,p1[i].y));
		p4.push_back(Point2d(p2[i].x,p2[i].y));
	}
	estimateTransform2D(p3, p4, affine_tmp,final_error);
	cout << "affine: " << affine_tmp << endl;
	// affine_tmp = estimateRigidTransform(p3, p4, false);
	if (affine_tmp.rows == 0){
    	cout << "Fail to estimate affine transformation, number of points: " << p3.size() << endl;
    }
 //    affine_tmp /= pow(pow(affine_tmp.at<double>(0,0),2)+pow(affine_tmp.at<double>(0,1),2),0.5);
 //    affine_tmp.convertTo(affine.rowRange(0,2),CV_64F);
}

void init_structure(//初始化
	vector<vector<KP>>& keypoints_for_all,
	vector<vector<Vec3b>>& colors_for_all,
	//vector<vector<DMatch>>& matches_for_all,
	vector<Point3d>& structure,
	vector<Vec3b>& colors,
	vector<vector<int>>& correspond_struct_idx,
	vector<Mat>& affines,
	double& init_last_img
	)
{   
	vector<DMatch> matched,matched1,matched2;
	Mat affine;
	vector<Vec3b> colors1, colors2;
	vector<Point3d> p2;
	double final_error1,final_error2;
	double affine_threshold=0.8;
	//reconstruct(keypoints_for_all[0], keypoints_for_all[1], colors_for_all[0], colors_for_all[1], matches_for_all[0], colors, p2);
    // matchFeaturesAffine(affine, keypoints_for_all[0], keypoints_for_all[1], matches_for_all[0]);
    matchFeatures(keypoints_for_all[0],keypoints_for_all[1], matched1);
    reconstruct(keypoints_for_all[0], keypoints_for_all[1], colors_for_all[0], colors_for_all[1], matched1, colors, p2, affine, final_error1);
    
    //cout<<"res 2"<<" "<<final_error2<<endl;
    if (final_error1>affine_threshold){
    	matchFeatures(keypoints_for_all[0],keypoints_for_all[2], matched2);
    	reconstruct(keypoints_for_all[0], keypoints_for_all[2], colors_for_all[0], colors_for_all[2], matched2, colors, p2, affine, final_error2);
    	if (final_error2>final_error1){
    		matched=matched1;
    		init_last_img=1;
    		//system ("pause");
    		//return 0; 
    	}
    	else{
    		matched=matched2;
    		init_last_img=2;
    	}
    }
    else{
    	matched=matched1;
    	init_last_img=1;
    }
    

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
	vector<DMatch>& matches = matched;//将matches_for_all的第一个值赋给matches
	for (int i = 0; i < matches.size(); ++i)
	{
		// if (mask.at<uchar>(i) == 0)//如果这一对match是outlier
		// 	continue;//如果mask的值等于0的话，继续从头开始循环，否则往下继续

		correspond_struct_idx[0][matches[i].queryIdx] = idx;//将第idx个match,即idx，存入correspond_struct_idx中，存的位置为：如果是前一张，则是第一行，后一张则是第二行，纵坐标对应点在图重视第几个feature point
		correspond_struct_idx[init_last_img][matches[i].trainIdx] = idx;
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
	//vector<vector<DMatch>> matches_for_all;
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
	        if(stod(Z)<5){
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
		// if (imgId > 0){
		// 	vector<DMatch> matched;
		// 	matchFeatures(keypoints_for_all[imgId-1], keypoints_for_all[imgId], matched);
		// 	matches_for_all.push_back(matched);

		// 	// imgLast = imread("result/"+to_string(imgId-1)+".png");
		// 	// imgNext = imread("result/"+to_string(imgId)+".png");
		// 	// resize(imgLast, imgLast, Size(320, 180));
		// 	// resize(imgNext, imgNext, Size(320, 180));
		// 	// drawMatches(imgLast, keypoints_for_all[imgId-1], imgNext, keypoints_for_all[imgId], matched, outImg);
		// 	// namedWindow("MatchSIFT", WINDOW_NORMAL);
		// 	// imshow("MatchSIFT",outImg);
		// 	// waitKey(0);
		// }
		imgId++;
	}

	vector<Point3d> structure;
	vector<Vec3b> colors;
	vector<vector<int>> correspond_struct_idx;
	double init_last_img;

	init_structure(//此时已做完第一张图和第二张图的重建，rotations和motions里放了两张图的R和T
		keypoints_for_all,
		colors_for_all,
		//matches_for_all,
		structure,
		colors,
		correspond_struct_idx,
		affines,
		init_last_img
		);

	for (int i = init_last_img; i < keypoints_for_all.size(); ++i)//遍历，从第二张图和第三张图开始，每次两张图的match
	{
	//int i=3;
		//Mat affine(Matx33d(0,0,0,0,0,0,0,0,1));
		vector<Point3d> p2;
		vector<Vec3b> c1;
		vector<DMatch>matched1,matched2,matched;
		double next_img;
		double affine_threshold=0.85;
		vector<Vec3b> c, c2;
		double final_error1, final_error2;
		matchFeatures(keypoints_for_all[i], keypoints_for_all[i+1], matched1);
		reconstruct(keypoints_for_all[i], keypoints_for_all[i+1], colors_for_all[i], colors_for_all[i+1], matched1, c, p2, affine, final_error1);
		if (final_error1>affine_threshold){
    	matchFeatures(keypoints_for_all[i],keypoints_for_all[i+2], matched2);
    	reconstruct(keypoints_for_all[i], keypoints_for_all[i+2], colors_for_all[i], colors_for_all[i+2], matched2, c2, p2, affine, final_error2);
    	if (final_error2>final_error1){
    		matched=matched1;
    		c1=c;
    		next_img=i+1;
    		//system ("pause");
    		//return 0; 
    	}
    	else{
    		matched=matched2;
    		c1=c2;
    		next_img=i+2;
    	}
    }
    else{
    	matched=matched1;
    	c1=c;
    	next_img=i+1;
    }cout<<"res 1"<<" "<<final_error1<<endl;



	    // matchFeaturesAffine(affine, keypoints_for_all[i], keypoints_for_all[i+1], matches_for_all[i]);
	    // reconstruct(keypoints_for_all[i], keypoints_for_all[i+1], colors_for_all[i], colors_for_all[i+1], matches_for_all[i], c1, p2, affine);

	    affine = affine*affines.back();
		affines.push_back(affine);

		vector<Point3d> next_structure;
		for(int i = 0; i < p2.size(); i++){
			next_structure.push_back(Point3d(Mat(affine.inv()*Mat(p2[i]))));
		}

		fusion_structure(
			matched1,
			correspond_struct_idx[i],
			correspond_struct_idx[next_img],
			structure,
			next_structure,
			colors,
			c1
			);
		cout<<"from"<<""<<i<<""<<"to"<<next_img<<endl;
	}
	
	vector<int> count_same_structure;
	count_same_structure.resize(structure.size());
	for (int i = 0; i < correspond_struct_idx.size(); ++i){
		for (int j = 0; j < correspond_struct_idx[i].size(); ++j){
			cout << correspond_struct_idx[i][j] << " ";
			for(int k = 0; k < structure.size(); k++){
				if(correspond_struct_idx[i][j] == k)
					count_same_structure[k]++;
			}
		}
		cout << "\n";
	}

	int resultSize = 1000;
	double resultResize = 50;
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
		Mat camera_cor(Matx31d(0,0.5,1));
		camera_cor = affines[i].inv() * camera_cor;
		if(i>0)
			cout << acos(affines[i].at<double>(0,0))-acos(affines[i-1].at<double>(0,0)) << endl;
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