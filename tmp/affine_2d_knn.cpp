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
#include "gflags/gflags.h"
#include "glog/logging.h"
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <ceres/problem.h>

using namespace cv;
using namespace std;

#define PI 3.14159265
double error_threshold = 0.7;
double match_threshold = 0.35;
double alpha_threshold = 1;
int rm_cone_threshold = 7;
vector<Scalar> COLOR = {{255,0,0},{0,255,255},{0,165,255},{0,0,255}};


struct KP
{
	Point3d pt;
	int id;
};

double computeResidual(Point3d pt1, Point3d pt2){
	return pow((pow ((pt1.x-pt2.x),2) + pow ((pt1.y-pt2.y),2)),0.5);
}

void matchFeatures(vector<KP> featureLast, vector<KP> featureNext, Mat& query, Mat& train, vector<DMatch>& matches)
{
	int matchSize = 1000;
	double matchResize = 100;
	Mat match = Mat::zeros(matchSize, matchSize, CV_8UC3);

	vector<vector<DMatch>> knn_matches;
	BFMatcher matcher(NORM_L2);
	matcher.knnMatch(query, train, knn_matches, 2);

	//��ȡ����Ratio Test����Сƥ���ľ���
	float min_dist = FLT_MAX;
	for (int r = 0; r < knn_matches.size(); ++r)
	{
		//Ratio Test
		if (knn_matches[r][0].distance > 0.6*knn_matches[r][1].distance)
			continue;

		float dist = knn_matches[r][0].distance;
		if (dist < min_dist) min_dist = dist;
	}

	matches.clear();
	for (size_t r = 0; r < knn_matches.size(); ++r)
	{
		//�ų�������Ratio Test�ĵ���ƥ�����������ĵ�
		if (
			knn_matches[r][0].distance > 0.6*knn_matches[r][1].distance ||
			knn_matches[r][0].distance > 5 * max(min_dist, 10.0f)
			)
			continue;

		//����ƥ����
		matches.push_back(knn_matches[r][0]);
		int last_index = knn_matches[r][0].queryIdx;
		int next_index = knn_matches[r][0].trainIdx;
		int x = int(featureLast[last_index].pt.x * matchResize + matchSize/2);
		int y = int(featureLast[last_index].pt.y * matchResize + matchSize/4);
		int x1 = int(featureNext[next_index].pt.x * matchResize + matchSize/2);
		int y1 = int(featureNext[next_index].pt.y * matchResize + matchSize/4);
		circle(match, Point (x,y), 5, COLOR[featureNext[next_index].id], -1);
		circle(match, Point (x1,y1), 3, COLOR[featureNext[next_index].id], -1);
		line(match, Point(x, y), Point(x1, y1), Scalar(255, 255, 255), 1);
	}
	// flip(match, match, 0);
 //    namedWindow("match", WINDOW_NORMAL);
	// imshow("match", match);
	// waitKey(0);
}

void matchFeatures2(vector<KP> featureLast, vector<KP> featureNext, vector<DMatch> &matched){
	matched.clear();
	double res, minRes;
	int featureLastRow = featureLast.size();
	int featureNextRow = featureNext.size();
	int index;
	int matchSize = 1000;
	double matchResize = 100;
	Mat match = Mat::zeros(matchSize, matchSize, CV_8UC3);

	for(int i = 0; i < featureNextRow; i++){
		minRes = match_threshold;
		for(int j = 0; j < featureLastRow; j++){
		    if(featureNext[i].id == featureLast[j].id){
		        res = computeResidual(featureNext[i].pt, featureLast[j].pt);//check residuals, find the smallest one, save it
				if(res < minRes){
				    minRes = res;
				    index = j;
		    	}
			}
		}
		if(minRes < match_threshold){
			matched.push_back(DMatch(index,i,minRes));
			int x = int(featureLast[index].pt.x * matchResize + matchSize/4);
			int y = int(featureLast[index].pt.y * matchResize + matchSize/4);
			int x1 = int(featureNext[i].pt.x * matchResize + matchSize/4);
			int y1 = int(featureNext[i].pt.y * matchResize + matchSize/4);
			circle(match, Point (x,y), 5, COLOR[featureNext[i].id], -1);
			circle(match, Point (x1,y1), 3, COLOR[featureNext[i].id], -1);
			line(match, Point(x, y), Point(x1, y1), Scalar(255, 255, 255), 1);

			// cout << index << " " << i << " " << minRes << endl;
		} 
	}
	// flip(match, match, 0);
 //    namedWindow("match", WINDOW_NORMAL);
	// imshow("match", match);
	// waitKey(0);
}


// void matchFeaturesAffine(Mat affine, vector<KP> featureLast, 
//                   vector<KP> featureNext, vector<DMatch> &matched){
// 	double match_threshold = 0.1;
// 	double res, minRes;
// 	int featureLastRow = featureLast.size();
// 	int featureNextRow = featureNext.size();
// 	int index;
// 	matched.clear();

// 	for(int i = 0; i < featureNextRow; i++){
// 		minRes = match_threshold;
// 		for(int j = 0; j < featureLastRow; j++){
// 		    if(featureNext[i].id == featureLast[j].id){
// 		  	    Point3d affine_pt(Mat(affine*Mat(featureLast[j].pt))); 
// 		        res = computeResidual(featureNext[i].pt, affine_pt);//check residuals, find the smallest one, save it
// 		        if(res < minRes){
// 		            minRes = res;
// 		            index = j;
// 		        }
// 		    }
// 		}
// 		if(minRes < match_threshold){
// 		    matched.push_back(DMatch(index,i,minRes));
// 		    // cout << index << " " << i << " " << imageId << " " << minRes << endl;
// 		} 
// 	}
// }

void get_matched_points(//根据matched,返回匹配时两张图分别的坐标
	vector<KP>& p1,
	vector<KP>& p2,
	vector<Vec3b>& c1,
	vector<Vec3b>& c2,
	vector<DMatch>& matched,
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
	for (int i = 0; i < matched.size(); ++i)
	{
		out_p1.push_back(p1[matched[i].queryIdx].pt);
		out_p2.push_back(p2[matched[i].trainIdx].pt);
		out_c1.push_back(c1[matched[i].queryIdx]);
		out_c2.push_back(c2[matched[i].trainIdx]);
	}
}

void reprojectionErrors(Mat affine_tmp, vector<Point3d> p1, vector<Point3d> p2, double& error){
	for(int i=0; i<p1.size();i++){
		Mat p1est = affine_tmp.inv()*Mat(p2[i]);
		error += pow(pow(p1est.at<double>(0,0)-p1[i].x,2)+pow(p1est.at<double>(1,0)-p1[i].y,2),0.5f);
	}
}

// bool m_compare(double error)
// {
//     return error<0.08;
// }

void estimateTransform2D(vector<Point3d> p1, vector<Point3d> p2, Mat& best_affine, double& min_error){
	if(p1.size()<2){
		cout << "Too few points to estimate transformation!" << endl;
		return;
	}
	best_affine.release();
	min_error = 100;

	for(int i = 0; i < p1.size()-1; i++){
		for(int k = i+1; k<p1.size();k++){
			vector<double> alpha;
			//vector<Mat>affine_forransac-temp;
			double a = pow(p1[k].x-p1[i].x,2)+pow(p1[i].y-p1[k].y,2);
			double b = 2*(p2[i].x-p2[k].x)*(p1[i].y-p1[k].y);
			double c = pow(p2[k].x-p2[i].x,2)-pow(p1[k].x-p1[i].x,2);
			alpha.push_back(asin((-b+pow(pow(b,2)-4*a*c,0.5f))/(2*a)));
			alpha.push_back(asin((-b-pow(pow(b,2)-4*a*c,0.5f))/(2*a)));
			for(int j = 0; j < 2; j++){
				if(alpha[j]>alpha_threshold||alpha[j]<-alpha_threshold||isnan(alpha[j])){continue;}
				double cosa = cos(alpha[j]);
				double sina = sin(alpha[j]);
				double error = 0;
				double tx = p2[i].x-cosa*p1[i].x+sina*p1[i].y;
				double ty = p2[i].y-sina*p1[i].x-cosa*p1[i].y;
				Mat affine_tmp(Matx33d(cosa,-sina,tx,sina,cosa,ty,0,0,1));

				//cout<<"starts p"<<i<<"loop"<<j<<endl;
				reprojectionErrors(affine_tmp, p1, p2, error);
				// cout<<"done p"<<i<<"loop"<<j<<endl;
				// int inlierscount_if(error.begin(),error.end(),m_compare);
				// cout<<"inliers size"<<inliers<<endl;
				// if (inliers>most_inliers){
				// 	most_inliers=inliers;
				// 	best_affine=affine_tmp;
				// }
				if (error < min_error){
					min_error = error;
					best_affine = affine_tmp;
				}
			}
		}
	}
}


void reconstruct(//初始化
	vector<KP>& last_keypoints,
	vector<KP>& next_keypoints,
	Mat& last_descriptor,
	Mat& next_descriptor,
	vector<Vec3b>& last_colors,
	vector<Vec3b>& next_colors,
	vector<DMatch>& matched,
	vector<Vec3b>& c1,
	vector<Point3d>& p2,
	Mat& affine,
	double& min_error)
{   
	matchFeatures(last_keypoints, next_keypoints, last_descriptor, next_descriptor, matched);
	vector<Point3d> p1;
	vector<Vec3b> c2;
	get_matched_points(last_keypoints, next_keypoints, last_colors, next_colors, matched, p1, p2, c1, c2);
	estimateTransform2D(p1, p2, affine, min_error);

	// matchFeaturesAffine(affine, last_keypoints, next_keypoints, matched);
	// get_matched_points(last_keypoints, next_keypoints, last_colors, next_colors, matched, p1, p2, c1, c2);
	// estimateTransform2D(p1, p2, affine, min_error);

	// cout << "affine: " << affine_tmp << endl;
	// affine_tmp = estimateRigidTransform(p3, p4, false);
	if (affine.rows == 0){
    	cout << "Fail to estimate affine transformation, number of points: " << p1.size() << endl;
    }
 //    affine_tmp /= pow(pow(affine_tmp.at<double>(0,0),2)+pow(affine_tmp.at<double>(0,1),2),0.5);
 //    affine_tmp.convertTo(affine.rowRange(0,2),CV_64F);
}

void init_structure(//初始化
	vector<vector<KP>>& KPs_for_all,
	vector<vector<Vec3b>>& colors_for_all,
	vector<Mat>& descriptors_for_all,
	//vector<vector<DMatch>>& matched_for_all,
	vector<Point3d>& structure,
	vector<Vec3b>& colors,
	vector<vector<int>>& correspond_struct_idx,
	vector<Mat>& affines,
	int& next_img_id)
{   
	vector<DMatch> matched, matched_tmp;
	Mat affine, affine_tmp;
	vector<Vec3b> colors_tmp;
	vector<Point3d> p2, p2_tmp;
	double min_error1, min_error2;
	//reconstruct(keypoints_for_all[0], keypoints_for_all[1], colors_for_all[0], colors_for_all[1], matched_for_all[0], colors, p2);
    // matchFeaturesAffine(affine, keypoints_for_all[0], keypoints_for_all[1], matched_for_all[0]);
    
    reconstruct(KPs_for_all[0], KPs_for_all[1], descriptors_for_all[0], descriptors_for_all[1], colors_for_all[0], colors_for_all[1], matched, colors, p2, affine, min_error1);
    
	next_img_id = 1;

	// skip frames
    if (min_error1 > error_threshold){
    	reconstruct(KPs_for_all[0], KPs_for_all[2], descriptors_for_all[0], descriptors_for_all[2], colors_for_all[0], colors_for_all[2], matched_tmp, colors_tmp, p2_tmp, affine_tmp, min_error2);
    	if (min_error2 < min_error1){
    		matched = matched_tmp;
    		colors = colors_tmp;
    		p2 = p2_tmp;
    		affine = affine_tmp;
    		next_img_id = 2;
    	}
    }
    

    affine = affine*affines.back();
	affines.push_back(affine);

	for(int i = 0; i < p2.size(); i++){
		structure.push_back(Point3d(Mat(affine.inv()*Mat(p2[i]))));
	}

    correspond_struct_idx.clear();
	correspond_struct_idx.resize(KPs_for_all.size());
	for (int i = 0; i < KPs_for_all.size(); ++i)
	{
		correspond_struct_idx[i].resize(KPs_for_all[i].size(), -1);//与key points的size保持一致。为什么要加-1？
	}

	//��дͷ����ͼ���Ľṹ����
	int idx = 0;
	for (int i = 0; i < matched.size(); ++i)
	{
		// if (mask.at<uchar>(i) == 0)//如果这一对match是outlier
		// 	continue;//如果mask的值等于0的话，继续从头开始循环，否则往下继续

		correspond_struct_idx[0][matched[i].queryIdx] = idx;//将第idx个match,即idx，存入correspond_struct_idx中，存的位置为：如果是前一张，则是第一行，后一张则是第二行，纵坐标对应点在图重视第几个feature point
		correspond_struct_idx[next_img_id][matched[i].trainIdx] = idx;
		++idx;
	}	
}
// fusion_structure(
// 				matched,
// 				correspond_struct_idx[i],
// 				correspond_struct_idx[next_img_id],
// 				structure,
// 				next_structure,
// 				colors,
// 				c1
// 				);
void fusion_structure(
	vector<DMatch>& matched,
	vector<int>& struct_indices,
	vector<int>& next_struct_indices,
	vector<Point3d>& structure,
	vector<Point3d>& next_structure,
	vector<Vec3b>& colors,
	vector<Vec3b>& next_colors
	)
{
	for (int i = 0; i < matched.size(); ++i)
	{
		int query_idx = matched[i].queryIdx;
		int train_idx = matched[i].trainIdx;

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




//定义代价函数
struct ReprojectCost
{
    Point2d observation;

    ReprojectCost(Point2d& observation)
        : observation(observation)
    {
    }
   //使用模板的目的就是能够让程序员编写与类型无关的代码
  //AutoDiffCostFunction<ReprojectCost, 2, 4, 6, 3>
    template <typename T>
    bool operator()(const T* const extrinsic, const T* const structure, T* residuals) const//pos3d:对应的3D点；observation:相应的图片的坐标，
   //通过对3D点来计算2D坐标值与实际值比较，检查R，T的正确性
    {
        residuals[0] = cos(extrinsic[0])*structure[0]-sin(extrinsic[0])*structure[1]+extrinsic[1] - T(observation.x);//反向投影误差
        residuals[1] = sin(extrinsic[0])*structure[0]+cos(extrinsic[0])*structure[1]+extrinsic[2] - T(observation.y);

        return true;
    }
};
//使用
 //Solver求解BA，其中使用了Ceres提供的Huber函数作为损失函数
void bundle_adjustment(
    vector<Mat>& extrinsics,
    vector<vector<int> >& correspond_struct_idx,
    vector<vector<KP> >& keypoints_for_all,
    vector<Point3d>& structure
)
{
	vector<Point2d> structure_2d;
	convertPointsHomogeneous(structure, structure_2d);
    ceres::Problem problem;

    // load extrinsics (rotations and motions)
    for (size_t i = 0; i < extrinsics.size(); ++i)
    {
        problem.AddParameterBlock(extrinsics[i].ptr<double>(), 3);//Add a parameter block with appropriate size and parameterization to the problem.
       //Repeated calls with the same arguments are ignored. Repeated calls with the same double pointer but a different size results in undefined behavior.
    }
    // fix the first camera.
    problem.SetParameterBlockConstant(extrinsics[0].ptr<double>());//Hold the indicated parameter block constant during optimization.保持第一个外惨矩阵不变

    // load points
    ceres::LossFunction* loss_function = new ceres::HuberLoss(4);   // loss function make bundle adjustment robuster.
    for (size_t img_idx = 0; img_idx < correspond_struct_idx.size(); ++img_idx)
    {
        vector<int>& point3d_ids = correspond_struct_idx[img_idx];
        vector<KP>& keypoints = keypoints_for_all[img_idx];
        for (size_t point_idx = 0; point_idx < point3d_ids.size(); ++point_idx)
        {
            int point3d_id = point3d_ids[point_idx];
            if (point3d_id < 0)
                continue;

            Point2d observed;
            observed.x = keypoints[point_idx].pt.x;
            observed.y = keypoints[point_idx].pt.y;//corresponding 2D points coordinates with feasible 3D point
            
            // 模板参数中，第一个为代价函数的类型，第二个为代价的维度，剩下三个分别为代价函数第一第二还有第三个参数的维度
            ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<ReprojectCost, 2, 3, 2>(new ReprojectCost(observed));
            //向问题中添加误差项
            problem.AddResidualBlock(//adds a residual block to the problem,implicitly adds the parameter blocks(This causes additional correctness checking) if they are not present
                cost_function,
                loss_function,
                extrinsics[img_idx].ptr<double>(),  // View Rotation and Translation
                &(structure_2d[point3d_id].x)          // Point in 3D space
            );
        }
    }

    // Solve BA
    ceres::Solver::Options ceres_config_options;
    ceres_config_options.minimizer_progress_to_stdout = false;
    ceres_config_options.logging_type = ceres::SILENT;
    ceres_config_options.num_threads = 1;//Number of threads to be used for evaluating the Jacobian and estimation of covariance.
    ceres_config_options.preconditioner_type = ceres::JACOBI;
    ceres_config_options.linear_solver_type = ceres::DENSE_SCHUR;
    // ceres_config_options.linear_solver_type = ceres::SPARSE_SCHUR;//ype of linear solver used to compute the solution to the linear least squares problem in each iteration of the Levenberg-Marquardt algorithm
    // ceres_config_options.sparse_linear_algebra_library_type = ceres::EIGEN_SPARSE;

    ceres::Solver::Summary summary;
    ceres::Solve(ceres_config_options, &problem, &summary);

    if (!summary.IsSolutionUsable())
    {
        cout << "Bundle Adjustment failed." << endl;
    }
    else
    {
        // Display statistics about the minimization
        cout << endl
            << "Bundle Adjustment statistics (approximated RMSE):\n"
            << " #views: " << extrinsics.size() << "\n"
            << " #residuals: " << summary.num_residuals << "\n"
            << " Initial RMSE: " << sqrt(summary.initial_cost / summary.num_residuals) << "\n"
            << " Final RMSE: " << sqrt(summary.final_cost / summary.num_residuals) << "\n"
            << " Time (s): " << summary.total_time_in_seconds << "\n"
            << endl;
    }
    convertPointsHomogeneous(structure_2d, structure);
}




int main( int argc, char** argv )
{
	string data_path = argv[1];
	int start = stoi(argv[2]);
	int end = stoi(argv[3]);
	Mat K(Matx33d(
		350.6847, 0, 332.4661,
		0, 350.0606, 163.7461,
		0, 0, 1));
	Vec3b blue(255,0,0);
	Vec3b yellow(0,255,255);
	Vec3b orange(0,0,255);
	
	vector<vector<KP>> KPs_for_all;
	vector<vector<KeyPoint>> keypoints_for_all;
	vector<Mat> descriptors_for_all;
	vector<vector<Vec3b>> colors_for_all;
	vector<vector<DMatch>> matched_for_all;
	vector<Mat> affines;
	Mat affine = Mat::eye(3,3,CV_64F);
	affines.push_back(affine);

	int keyframe_id = 64;
	Ptr<Feature2D> sift = xfeatures2d::SIFT::create(0, 3, 0.04, 10);

	int imgId = 0;
	for(int i = start; i <= end; i++)
	{
		vector<KP> KPs;
		vector<KeyPoint> keypoints;
		vector<Vec3b> colors;
		ifstream csvPath ( data_path+"/"+to_string(i)+".csv" );
		string line, x, y, label, X, Y, Z; 
		int id;
		// Mat imgLast, imgNext, outImg;
	    Mat img = imread(data_path+"/"+to_string(i)+".png"); 
	    while (getline(csvPath, line)) 
	    {  
	        stringstream liness(line);  
	        getline(liness, x, ',');  
	        getline(liness, y, ','); 
	        getline(liness, label, ',');
	        getline(liness, X, ','); 
	        getline(liness, Y, ','); 
	        getline(liness, Z, ','); 
	        
	        // circle(img, Point (stoi(x),stoi(y)), 3, Scalar (0,0,0), -1);
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
	            KP kp = {pt, id};
	        	KPs.push_back(kp);
	        	KeyPoint keypoint = {stof(x),stof(y),3,-1,0,0,id};
	        	keypoints.push_back(keypoint);
	        }
	        
    	}
    	if(keypoints.size()<2){
    		cout << "Too few keypoint!" << endl;
    		return 0;
    	}
    	Mat descriptors;
    	sift->compute(img, keypoints, descriptors);

		keypoints_for_all.push_back(keypoints);
		KPs_for_all.push_back(KPs);
		descriptors_for_all.push_back(descriptors);
		colors_for_all.push_back(colors);

		if (imgId > 0){
			vector<DMatch> matched;
			// matchFeatures(keypoints_for_all[imgId-1], keypoints_for_all[imgId], matched);
			matchFeatures(KPs_for_all[imgId-1], KPs_for_all[imgId], descriptors_for_all[imgId-1], descriptors_for_all[imgId], matched);
			matched_for_all.push_back(matched);

			// Mat imgLast = imread(data_path+"/"+to_string(imgId-1)+".png");
			// Mat imgNext = imread(data_path+"/"+to_string(imgId)+".png");
			// Mat outImg;
			// resize(imgLast, imgLast, Size(320, 180));
			// resize(imgNext, imgNext, Size(320, 180));
			// drawMatches(imgLast, keypoints_for_all[imgId-1], imgNext, keypoints_for_all[imgId], matched, outImg);
			// namedWindow("MatchSIFT", WINDOW_NORMAL);
			// imshow("MatchSIFT",outImg);
			// waitKey(0);
		}
		imgId++;
	}

	// vector<DMatch> matched;
	// matchFeatures(KPs_for_all[64], KPs_for_all[236], descriptors_for_all[64], descriptors_for_all[236], matched);
	// matchFeatures2(KPs_for_all[64], KPs_for_all[236], matched);

	vector<Point3d> structure;
	vector<Vec3b> colors;
	vector<vector<int>> correspond_struct_idx;
	int first_next_img_id;

	init_structure(//此时已做完第一张图和第二张图的重建，rotations和motions里放了两张图的R和T
		KPs_for_all,
		colors_for_all,
		descriptors_for_all,
		//matched_for_all,
		structure,
		colors,
		correspond_struct_idx,
		affines,
		first_next_img_id
		);

	if(keypoints_for_all.size()-2 > first_next_img_id)
		for (int i = first_next_img_id; i < keypoints_for_all.size()-2; ++i)//遍历，从第二张图和第三张图开始，每次两张图的match
		{
			Mat affine, affine_tmp;
			vector<Point3d> p2, p2_tmp;
			vector<Vec3b> c1, c1_tmp;
			vector<DMatch> matched, matched_tmp;
			int next_img_id;
			double min_error1, min_error2;
			reconstruct(KPs_for_all[i], KPs_for_all[i+1], descriptors_for_all[i], descriptors_for_all[i+1], colors_for_all[i], colors_for_all[i+1], matched, c1, p2, affine, min_error1);
	    	
	    	next_img_id = i+1;

	    	//skip frames
			if (min_error1 > error_threshold){
		    	reconstruct(KPs_for_all[i], KPs_for_all[i+2], descriptors_for_all[i], descriptors_for_all[i+2], colors_for_all[i], colors_for_all[i+2], matched_tmp, c1_tmp, p2_tmp, affine_tmp, min_error2);
		    	if (min_error2 < min_error1){
		    		matched = matched_tmp;
		    		c1 = c1_tmp;
		    		p2 = p2_tmp;
		    		affine = affine_tmp;
		    		next_img_id = i+2;
		    	}
		    }
	    	cout << "min_error1: " << min_error1 << ", min_error2: " << min_error2 << endl;

		    // matchFeaturesAffine(affine, keypoints_for_all[i], keypoints_for_all[i+1], matched_for_all[i]);
		    // reconstruct(keypoints_for_all[i], keypoints_for_all[i+1], colors_for_all[i], colors_for_all[i+1], matched_for_all[i], c1, p2, affine);

		    affine = affine*affines.back();
			affines.push_back(affine);

			vector<Point3d> next_structure;
			for(int i = 0; i < p2.size(); i++){
				next_structure.push_back(Point3d(Mat(affine.inv()*Mat(p2[i]))));
			}

			fusion_structure(
				matched,
				correspond_struct_idx[i],
				correspond_struct_idx[next_img_id],
				structure,
				next_structure,
				colors,
				c1
				);
			cout << "from "<< i << " to " << next_img_id << endl;
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


	// // bundle_adjustment
	// google::InitGoogleLogging(argv[0]);
	// vector<Mat> extrinsics;
	// for (size_t i = 0; i < affines.size(); ++i)
	// {
	//   Mat extrinsic(Matx31d(asin(affines[i].at<double>(1,0)), affines[i].at<double>(0,2), affines[i].at<double>(1,2)));
	//   extrinsics.push_back(extrinsic);
	// }
	// bundle_adjustment(extrinsics, correspond_struct_idx, KPs_for_all, structure);

	// for (size_t i = 0; i < affines.size(); ++i)
	// {
	// 	double alpha = extrinsics[i].at<double>(0,0);
	// 	double tx = extrinsics[i].at<double>(1,0);
	// 	double ty = extrinsics[i].at<double>(2,0);
	// 	affines[i] = (Mat_<double>(3, 3) << cos(alpha), -sin(alpha), tx, sin(alpha), cos(alpha), ty, 0, 0, 1);
	// }



	int resultSize = 1000;
	double resultResize = 50;
	Mat result = Mat::zeros(resultSize, resultSize, CV_8UC3);
	vector<Point2d> path;
	
	int count = 0;
	for(int i = 0; i < structure.size(); i++){
		// cout << count_same_structure[i] << endl;
		// cout << structure[i] << colors[i] << endl;
		if(count_same_structure[i] > rm_cone_threshold){
			count++;
			int x = int(structure[i].x * resultResize + resultSize/4);
			int y = int(structure[i].y * resultResize + resultSize/4);
			if (x >= 0 && x <= resultSize && y >= 0 && y <= resultSize){
				circle(result, Point (x,y), 3, colors[i], -1);
				// putText(result, to_string(i), Point(x,y), 1, 0.5, Scalar(255, 255, 255));
			}
		}
	}
	cout << "Number of structure: " << count << endl;

	for(int i = 0; i < affines.size(); i++){
		Mat camera_cor(Matx31d(0,0.5,1));
		camera_cor = affines[i].inv() * camera_cor;
		// if(i>0)
		// 	cout << "heading change: " << acos(affines[i].at<double>(0,0))-acos(affines[i-1].at<double>(0,0)) << endl;
		int x = int(camera_cor.at<double>(0,0) * resultResize + resultSize/4);
		int y = int(camera_cor.at<double>(1,0) * resultResize + resultSize/4);
		if (x >= 0 && x <= resultSize && y >= 0 && y <= resultSize){
			circle(result, Point (x,y), 3, Scalar (255,255,255), -1);
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