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

void matchFeatures(int imageId, vector<KP> featureLast, 
                  vector<KP> featureNext, vector<DMatch> &matched){
	float mThreshold = 0.1;
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
			matched.push_back(DMatch(index,i,imageId,minRes));
			cout << index << " " << i << " " << imageId << " " << minRes << endl;
		} 
	}
}

void matchFeaturesAffine(int imageId, Mat affine, vector<KP> featureLast, 
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
		    matched.push_back(DMatch(index,i,imageId,minRes));
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

void reconstruct(//初始化
	vector<KP>& last_keypoints,
	vector<KP>& next_keypoints,
	vector<Vec3b>& last_colors,
	vector<Vec3b>& next_colors,
	vector<DMatch>& matches,
	vector<Vec3b>& c1,
	vector<Point3d>& p2,
	Mat& affine
	)
{
	vector<Point3d> p1;
	vector<Point2d> p3, p4;
	vector<double> res;
	vector<Vec3b> c2;
	Mat affine_tmp;
	get_matched_points(last_keypoints, next_keypoints, last_colors, next_colors, matches, p1, p2, c1, c2);

	// Mat mask;

	for(int i = 0; i < p1.size(); i++){
		p3.push_back(Point2d(p1[i].x,p1[i].y));
		p4.push_back(Point2d(p2[i].x,p2[i].y));
	}
	
	affine_tmp = estimateRigidTransform(p3, p4, false);
	if (affine_tmp.rows == 0){
    	cout << "Fail to estimate affine transformation" << endl;
    }
    affine_tmp /= pow(pow(affine_tmp.at<double>(0,0),2)+pow(affine_tmp.at<double>(0,1),2),0.5);
    affine_tmp.convertTo(affine.rowRange(0,2),CV_64F);
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
	Mat affine(Matx33d(0,0,0,0,0,0,0,0,1));
	vector<Point3d> p2;
	reconstruct(keypoints_for_all[0], keypoints_for_all[1], colors_for_all[0], colors_for_all[1], matches_for_all[0], colors, p2, affine);

    matchFeaturesAffine(1, affine, keypoints_for_all[0], keypoints_for_all[1], matches_for_all[0]);
    reconstruct(keypoints_for_all[0], keypoints_for_all[1], colors_for_all[0], colors_for_all[1], matches_for_all[0], colors, p2, affine);

    // for (int i = 0; i < matches_for_all.size(); ++i)
    // 	for (int j = 0; j < matches_for_all[i].size(); ++j)
    // 		cout << matches_for_all[i][j].distance << endl;

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

void get_objpoints_and_imgpoints(
	vector<DMatch>& matches,//从第二张图开始，每两张图的matches，其中包含很多个match_features
	vector<int>& struct_indices,
	vector<Point3d>& structure,
	vector<KP>& keypoints,//后一张图的key points
	vector<Point3d>& object_points,
	vector<Point3d>& image_points)
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
		image_points.push_back(keypoints[train_idx].pt);//输出有用的match在最新的图上match的坐标
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

// struct ReprojectCost
// {
//     cv::Point3d observation;

//     ReprojectCost(cv::Point3d& observation)
//         : observation(observation)
//     {
//     }
//    //使用模板的目的就是能够让程序员编写与类型无关的代码
//   //AutoDiffCostFunction<ReprojectCost, 2, 4, 6, 3>
//     template <typename T>
//     bool operator()(const T* const intrinsic, const T* const extrinsic, const T* const pos3d, T* residuals) const//pos3d:对应的3D点；observation:相应的图片的坐标，
//    //通过对3D点来计算2D坐标值与实际值比较，检查R，T的正确性
//     {
//         const T* r = extrinsic;//外参 R
//         const T* t = &extrinsic[3];//T

//         T pos_proj[3];//定义一个长度为4的pos_proj

//        AngleAxisRotatePoint(r, pos3d, pos_proj);//y=r(angle_axis)pos3d,根据R进行旋转变化

//         // Apply the camera translation
//         pos_proj[0] += t[0];//平移变化
//         pos_proj[1] += t[1];
//         pos_proj[2] += t[2];

//         const T x = pos_proj[0] / pos_proj[2];//求x和y
//         const T y = pos_proj[1] / pos_proj[2];

//         const T fx = intrinsic[0];//读取内参矩阵
//         const T fy = intrinsic[1];
//         const T cx = intrinsic[2];
//         const T cy = intrinsic[3];

//         // Apply intrinsic
//         const T u = fx * x + cx;//这是啥？
//         const T v = fy * y + cy;

//         residuals[0] = u - T(observation.x);//反向投影误差
//         residuals[1] = v - T(observation.y);

//         return true;
//     }
// };

// void bundle_adjustment(
//     cv::Mat& intrinsic,
//     vector<cv::Mat>& extrinsics,
//     vector<vector<int> >& correspond_struct_idx,
//     vector<vector<keypoint> >& keypoints_for_all,
//     vector<cv::Point3d>& structure
// )
// {
//     Problem problem;

//     // load extrinsics (rotations and motions)
//     for (size_t i = 0; i < extrinsics.size(); ++i)
//     {
//         problem.AddParameterBlock(extrinsics[i].ptr<double>(), 6);//Add a parameter block with appropriate size and parameterization to the problem.
//        //Repeated calls with the same arguments are ignored. Repeated calls with the same double pointer but a different size results in undefined behavior.
//     }
//     // fix the first camera.
//     problem.SetParameterBlockConstant(extrinsics[0].ptr<double>());//Hold the indicated parameter block constant during optimization.保持第一个外惨矩阵不变

//     // load intrinsic
//     problem.AddParameterBlock(intrinsic.ptr<double>(), 4); // fx, fy, cx, cy

//     // load points
//     LossFunction* loss_function = new HuberLoss(4);   // loss function make bundle adjustment robuster.
//     for (size_t img_idx = 0; img_idx < correspond_struct_idx.size(); ++img_idx)
//     {
//         vector<int>& point3d_ids = correspond_struct_idx[img_idx];
//         vector<keypoint>& keypoints = keypoints_for_all[img_idx];
//         for (size_t point_idx = 0; point_idx < point3d_ids.size(); ++point_idx)
//         {
//             int point3d_id = point3d_ids[point_idx];
//             if (point3d_id < 0)
//                 continue;

//             cv::Point3d observed = keypoints[point_idx].pt;//corresponding 2D points coordinates with feasible 3D point
//             // 模板参数中，第一个为代价函数的类型，第二个为代价的维度，剩下三个分别为代价函数第一第二还有第三个参数的维度
//             CostFunction* cost_function = new AutoDiffCostFunction<ReprojectCost, 2, 4, 6, 3>(new ReprojectCost(observed));
//             //向问题中添加误差项
//             problem.AddResidualBlock(//adds a residual block to the problem,implicitly adds the parameter blocks(This causes additional correctness checking) if they are not present
//                 cost_function,
//                 loss_function,
//                 intrinsic.ptr<double>(),            // Intrinsic
//                 extrinsics[img_idx].ptr<double>(),  // View Rotation and Translation
//                 &(structure[point3d_id].x)          // Point in 3D space
//             );
//         }
//     }

//     // Solve BA
//     Solver::Options ceres_config_options;
//     ceres_config_options.minimizer_progress_to_stdout = false;
//     ceres_config_options.logging_type = SILENT;
//     ceres_config_options.num_threads = 1;//Number of threads to be used for evaluating the Jacobian and estimation of covariance.
//     ceres_config_options.preconditioner_type = JACOBI;
//     ceres_config_options.linear_solver_type = DENSE_SCHUR;
//     // ceres_config_options.linear_solver_type = ceres::SPARSE_SCHUR;//ype of linear solver used to compute the solution to the linear least squares problem in each iteration of the Levenberg-Marquardt algorithm
//     // ceres_config_options.sparse_linear_algebra_library_type = ceres::EIGEN_SPARSE;

//     Solver::Summary summary;
//     Solve(ceres_config_options, &problem, &summary);

//     if (!summary.IsSolutionUsable())
//     {
//         std::cout << "Bundle Adjustment failed." << std::endl;
//     }
//     else
//     {
//         // Display statistics about the minimization
//         std::cout << std::endl
//         << "Bundle Adjustment statistics (approximated RMSE):\n"
//         << " #views: " << extrinsics.size() << "\n"
//         << " #residuals: " << summary.num_residuals << "\n"
//         << " Initial RMSE: " << std::sqrt(summary.initial_cost / summary.num_residuals) << "\n"
//         << " Final RMSE: " << std::sqrt(summary.final_cost / summary.num_residuals) << "\n"
//         << " Time (s): " << summary.total_time_in_seconds << "\n"
//         << std::endl;
//     }
// }





// struct ReprojectCost
// {
//     cv::Point3d observation;

//     ReprojectCost(cv::Point3d& observation)
//         : observation(observation)
//     {
//     }
//    //使用模板的目的就是能够让程序员编写与类型无关的代码
//   //AutoDiffCostFunction<ReprojectCost, 2, 4, 6, 3>
//     template <typename T>
//     bool operator()(const T* const intrinsic, const T* const extrinsic, const T* const pos3d, T* residuals) const//pos3d:对应的3D点；observation:相应的图片的坐标，
//    //通过对3D点来计算2D坐标值与实际值比较，检查R，T的正确性
//     {
//         const T* r = extrinsic;//外参 R
//         const T* t = &extrinsic[3];//T

//         T pos_proj[3];//定义一个长度为4的pos_proj

//        AngleAxisRotatePoint(r, pos3d, pos_proj);//y=r(angle_axis)pos3d,根据R进行旋转变化

//         // Apply the camera translation
//         pos_proj[0] += t[0];//平移变化
//         pos_proj[1] += t[1];
//         pos_proj[2] += t[2];

//         const T x = pos_proj[0] / pos_proj[2];//求x和y
//         const T y = pos_proj[1] / pos_proj[2];

//         const T fx = intrinsic[0];//读取内参矩阵
//         const T fy = intrinsic[1];
//         const T cx = intrinsic[2];
//         const T cy = intrinsic[3];

//         // Apply intrinsic
//         const T u = fx * x + cx;//这是啥？
//         const T v = fy * y + cy;

//         residuals[0] = u - T(observation.x);//反向投影误差
//         residuals[1] = v - T(observation.y);

//         return true;
//     }
// };

// void bundle_adjustment(
//     cv::Mat& intrinsic,
//     vector<cv::Mat>& extrinsics,
//     vector<vector<int> >& correspond_struct_idx,
//     vector<vector<keypoint> >& keypoints_for_all,
//     vector<cv::Point3d>& structure
// )
// {
//     Problem problem;

//     // load extrinsics (rotations and motions)
//     for (size_t i = 0; i < extrinsics.size(); ++i)
//     {
//         problem.AddParameterBlock(extrinsics[i].ptr<double>(), 6);//Add a parameter block with appropriate size and parameterization to the problem.
//        //Repeated calls with the same arguments are ignored. Repeated calls with the same double pointer but a different size results in undefined behavior.
//     }
//     // fix the first camera.
//     problem.SetParameterBlockConstant(extrinsics[0].ptr<double>());//Hold the indicated parameter block constant during optimization.保持第一个外惨矩阵不变

//     // load intrinsic
//     problem.AddParameterBlock(intrinsic.ptr<double>(), 4); // fx, fy, cx, cy

//     // load points
//     LossFunction* loss_function = new HuberLoss(4);   // loss function make bundle adjustment robuster.
//     for (size_t img_idx = 0; img_idx < correspond_struct_idx.size(); ++img_idx)
//     {
//         vector<int>& point3d_ids = correspond_struct_idx[img_idx];
//         vector<keypoint>& keypoints = keypoints_for_all[img_idx];
//         for (size_t point_idx = 0; point_idx < point3d_ids.size(); ++point_idx)
//         {
//             int point3d_id = point3d_ids[point_idx];
//             if (point3d_id < 0)
//                 continue;

//             cv::Point3d observed = keypoints[point_idx].pt;//corresponding 2D points coordinates with feasible 3D point
//             // 模板参数中，第一个为代价函数的类型，第二个为代价的维度，剩下三个分别为代价函数第一第二还有第三个参数的维度
//             CostFunction* cost_function = new AutoDiffCostFunction<ReprojectCost, 2, 4, 6, 3>(new ReprojectCost(observed));
//             //向问题中添加误差项
//             problem.AddResidualBlock(//adds a residual block to the problem,implicitly adds the parameter blocks(This causes additional correctness checking) if they are not present
//                 cost_function,
//                 loss_function,
//                 intrinsic.ptr<double>(),            // Intrinsic
//                 extrinsics[img_idx].ptr<double>(),  // View Rotation and Translation
//                 &(structure[point3d_id].x)          // Point in 3D space
//             );
//         }
//     }

//     // Solve BA
//     Solver::Options ceres_config_options;
//     ceres_config_options.minimizer_progress_to_stdout = false;
//     ceres_config_options.logging_type = SILENT;
//     ceres_config_options.num_threads = 1;//Number of threads to be used for evaluating the Jacobian and estimation of covariance.
//     ceres_config_options.preconditioner_type = JACOBI;
//     ceres_config_options.linear_solver_type = DENSE_SCHUR;
//     // ceres_config_options.linear_solver_type = ceres::SPARSE_SCHUR;//ype of linear solver used to compute the solution to the linear least squares problem in each iteration of the Levenberg-Marquardt algorithm
//     // ceres_config_options.sparse_linear_algebra_library_type = ceres::EIGEN_SPARSE;

//     Solver::Summary summary;
//     Solve(ceres_config_options, &problem, &summary);

//     if (!summary.IsSolutionUsable())
//     {
//         std::cout << "Bundle Adjustment failed." << std::endl;
//     }
//     else
//     {
//         // Display statistics about the minimization
//         std::cout << std::endl
//         << "Bundle Adjustment statistics (approximated RMSE):\n"
//         << " #views: " << extrinsics.size() << "\n"
//         << " #residuals: " << summary.num_residuals << "\n"
//         << " Initial RMSE: " << std::sqrt(summary.initial_cost / summary.num_residuals) << "\n"
//         << " Final RMSE: " << std::sqrt(summary.final_cost / summary.num_residuals) << "\n"
//         << " Time (s): " << summary.total_time_in_seconds << "\n"
//         << std::endl;
//     }
// }

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
		keypoints_for_all.push_back(keypoints);
		colors_for_all.push_back(colors);
		if (imgId > 0){
			vector<DMatch> matched;
			matchFeatures(imgId, keypoints_for_all[imgId-1], keypoints_for_all[imgId], matched);
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
		vector<Point3d> object_points; 
		vector<Point3d> image_points;
		
		get_objpoints_and_imgpoints(
			matches_for_all[i],
			correspond_struct_idx[i],
			structure,
			keypoints_for_all[i+1],
			object_points,
			image_points
			);

		Mat affine(Matx33d(0,0,0,0,0,0,0,0,1));
		vector<Point3d> p2;
		vector<Vec3b> c1;
		reconstruct(keypoints_for_all[i], keypoints_for_all[i+1], colors_for_all[i], colors_for_all[i+1], matches_for_all[i], c1, p2, affine);

	    matchFeaturesAffine(i+1, affine, keypoints_for_all[i], keypoints_for_all[i+1], matches_for_all[i]);
	    reconstruct(keypoints_for_all[i], keypoints_for_all[i+1], colors_for_all[i], colors_for_all[i+1], matches_for_all[i], c1, p2, affine);

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

	// google::InitGoogleLogging(argv[0]);
	// vector<cv::Mat> extrinsics;
	// for (size_t i = 0; i < affines.size(); ++i)
	// {
	// 	cv::Mat extrinsic(6, 1, CV_64F);

	// 	r.copyTo(extrinsic.rowRange(0, 3));
	// 	motions[i].copyTo(extrinsic.rowRange(3, 6));

	// 	extrinsics.push_back(extrinsic);
	// }
	// bundle_adjustment(intrinsic, extrinsics, correspond_struct_idx, keypoints_for_all, structure);

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
			int x = int(structure[i].x * resultResize + resultSize/2);
			int y = int(structure[i].y * resultResize + resultSize/2);
			if (x >= 0 && x <= resultSize && y >= 0 && y <= resultSize){
				circle(result, Point (x,y), 3, colors[i], CV_FILLED);
			}
		}
	}
	cout << "Number of structure: " << count << endl;

	for(int i = 0; i < affines.size(); i++){
		Mat camera_cor(Matx31d(0,0,1));
		camera_cor = affines[i].inv() * camera_cor;
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