#include <iostream>
#include <string>
#include <vector>
#include <utility>
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
// #include <ceres/ceres.h>
// #include <Eigen/Core>
// #include <iostream>
// #include <utility>
// #include <string>
// #include <vector>
// #include <ceres/rotation.h>
// #include <ceres/problem.h>
// #include <fstream>
// #include <algorithm>
// #include <cmath>
// #include <cstdio>
// #include <cstdlib>


using std::cin;
//using namespace cv;
using namespace std;


float threshold = 70;

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
     // cout << index << " " << i << " " << imageId << " " << minRes << endl;
    } 
  }

 //cout<<matched[0].trainIdx<<endl;
}
void matchFeaturesForAll(vector<vector<cv::KeyPoint> >&key_points_for_all, vector<vector<cv::DMatch> >& matches_for_all)
{
  matches_for_all.clear();
  for (int i = 0; i < key_points_for_all.size() - 1; ++i)//遍历每个图的descriptor
  {
    cout << "Matching images " << i << " - " << i + 1 << endl;//每次两张图
    vector<cv::DMatch> matches;
    int imageId = i+1;
    matchFeatures(imageId, key_points_for_all[i], key_points_for_all[i+1], matches);//将每次匹配的结果作为一个元素放入matchFeaturesForAll中
    matches_for_all.push_back(matches);
  }

}

bool find_transform(cv::Mat& K, vector<cv::Point2f>& p1, vector<cv::Point2f>& p2, cv::Mat& R, cv::Mat& T, cv::Mat& mask)
{

  double focal_length = 0.5*(K.at<double>(0) + K.at<double>(4));
  cv::Point2d principle_point(K.at<double>(2), K.at<double>(5));
  cv::Mat E = cv::findEssentialMat(p1, p2, focal_length, principle_point, cv::RANSAC, 0.99, 1, mask);
  // cout<<p1<<endl;
  // cout<<p2<<endl;
  // cout<<E<<endl;
  if (E.empty()) return false;
  
  double feasible_count = countNonZero(mask);
  cout << (int)feasible_count << " -in- " << p1.size() << endl;

  if (feasible_count <= 4 || (feasible_count / p1.size()) < 0.6)
    return false;

  int pass_count = recoverPose(E, p1, p2, R, T, focal_length, principle_point, mask);
  // cout<<R<<endl;
  // cout<<T<<endl;
  if (((double)pass_count) / feasible_count < 0.7)
    return false;

  return true;
}
//get_matched_points(key_points_for_all[0], key_points_for_all[1], matches_for_all[0], p1, p2);
void get_matched_points(//根据matches,返回匹配时两张图分别的坐标
  vector<cv::KeyPoint>& p1,
  vector<cv::KeyPoint>& p2,
  vector<cv::DMatch> matches,
  vector<cv::Point2f>& out_p1,
  vector<cv::Point2f>& out_p2
  )
{
  out_p1.clear();
  out_p2.clear();
  for (int i = 0; i < matches.size(); ++i)
  {
    out_p1.push_back(p1[matches[i].queryIdx].pt);
    out_p2.push_back(p2[matches[i].trainIdx].pt);
  }
}

void get_matched_colors(
  vector<cv::Vec3b>& c1,
  vector<cv::Vec3b>& c2,
  vector<cv::DMatch> matches,
  vector<cv::Vec3b>& out_c1,
  vector<cv::Vec3b>& out_c2
  )
{
  out_c1.clear();
  out_c2.clear();
  for (int i = 0; i < matches.size(); ++i)
  {
    out_c1.push_back(c1[matches[i].queryIdx]);
    out_c2.push_back(c2[matches[i].trainIdx]);
  }
}

void maskout_points(vector<cv::Point2f>& p1, cv::Mat& mask)
{
  vector<cv::Point2f> p1_copy = p1;
  p1.clear();

  for (int i = 0; i < mask.rows; ++i)
  {
    if (mask.at<uchar>(i) > 0)
      p1.push_back(p1_copy[i]);
  }
}

void maskout_colors(vector<cv::Vec3b>& p1, cv::Mat& mask)
{
  vector<cv::Vec3b> p1_copy = p1;
  p1.clear();

  for (int i = 0; i < mask.rows; ++i)
  {
    if (mask.at<uchar>(i) > 0)
      p1.push_back(p1_copy[i]);
  }
}
void get_objpoints_and_imgpoints(
  vector<cv::DMatch>& matches,//从第二张图开始，每两张图的matches，其中包含很多个matchFeaturesForAll
  vector<int>& correspond_struct_idx,
  vector<cv::Point3d>& structure,
  vector<cv::KeyPoint>& key_points,//后一张图的key points
  vector<cv::Point3f>& object_points,
  vector<cv::Point2f>& image_points)
{
  object_points.clear();
  image_points.clear();

  for (int i = 0; i < matches.size(); ++i)
  {
    int query_idx = matches[i].queryIdx;//在相应的图中是第几个feature point
    int train_idx = matches[i].trainIdx;
    //cout<<correspond_struct_idx[1]<<endl;
    int struct_idx = correspond_struct_idx[query_idx];//访问对应列，输出为第几个有用的match，一个有用的match能组成一个U
    //cout<<structure[1]<<endl;
    cout << query_idx << " " << train_idx << " " << struct_idx << endl;

    if (struct_idx < 0) continue;

    object_points.push_back(structure[struct_idx]);//输出该次匹配有用的match的3D坐标
    image_points.push_back(key_points[train_idx].pt);//输出有用的match在最新的图上match的坐标
  }
}
void reconstruct(cv::Mat& K, cv::Mat& R1, cv::Mat& T1, cv::Mat& R2, cv::Mat& T2, vector<cv::Point2f>& p1, vector<cv::Point2f>& p2, vector<cv::Point3d>& structure)//与双目的不同，有修改
{
  //����������ͶӰ����[R T]��triangulatePointsֻ֧��float��
  cv::Mat proj1(3, 4, CV_32FC1);
  cv::Mat proj2(3, 4, CV_32FC1);

  R1.convertTo(proj1(cv::Range(0, 3), cv::Range(0, 3)), CV_32FC1);//将R1的值赋值给proj1的前三列前三行
  T1.convertTo(proj1.col(3), CV_32FC1);//将T的值赋值到第四列
  // cout<<R2;

  R2.convertTo(proj2(cv::Range(0, 3),cv::Range(0, 3)), CV_32FC1);//将R2的值赋值给proj2的前三列前三行
  T2.convertTo(proj2.col(3), CV_32FC1);

  cv::Mat fK;
  K.convertTo(fK, CV_32FC1);
  proj1 = fK*proj1;
  proj2 = fK*proj2;

  //�����ؽ�
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
void init_structure(//初始化
  cv::Mat K,
  vector<vector<cv::KeyPoint> >& key_points_for_all,
  vector<vector<cv::Vec3b> >& colors_for_all,
  vector<vector<cv::DMatch> >& matches_for_all,
  vector<cv::Point3d>& structure,
  vector<vector<int> >& correspond_struct_idx,
  vector<cv::Vec3b>& colors,
  vector<cv::Mat>& rotations,
  vector<cv::Mat>& motions
  )
{
  vector<cv::Point2f> p1, p2;
  vector<cv::Vec3b> c2;
  cv::Mat R, T; 
  cv::Mat mask; 
  get_matched_points(key_points_for_all[0], key_points_for_all[1], matches_for_all[0], p1, p2);
  get_matched_colors(colors_for_all[0], colors_for_all[1], matches_for_all[0], colors, c2);
  find_transform(K, p1, p2, R, T, mask);//求得第二张图的P

  maskout_points(p1, mask);
  maskout_points(p2, mask);
  maskout_colors(colors, mask);

  cv::Mat R0 = cv::Mat::eye(3, 3, CV_64FC1);//第一张图的P
  cv::Mat T0 = cv::Mat::zeros(3, 1, CV_64FC1);
  reconstruct(K, R0, T0, R, T, p1, p2, structure);
  //cout<<structure[1]<<endl;
  rotations.push_back(R0);
  rotations.push_back(R);
  motions.push_back(T0);
  motions.push_back(T);

  //将correspond_struct_idx变为key_points_for_all的形式
  correspond_struct_idx.clear();
  correspond_struct_idx.resize(key_points_for_all.size());
  for (int i = 0; i < key_points_for_all.size(); ++i)
  {
    correspond_struct_idx[i].resize(key_points_for_all[i].size(), -1);//与key points的size保持一致。为什么要加-1？
  }

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

int main( int argc, char** argv )
{
  cv::Mat K(cv::Matx33d(
    350.6847, 0, 332.4661,
    0, 350.0606, 163.7461,
    0, 0, 1));
  cv::Mat yellow = (cv::Mat_<double>(1, 3) << 0,255,255);
  cv::Mat blue = (cv::Mat_<double>(1, 3) << 255,0,0);
  cv::Mat orange = (cv::Mat_<double>(1, 3) << 0,165,255);

  vector<vector<cv::KeyPoint> > key_points_for_all;
  vector<vector<cv::Vec3b> > colors_for_all;
  vector<vector<cv::DMatch> > matches_for_all;

  for(int i = 0; i < 4; i++)
  {
    vector<cv::KeyPoint> feature;
    vector<cv::Vec3b> colors;
    ifstream file ( "result/"+to_string(i)+".csv" );
    string line, x, y, label;  
    while (getline(file, line)) 
    {  
        stringstream liness(line);  
        getline(liness, x, ',');  
        getline(liness, y, ','); 
        getline(liness, label);
        feature.push_back(cv::KeyPoint(stof(x),stof(y),3,-1,0,0,stof(label)));
        if(stof(label) == 1)
          colors.push_back(yellow);
        if(stof(label) == 2)
          colors.push_back(blue);
        if(stof(label) == 3)
          colors.push_back(orange);
    }
    key_points_for_all.push_back(feature);
    colors_for_all.push_back(colors);
    if (i > 0){
      vector<cv::DMatch> matched;
      matchFeatures(i, key_points_for_all[i-1], key_points_for_all[i], matched);
      matches_for_all.push_back(matched);
    }
  }      
  
  // matchFeaturesForAll(key_points_for_all, matches_for_all);
  //cout<<matches_for_all[0][0].queryIdx;
  // vector<cv::Point2f> p1;
  // vector<cv::Point2f> p2;
  vector<cv::Point3d> structure;
  vector<vector<int> > correspond_struct_idx;
  vector<cv::Vec3b> colors;
  vector<cv::Mat> rotations;
  vector<cv::Mat> motions;
  //get_matched_points(key_points_for_all[0], key_points_for_all[1], matches_for_all[0], p1, p2);
  // find_transform(cv::Mat& K, vector<cv::Point2f>& p1, vector<cv::Point2f>& p2, cv::Mat& R, cv::Mat& T, cv::Mat& mask)
//初始化structure
  //  cv::Mat K,
//   vector<vector<cv::KeyPoint> >& key_points_for_all,
//   vector<vector<cv::Vec3b> >& colors_for_all,
  init_structure(//此时已做完第一张图和第二张图的重建，rotations和motions里放了两张图的R和T
    K,
    key_points_for_all,
    colors_for_all,
    matches_for_all,
    structure,
    correspond_struct_idx,
    colors,
    rotations,
    motions
  );

  for (int i = 1; i < matches_for_all.size(); ++i)//遍历，从第二张图和第三张图开始，每次两张图的match
  {
    vector<cv::Point3f> object_points;//3D points
    vector<cv::Point2f> image_points;
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
    // cout << matches_for_all[i].size() << endl;
    // cout << image_points.size() << endl;
    // bool solvePnPRansac(InputArray objectPoints, InputArray imagePoints, InputArray cameraMatrix, InputArray distCoeffs, OutputArray rvec, OutputArray tvec,
    // bool useExtrinsicGuess=false, int iterationsCount=100, float reprojectionError=8.0, double confidence=0.99, OutputArray inliers=noArray(), int flags=SOLVEPNP_ITERATIVE )
    // rvec – Output rotation vector (see Rodrigues() ) that, together with tvec , brings points from the model coordinate system to the camera coordinate system.
    // tvec – Output translation vector.
    solvePnPRansac(object_points, image_points, K, cv::noArray(), r, T);
    Rodrigues(r, R);//Converts a rotation matrix to a rotation vector or vice versa.
    // //得到最新一张图的R和T
    rotations.push_back(R);
    motions.push_back(T);

    vector<Point2f> p1, p2;
    vector<Vec3b> c1, c2;
    get_matched_points(key_points_for_all[i], key_points_for_all[i + 1], matches_for_all[i], p1, p2);//返回p1,p2，即两张图在匹配时的对应的坐标
    get_matched_colors(colors_for_all[i], colors_for_all[i + 1], matches_for_all[i], c1, c2);

    //求3D点
    vector<Point3d> next_structure;
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
}
