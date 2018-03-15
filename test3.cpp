// * Struct for matching: query descriptor index, train descriptor index, train image index and distance between descriptors.
  //*
  //* DMatch主要用来储存匹配信息的结构体，query是要匹配的描述子，train是被匹配的描述子，在Opencv中进行匹配时
  //* void DescriptorMatcher::match( const Mat& queryDescriptors, const Mat& trainDescriptors, vector<DMatch>& matches, const Mat& mask ) const
  ////* match函数的参数中位置在前面的为query descriptor，后面的是 train descriptor
  //* 例如：query descriptor的数目为20，train descriptor数目为30，则DescriptorMatcher::match后的vector<DMatch>的size为20
  //* 若反过来，则vector<DMatch>的size为30




/**********reference**************************/
//http://blog.csdn.net/weixin_38285131/article/details/78487782
//http://blog.csdn.net/yangtrees/article/details/19928191
/********************************************/
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
//#include <opencv2/nonfree/nonfree.hpp>
#include "opencv2/calib3d/calib3d.hpp"
#include <highgui.h>
#include <cv.h>
#include<vector>
#include<cmath>

//using namespace Eigen;
using namespace std;

//KNN 介绍：http://www.cnblogs.com/pinard/p/6061661.html

//Mat c =(Mat_<double>(3,3)<<1,0,0,0,1,0,0,0,1);

void match_features(cv::Mat query, cv::Mat train, vector< cv::DMatch>& matches)//Mat:OpenCV中表示矩阵，
{

    // cv::BFMatcher matcher(cv::NORM_L2, false); //BFMatcher class已经支持交叉验证, 命名为matcher； NORM_L2：归一化数组的(欧几里德)L2-范数， euclidean；返回欧几里距离
    //  vector< vector < cv::DMatch> > knn_matches; //vector<vector< 二维向量：定义二维向量 knn_matches
    //  matcher.knnMatch(query, train, knn_matches, 2);//knnMatch是matcher的一个属性。此属性包含在BFMatcher中。为每个descriptor查找K-nearest-matches;使用KNN-matching算法，令K=2。

    /////////////////////////////////////////////////////////////////////////////////

    cv::BFMatcher matcher(cv::NORM_L2, false);//定义一个匹配对象
    vector<vector< cv:: DMatch> > matches2;//定义一个容器用来装最近邻点和次近邻点
    //vector<DMatch>matches;//定义一个容器用来装符合条件的点
    matcher.match(query, train, matches);//进行匹配
    // const float ratio = 0.7;//将比值设为0.7  可以自己调节
    // matches.clear();//清空matches
    // matcher.knnMatch(query, train, matches2, 2);//运用knnmatch
    // for (int n = 0; n < matches2.size(); n++)
    // {
    //     DMatch& bestmatch = matches2[n][0];
    //     DMatch& bettermatch = matches2[n][1];
    //     if (bestmatch.distance < ratio*bettermatch.distance)//筛选出符合条件的点
    //     {
    //         matches.push_back(bestmatch);//将符合条件的点保存在matches
    //     }
    // }
    // cout << "match个数:" << matches.size() << endl;



    // matcher.knnMatch(query, train, matches, 2);
    //则每个match得到两个最接近的descriptor，返回两个最佳匹配。用交叉验证结合KNN的算法进行匹配，将匹配结果放入maches中

    // //获取满足Ratio Test的最小匹配的距离
    // float min_dist = FLT_MAX; //将float的最大值赋值给min_dist
    // for (int r = 0; r < knn_matches.size(); ++r)
    // {
    //     //Ratio Test；计算最接近距离和次接近距离之间的比值，当比值大于既定值时，才作为最终match
    //     if (knn_matches[r][0].distance > 0.6*knn_matches[r][1].distance)//思想：同种类型的东西，离得不会太远（在此项目中仿佛不可行？）
    //         continue;
    //
    //     float dist = knn_matches[r][0].distance;//存储了最近的距离进入dist
    //     if (dist < min_dist) min_dist = dist;//存入匹配对中的最短距离
    // }
    //
    // matches.clear();//清空matches
    // for (size_t r = 0; r < knn_matches.size(); ++r)
    // {
    //     //排除不满足Ratio Test的点和匹配距离过大的点
    //     if (
    //         knn_matches[r][0].distance > 0.6*knn_matches[r][1].distance ||
    //         knn_matches[r][0].distance > 5 * max(min_dist, 10.0f)
    //         )
    //         continue;
    //
    //     //保存匹配点
    //     matches.push_back(knn_matches[r][0]);
    // }
}

int main( int argc, char** argv )
{    cv::Mat featurepoints_lastimage =(cv::Mat_<int>(3, 3)<< 1,3,4,5,7,2,4,6,2);
     cv::Mat featurepoints_currentimage =(cv::Mat_<int>(3, 3)<< 4,2,6,8,2,5,7,8,2);



  // cv::Mat featurepoints_lastimage =(cv::Mat_<int>(10, 2)<< 280,209,624,207,457,194,113,214,268,188,462,204,142,192,324,181,538,241,209,228);
  // cv::Mat featurepoints_currentimage =(cv::Mat_<int>(10, 2)<< 280,209,624,207,457,194,113,214,268,188,462,204,142,192,324,181,538,241,209,228);
  vector< cv::DMatch >matches;
  //int matchedpair;

  //int matchedpair;
  match_features(featurepoints_lastimage, featurepoints_currentimage, matches);
  //std::cout<< matches<<std::endl;
  return 0;

}
