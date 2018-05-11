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
#include <nanoflann.hpp>
#include <map>
#include "Eigen/Dense"

using namespace cv;
using namespace std;
using namespace nanoflann;
double radius=0.5;
unsigned int max_neighbours = 8;

struct Points
{
	Point2d pt;
	int id;
	int group;
};

void gather_points(//初始化
	Mat source,
	vector<float> vecQuery,
	vector<int> &vecIndex,
	vector<float> &vecDist
	)
{  
	//source.convertTo(source,CV_32F);
	flann::KDTreeIndexParams indexParams(2);
	flann::Index kdtree(source, indexParams); //此部分建立kd-tree索引同上例，故不做详细叙述

	/**预设knnSearch所需参数及容器**/
	//unsigned queryNum = 7;//用于设置返回邻近点的个数
	//eryNum);//存放返回的点索引
	//vector<int> vecIndex;
	//vector<float> vecDist(queryNum);//存放距离
	//vector<float> vecDist;
	flann::SearchParams params(1024);//设置knnSearch搜索参数

	/**KD树knn查询**/
	// vecQuery[0] = 3; //查询点x坐标
	// vecQuery[1] = 3; //查询点y坐标
	//kdtree.knnSearch(vecQuery, vecIndex, vecDist, queryNum, params);
	kdtree.radiusSearch(vecQuery, vecIndex, vecDist, radius, max_neighbours, params);
}

bool GreaterSort (Points a, Points b)
{ 
	return (a.pt.x < b.pt.x); 
} 

bool setRange(int x)

{

    return (x != 0);

}

bool comparegroup(Points a)

{

    return (a.group == -1);

}

int findMax(vector<int> vec) {
    //int max = -999;
    int max = INT_MIN;
    for (auto v : vec) {
        if (max < v) max = v;
    }
    return max;
}

float getPositionOfMax(vector<float> vec, float max) {
    auto distance = find(vec.begin(), vec.end(), max);
    return distance - vec.begin();
}


int main( int argc, char** argv )
// {   template <typename num_t>
{   //read pixels
	double t = (double)getTickCount();  
	string data_path = argv[1]; 
	cout<<"one"<<endl;
	ifstream csvPath ( data_path+"/88.csv" );
	Mat img=imread ( data_path+"/88.png" );
	cout<<"two"<<endl;
	vector<Points> data1;
	vector<Point2f> data;
	string line, x, y, color, X, Y, z; 
			// Mat imgLast, imgNext, outImg;
		    // Mat img = imread("result/"+to_string(i)+".png");
    cout<<"three"<<endl; 
    int i=0;
	while (getline(csvPath, line)) 
	{  
		stringstream liness(line);  
		getline(liness, X, ',');  
		getline(liness, Y, ',');
		getline(liness, color, ',');  
		getline(liness, x, ',');
		getline(liness, z, ',');  
		getline(liness, y, ',');
		
		Point2d ptl(stod(x),stod(y));
		Points ptll{ptl,i,-1};
		data1.push_back(ptll);
		data.push_back(ptl);
		i++;
	}

    


//vector<int> vecIndex(qu
	Mat source = cv::Mat(data).reshape(1);

	//gather_points(source, vecQuery, vecIndex, vecDist);
	vector<Points> finaldata;
	vector<float> datax;
	// sort(data1.begin(),data1.end(),GreaterSort);// sort according to x axis, from small to large 
	int flag=-1; //initiate flag
	int resultSize = 1000;
	double resultResize = 50;
	int draw=0;
	cv::RNG rng(time(0));
	Mat result = Mat::zeros(resultSize, resultSize, CV_8UC3);
	for (int j=0;j<data1.size()-1;j++)
	{   /**建立kd树索引**/
	// int j=2;
		cout<<"loop "<<j<<endl;
		
		Point2f pt(data1[j].pt.x, data1[j].pt.y);
		vector<float> vecQuery;//存放 查询点 的容器（本例都是vector类型）
		vecQuery.push_back(pt.x);
		vecQuery.push_back(pt.y);
		vector<int> vecIndex;
		vector<float> vecDist;
        
		
		gather_points(source, vecQuery, vecIndex, vecDist);//kd tree finish; find the points in the circle with point center vecQuery and radius, return index in vecIndex
		int num = count_if(vecIndex.begin(), vecIndex.end(), setRange);//if there is one lonely point, build it as an individual group
		for (int j=1; j<vecDist.size();j++){
			if (vecIndex[j]==0&&vecIndex[j+1]!=0){
				num=num+1;
			}
		}
		if (num ==0){

			if (data1[j].group == -1)
		  { flag++;
			data1[j].group = flag;
			data1[j].id = j;
			cout<<j<<" type 1"<<" "<<data1[j].pt.x<<","<<data1[j].pt.y<<" group "<<flag<<endl;
			float X1 = data1[j].pt.x;
			float Y1 = data1[j].pt.y;
			finaldata.push_back(data1[j]);
			circle(result, Point(200*(X1+3),100*Y1), 3, Scalar(0, 255, 255), -1);
			draw=draw+1;
		  }
//(200*(X+5),100*Y)
		}
		else
		{   vector<Points> groupall;
			vector<int> filteredindex;
			// Points finaltemp;
			Point2f finaltemp_points;
			float finaltemp_x=0;
			float finaltemp_y=0;
			for (int v=0; v<num; v++)
			{
                 groupall.push_back(data1[vecIndex[v]]);
             	 filteredindex.push_back(vecIndex[v]);
            }
		

		  int nogroup = count_if(groupall.begin(), groupall.end(), comparegroup);
		  cout<<"nogroup "<<nogroup<<endl;
		  if (nogroup>0){
		  	flag++;
		  Scalar color1=Scalar(rng.uniform(0,255),rng.uniform(0,255),rng.uniform(0,255));
		  cout<<"here1"<<endl;
		  // int indexcount=0;
		  for (int k=0; k<filteredindex.size(); k++)
		  { 
		  	cout<<"here2"<<endl;
		   if (data1[filteredindex[k]].group == -1)
		   { 
		  	data1[filteredindex[k]].group = flag;
		  	data1[filteredindex[k]].id = j;
		  	float X1 = data1[filteredindex[k]].pt.x;
            float Y1 = data1[filteredindex[k]].pt.y;
		  	cout<<k<<" type 2"<<" "<<data1[vecIndex[k]].pt.x<<","<<data1[vecIndex[k]].pt.y<<" group "<< data1[vecIndex[k]].group<<endl;
		  	circle(result, Point(200*(X1+3),100*Y1), 2, color1, -1);
		  	//(200*(X+3),100*Y)
		  	draw=draw+1;
		  	finaltemp_x=finaltemp_x + data1[vecIndex[k]].pt.x;
		  	finaltemp_y=finaltemp_y + data1[vecIndex[k]].pt.y;
		  	// indexcount++;


		   }

		  }
		  finaltemp_points.x = finaltemp_x / nogroup;
		  finaltemp_points.y = finaltemp_y / nogroup;
		  Points finaltemp{finaltemp_points, j, flag};
		  finaldata.push_back(finaltemp);
		  float X2 = finaltemp_points.x;
		  float Y2 = finaltemp_points.y;
		  circle(result, Point(200*(X2+3),100*Y2), 3, Scalar(0, 255, 255) , -1);


          
		}


		}


	}
	// vector<int> count_same_group;
	// count_same_group.resize(data1.size());
	// for (int i = 0; i < data1.size(); ++i){
	// 			if(correspond_struct_idx[i][j] == k)
	// 				count_same_structure[k]++;
	// 		}
	// 	}
	// 	cout << "\n";
	
	// int group_zero_all = count_if(data1.begin(), data1.end(), comparegroup);
	//group the missing points, need double-check!!!
	// for (int q=0; q<data1.size(); q++){
	// 	if (data1[q].group == 0 && data1[q].id!= 0)
	// 	{   flag++;
	// 		data1[q].group=flag;
	// 		float X = data1[q].pt.x;
	// 		float Y = data1[q].pt.y;
	// 		// cout<<" type 3"<<" group "<<flag<<endl;
	// 		// circle(result, Point (60*(X+10), 50*Y), 3, Scalar(rng.uniform(0,255),rng.uniform(0,255),rng.uniform(0,255)), -1);

	// 	}


	// }
	for (int p=0; p<data1.size(); p++){
		cout<<p<<" "<<data1[p].pt.x<<" "<<data1[p].pt.y<<" group "<<data1[p].group<<endl;
	}
    cout<<"center datas"<<endl;
	for (int r=0; r<finaldata.size(); r++){
        cout<<"NO."<<r<<" "<<finaldata[r].pt.x<<","<<finaldata[r].pt.y<<endl;


	}
	t = ((double)getTickCount() - t)/getTickFrequency(); 
	cout<<"draw"<<draw<<endl;
	flip(result, result, 0);
    namedWindow("result", WINDOW_NORMAL);
	imshow("result", result);
	waitKey(0);

	
	cout << "total time"<<t<<" sec"<<endl;
	// cout << “time:”<< t << “sec” << endl; //输出运行时间

	//std::vector<double>::iterator biggest = std::min_element(std::begin(datax), std::end(datax));


//请注意这句的逻辑：由先前生成的kdtree索引对象调用knnSearch()函数，进行点的knn搜索
  // vector<Point2f> data;
  // data.push_back(Point2f(1.0f, 1.0f));
  // data.push_back(Point2f(2.5f, 3.0f));
  // data.push_back(Point2f(3.0f, 3.0f));
  // data.push_back(Point2f(3.5f, 1.0f));
  // data.push_back(Point2f(2.5f, 3.0f));
  // data.push_back(Point2f(2.5f, 3.0f));
  // data.push_back(Point2f(10.0f, 10.0f));
  // data.push_back(Point2f(11.0f, 12.0f));
  // data.push_back(Point2f(11.5f, 9.0f));
  // data.push_back(Point2f(12.0f, 10.0f));
  // data.push_back(Point2f(20.0f, 20.0f));
  // data.push_back(Point2f(22.0f, 21.0f));
  // data.push_back(Point2f(22.5f, 18.0f));
  // data.push_back(Point2f(23.0f, 20.5f));














}