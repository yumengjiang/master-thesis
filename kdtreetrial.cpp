#include <fstream>
#include <string>
#include <vector>
#include <iostream>
#include <string>
#include <vector>
#include <utility>
#include <typeinfo>
#include <stdio.h>
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

#include <ctime>
#include <cstdlib>
#include <iostream>
#include "kdtree.h"
#include "pointcloud.h"
using namespace std;

void main( int argc, char** argv )
{
	// KDTree 
	KDTree tree;

	// 座標データの読み込み
	string src_path = "1.csv";
	ifstream fin(src_path);
	if (fin.bad()) {
		cout << "File Open Error:" << src_path << std::endl;
		return 1;
	}
	std::string str;
	size_t i=0;
	while (std::getline(fin, str))
	{
		double x,y;
		int res = sscanf_s(str.c_str(), "%lf\t%lf", &x, &y);

		if (res != 2)continue;
		// データをツリーに格納
		tree.add_point(++i, x, y);
	}
	fin.close();
	// 半径 座標の検索
	double dist = 2.0;
	PointCloud_Point pt{ ++i,2.0,3 };
	vector<PointCloud_Point> rad_result = tree.radius_search(pt,dist*dist);
	// 結果出力
	cout << " x  y" << endl;
	cout << "---------" << endl;
	for (const PointCloud_Point& it : rad_result)
	{
		cout <<" "<< it.x << " " << it.y << endl;
	}
	cout << endl;

	system("pause");
	return 0;
}