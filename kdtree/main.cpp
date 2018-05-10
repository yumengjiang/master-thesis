#include <fstream>
#include <string>
#include <vector>
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
#include "nanoflann.hpp"
#include <cstdio>
#include <stdio.h> 

using namespace cv;

#include <ctime>
#include <cstdlib>
#include <iostream>
#include "kdtree.h"
using namespace std;

int main()
{   




	KDTree::KDTree() 
{
	build_index({});
}
// コンストラクタ
KDTree::KDTree(const vector<PointCloud_Point>& pts)
{	
	build_index(pts);

}
// コンストラクタ
KDTree::~KDTree() = default;
// kd-tree インデックスの構築
void KDTree::build_index(const vector<PointCloud_Point>& pts)
{
	// 座標データを退避
	// データのクリア
	m_id_map.clear();
	m_point_cloud.pts.clear();
	// kd-treeの初期化
	m_kdtree.reset(new my_kd_tree_t(2, m_point_cloud, KDTreeSingleIndexAdaptorParams()));

	if (pts.size() > 0)
	{
		//座標データの追加し直し
		for (const PointCloud_Point& it : pts)
		{
			m_id_map[it.id] = m_point_cloud.pts.size();
			m_point_cloud.pts.push_back(it);
		}
		//kd-treeの再構築
		m_kdtree->addPoints(0, m_point_cloud.pts.size() - 1);
	}
}
//座標データの取得
PointCloud_Point & KDTree::get_point(const size_t id)  
{

	const auto iter = m_id_map.find(id);
#ifdef DEBUG
	if (iter != m_id_map.end())	cout << "ID: %d does not exists", id); << endl;
#endif // DEBUG
	size_t i = iter->second;
#ifdef DEBUG
	if (i < m_point_cloud.pts.size())	cout << "Index out bounds, index:", i << endl;
#endif // DEBUG
	return  m_point_cloud.pts[i];
}
//座標データの追加
size_t KDTree::add_point(const size_t id, const double x, const double y)
{
	const size_t last = m_point_cloud.pts.size();

	m_point_cloud.pts.emplace_back(id, x, y);	
	m_kdtree->addPoints(last, last);
	m_id_map[id] = last;
	return last;
}
// //半径距離による検索
vector<PointCloud_Point> KDTree::radius_search(const PointCloud_Point &pt, const double distance)
{
	// パラメータの設定
	const double query_pt[2]{ pt.x, pt.y };
	nanoflann::SearchParams params;
	params.sorted = false;
	// 検索結果出力パラメータの設置
	vector<pair<size_t, double>> indices_dists;
	RadiusResultSet<double, size_t> resultSet(distance, indices_dists);
	// 検索
	m_kdtree->findNeighbors(resultSet, query_pt, params);

	// 座標リストに変換
	int i = 0;
	vector<PointCloud_Point> result(resultSet.size());
	for (const pair<size_t, double>& it : resultSet.m_indices_dists)
	{
		result[i++] = m_point_cloud.pts[it.first];
	}
	return move(result);

}
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
		int res = sscanf(str.c_str(), "%lf\t%lf", &x, &y);

		if (res != 2)continue;
		// データをツリーに格納
		tree.add_point(++i, x, y);
	}
	fin.close();
	// 半径 座標の検索
	double dist = 2.0;
	PointCloud_Point pt{ ++i,6.0,12 };
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