//this function is used to match feature points between two images
//first, check the labels of two points, choose the feature points with the same color
//afterwards, calculate the residuals, choose the point with the least residuals which is the feature point
//featurepoints_lastimage is the central point of the cone, colorflag is the color of the corresponding cone
//featurepoints_lastimage size: N X 3 currentimage size: M x 3
#include <iostream>
#include <string>
#include "math.h"
#include "limits.h"
#include "eigen3/Eigen/Core"
#include "eigen3/Eigen/Dense"

//using namespace Eigen;
using namespace std;

//template <typename Derived>
//compute residuals
int computeResiduals(int beforeX, int afterX, int beforeY, int afterY)
{
  return pow((pow ((afterX-beforeX),2) + pow ((afterY-beforeY),2)),2);
}
//template <typename DerivedA, typename DerivedB>

Eigen::const Ref<const MatrixXf>  matchFeatures( Eigen::const Ref<const MatrixXf>& featurepoints_lastimage,
   Eigen::const Ref<const MatrixXf>& featurepoints_currentimage, int row1, int row2  )
{
  int threshold;
  threshold = 5;
  // featsize_last_col =featurepoints_lastimage.cols(); //column size of last image
  // featsize_last_row =featurepoints_lastimage.rows();//row size
  // featsize_current_col =featurepoints_currentimage.cols();//column size of current image
  // featsize_current_row =featurepoints_currentimage.rows();
  //Eigen::Matrix<int, featsize_last_row, 2> matchedpair; //featsize_last_row从这个函数进入，一直没有定义，然后这种<>的新参数的定义需要每个参数都是具体的值，而这个时候featsize_last_row并不是具体的值！
  //Eigen::MatrixXi matchedpair;
  Eigen::MatrixXi matchedpair;
  //typedef Eigen::Matrix<int,2,Dynamic> MatrixXd;
  //Eigen::Matrix <int, 2, Dynamic >matchedpair ; // 第二个参数填为5,作为测试
  //int matchedpair[featsize_last_row][2]={0}; //define matchedpair, default values are zeros
  // for(int i = 0; i < featsize_last_row; i=i+1) //check matchedpair one by one
  // {
  //   int x = featurepoints_lastimage (i,0); // x, y of the featurepoints
  //   int y = featurepoints_lastimage (i,1);
  //   //Eigen::Matrix<int, featsize_last_row, 2> matchedpair;
  //   Eigen::matchedpair(i,0)= i;
  //   Eigen::matchedpair(i,1)= 0; // first column is the feature in the first image
  //   int color = featurepoints_lastimage(i,2);
  //   //Eigen::residuals<int, featsize_current_row, 1> featurepoints_lastimage;   //错误1：同31行的错误一样！
  //   //Eigen::residuals<int, 5, 1> featurepoints_lastimage;  // 第二个参数填为5,作为测试，错误2：Eigen::residuals ,因为类名Eigen加上作用域::后面为取这个类下面的函数。在Eigen这个库中是不存在这个residuals函数的。
  //   //上面，我默认为你是创建一个新的矩阵
  //   Eigen::Matrix<int, Dynamic, 2> featurepoints_lastimage;
  //
  //   Eigen::Matrix<int, Dynamic> residuals; //find color
  //   for(int j=0; j < featsize_current_row; j=j+1)
  //   {    if (color == featurepoints_currentimage(j,2))
  //     //if (color == featurepoints_currentimage[j][2])// check if they are the same color
  //     {
  //       //residuals(j) = computeresiduals(x,featurepoints_currentimage(j,0),y, featurepoints_currentimage(j,1));//check residuals, find the smallest one, save it
  //       //上面这句话有个问题，因为你之前在46行定义的residuals为一个数组，而C/C++中数组取下标是array[i]
  //       residuals(j) = computeResiduals(x,featurepoints_currentimage(j,0),y, featurepoints_currentimage(j,1));//check residuals, find the smallest one, save it
  //       int min_res;
  //       min_res = threshold;
  //       if (residuals[j]<min_res)
  //       {
  //         min_res = residuals[j];
  //         matchedpair(i,1) = j;
  //       }
  //     }
  //   }
  // }
  return matchedpair;   //此处是返回一个矩阵，而你之前是int
}


int main( int argc, char** argv )
{
   //template <typename DerivedA, typename DerivedB>
  // Eigen::Matrix<double, 10, 3> featurepoints_lastimage;
  // Eigen::Matrix<double, 10, 3> featurepoints_currentimage;
  const Eigen::MatrixXf featurepoints_lastimage;
  const Eigen::MatrixXf featurepoints_currentimage;
  // const Eigen::DenseBase<DerivedB>& matchedpair;
  Eigen::MatrixXf matchedpair;
  featurepoints_lastimage << 299,198,1,
                                688,199,1,
                                143,201,1,
                                455,187,1,
                                455,195,2,
                                276,180,2,
                                161,183,2,
                                612,208,2,
                                234,211,3,
                                510,223,3;

  featurepoints_currentimage << 280,209,1,
                                    624,207,1,
                                    457,194,1,
                                    113,214,1,
                                    268,188,2,
                                    462,204,2,
                                    142,192,2,
                                    324,181,2,
                                    538,241,3,
                                    209,228,3;
  int featsize_last_col, featsize_last_row, featsize_current_col, featsize_current_row;
  featsize_last_col =featurepoints_lastimage.cols(); //column size of last image
  featsize_last_row =featurepoints_lastimage.rows();//row size
  featsize_current_col =featurepoints_currentimage.cols();//column size of current image
  featsize_current_row =featurepoints_currentimage.rows();
//int matchedpair;
matchedpair = matchFeatures(featurepoints_lastimage, featurepoints_currentimage, featsize_last_row, featsize_current_row);
std::cout << matchedpair<<std::endl;
return 0;

}
