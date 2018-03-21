//this function is used to match feature points between two images
// first, check the labels of two points, choose the feature points with the same color
//afterwards, calculate the residuals, choose the point with the least residuals which is the feature point
//featurepoints_lastimage is the central point of the cone, colorflag is the color of the corresponding cone
//featurepoints_lastimage size: N X 3 currentimage size: M x 3
#include <iostream>
using namespace std;
#include <string>
#include "math.h"
#include "limits.h"
#include "eigen3/Eigen/Core"
#include "eigen3/Eigen/Dense"
using namespace Eigen;
template <typename Derived>




int matchFeatures(int, int);
int computeresiduals(int, int, int, int);

int main( int argc, char** argv )
{
  Matrix<double, 10, 3> featurepoints_lastimage;
  Matrix<double, 10, 3> featurepoints_currentimage;
  Matrix<double, 10, 2> matchedpair;
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
//int matchedpair;
matchedpair = matchFeatures(featurepoints_lastimage, featurepoints_currentimage);
std::cout << matchedpair<<std::endl;
return 0;
}
#include "eigen3/Eigen/Core"
#include "eigen3/Eigen/Dense"
using namespace Eigen;
template <typename Derived>
int matchFeatures(const DenseBase<Derived> &featurepoints_lastimage, const DenseBase<Derived> &featurepoints_curentimage )
{
  int featsize_last_col, featsize_last_row, featsize_current_col, featsize_current_row,  threshold;

  threshold = 5;
  featsize_last_col = sizeof(featurepoints_lastimage[0])/sizeof(int); //column size of last image
  featsize_last_row = (sizeof(featurepoints_lastimage)/sizeof(int))/(sizeof(featurepoints_lastimage[0])/sizeof(int));//row size
  featsize_current_col = sizeof(featurepoints_currentimage[0])/sizeof(int);//column size of current image
  featsize_current_row = (sizeof(featurepoints_currentimage)/sizeof(int))/(sizeof(featurepoints_currentimage[0])/sizeof(int));
  Eigen::Matrix<int, featsize_last_row, 2> matchedpair;
  //int matchedpair[featsize_last_row][2]={0}; //define matchedpair, default values are zeros
  for(int i = 0; i < featsize_last_row; i=i+1) //check matchedpair one by one
  {
    int x = featurepoints_lastimage (i,0); // x, y of the featurepoints
    int y = featurepoints_lastimage (i,1);
    Matrix<double, 10, 2> matchedpair;
    matchedpair(i, 0)= i; // first column is the feature in the first image
    int color = featurepoints_lastimage(i,2);
    Eigen::residuals<int, featsize_current_row, 1> featurepoints_lastimage;
    int residuals[featsize_current_row]; //find color
    for(int j=0; j < featsize_current_row; j=j+1)
    {    if (color == featurepoints_currentimage(j,2))
      //if (color == featurepoints_currentimage[j][2])// check if they are the same color
      {
        residuals(j) = computeresiduals(x,featurepoints_currentimage(j,0),y, featurepoints_currentimage(j,1));//check residuals, find the smallest one, save it
        int min_res;
        min_res = threshold;
        if (residuals(j)<min_res)
        {
          min_res = residuals(j);
          matchedpair(i,1) = j;
        }
      }
    }
  }
  return matchedpair;
}
//compute residuals
#include "eigen3/Eigen/Core"
#include "eigen3/Eigen/Dense"
using namespace Eigen;
template <typename Derived>
const computeresiduals(const DenseBase<Derived> &beforex, const DenseBase<Derived> &afterx, const DenseBase<Derived> &beforey, const DenseBase<Derived> &aftery)
{
  const residuals;
  residuals = pow((pow ((afterx-beforex),2) + pow ((aftery-beforey),2)),2);
  return residuals;
}
