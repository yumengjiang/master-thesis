#include <iostream>
#include <string>
#include "math.h"
#include "limits.h"
#include "eigen3/Eigen/Core"
#include "eigen3/Eigen/Dense"

using namespace Eigen;
using namespace std;
template<typename Derived>
MatrixXi matchFeatures(MatrixXi& x, MatrixXi& y)

//MatrixXi <double, Dynamic, 3>  matchFeatures(const Eigen::MatrixBase<Derived>& x)
{ MatrixXi z;
  z=x;
  return z;
}
//MatrixXi <double, Dynamic, 3> matchFeatures( MatrixXi<double, Dynamic, 3> featurepoints_lastimage,  MatrixXi<double, Dynamic, 3>featurepoints_currentimage )





int main( int argc, char** argv )
{
  Matrix<int, 10, 3> featurepoints_lastimage;
  Matrix<int, 10, 3> featurepoints_currentimage;
  MatrixXi z;
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
//Matrix<double, 10,3> matchedpair;
//matchedpair = matchFeatures(featurepoints_lastimage, featurepoints_currentimage);
//int z;
z= matchFeatures(featurepoints_lastimage, featurepoints_currentimage);
cout<<z<<endl;
//std::cout << matchedpair<<std::endl;
return 0;
}
