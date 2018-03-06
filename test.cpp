#include <iostream>
#include <string>
#include "math.h"
#include "limits.h"
#include "eigen3/Eigen/Dense"
#include "eigen3/Eigen/Core"


using namespace Eigen;
using namespace std;

int matchFeatures(int, int);
int computeresiduals(int, int, int, int);

int main( int argc, char** argv )
{ //Matrix<double, 3, 3> A;
  //A << 1, 2, 3,     // Initialize A. The elements can also be
    // 4, 5, 6,     // matrices, which are stacked along cols
    // 7, 8, 9;
  Matrix<double, 10, 3> featurepoints_lastimage;
  Matrix<double, 10, 3> featurepoints_currentimage;
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
     int A; int B;
      A = featurepoints_lastimage.cols();
      B = featurepoints_currentimage.rows();
 cout << A << endl;
 cout << B << endl;

 Matrix<int, A, 2> matchedpair;
}
 //cout << featurepoints_currentimage << endl;
