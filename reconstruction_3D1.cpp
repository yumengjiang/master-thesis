//this function is used to match feature points between two images
//first, check the labels of two points, choose the feature points with the same label
//afterwards, calculate the residuals, choose the point with the least residuals which is the feature point
//featureLast is the central point of the cone, labelflag is the label of the corresponding cone
//featureLast size: N X 3 currentimage size: M x 3
#include <iostream>
#include <string>
#include "math.h"
#include "limits.h"
#include "eigen3/Eigen/Core"
#include "eigen3/Eigen/Dense"

using namespace std;

double threshold = 50;

double computeResiduals(int beforeX, int afterX, int beforeY, int afterY){
  return pow((pow ((afterX-beforeX),2) + pow ((afterY-beforeY),2)),0.5);
}

Eigen::MatrixXi matchFeatures(Eigen::MatrixXi matchedLast, 
                  Eigen::MatrixXi featureLast, Eigen::MatrixXi featureNext){
  double res, minRes;
  int featureLastRow = featureLast.rows();
  int featureNextRow = featureNext.rows();
  int matchedLastRow = matchedLast.rows();
  int matchedLastCol = matchedLast.cols();
  int count = matchedLastRow;
  int index, indexCandidate;
  Eigen::MatrixXi matchedNextTmp(matchedLastRow+featureNextRow, matchedLastCol+1);
  for(int i = 0; i < matchedLastRow; i++)
    for(int j = 0; j < matchedLastCol; j++)
      matchedNextTmp(i, j) = matchedLast(i, j);

  for(int i = 0; i < featureNextRow; i++){
    minRes = threshold;
    for(int j = 0; j < featureLastRow; j++){
      if(featureNext(i,2) == featureLast(j,2)){
        res = computeResiduals(featureNext(i,0), featureLast(j,0), featureNext(i,1), featureLast(j,1));//check residuals, find the smallest one, save it
        if(res < minRes){
          minRes = res;
          index = j;
        }
      }
    }
    if(minRes == threshold){
      matchedNextTmp(count++, matchedLastCol) = i+1;
    } 
    else{
      indexCandidate = matchedNextTmp(index, matchedLastCol)-1;
      if(indexCandidate+1 == 0){
        matchedNextTmp(index, matchedLastCol) = i+1;
      }
      else{
        res = computeResiduals(featureNext(i,0), featureLast(indexCandidate,0), featureNext(i,1), featureLast(indexCandidate,1));//check residuals, find the smallest one, save it
        if(res < minRes){
          matchedNextTmp(count++, matchedLastCol) = i+1;
        }
        else{
          matchedNextTmp(index, matchedLastCol) = i+1;
          matchedNextTmp(count++, matchedLastCol) = indexCandidate+1;
        }
      }
    }
  }
  Eigen::MatrixXi matchedNext(count, matchedLastCol+1);
  for(int i = 0; i < count; i++)
    for(int j = 0; j < matchedLastCol+1; j++)
      matchedNext(i, j) = matchedNextTmp(i, j);
  return matchedNext;
}


int main( int argc, char** argv )
{
  Eigen::MatrixXi featureLast(10, 3);
  Eigen::MatrixXi featureNext(10, 3);

  featureLast << 299,198,1,
                  688,199,1,
                  143,201,1,
                  455,187,1,
                  455,195,2,
                  276,180,2,
                  161,183,2,
                  612,208,2,
                  234,211,3,
                  510,223,3;

  featureNext << 280,209,1,
                  624,207,1,
                  457,194,1,
                  113,214,1,
                  268,188,2,
                  462,204,2,
                  142,192,2,
                  324,181,2,
                  538,241,3,
                  209,228,3;

  // featureNext << 299,198,1,
  //                 688,199,1,
  //                 143,201,1,
  //                 455,187,1,
  //                 455,195,2,
  //                 276,180,2,
  //                 161,183,2,
  //                 612,208,2,
  //                 234,211,3,
  //                 1,1,1,
  //                 1,1,1,
  //                 510,223,3;

  int featureLastRow = featureLast.rows();
  Eigen::MatrixXi matchedNext, matchedLast(featureLastRow, 1);
  for(int i = 0; i < featureLastRow; i++){
    matchedLast(i, 0) = i+1;
  }

  matchedNext = matchFeatures(matchedLast, featureLast, featureNext);
  std::cout << matchedNext << std::endl;
  return 0;

}
