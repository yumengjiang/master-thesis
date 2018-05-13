#include <iostream>
#include <fstream>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>


void blockMatching(cv::Mat &disp, cv::Mat imgL, cv::Mat imgR){
  cv::Mat grayL, grayR;

  cv::cvtColor(imgL, grayL, CV_BGR2GRAY);
  cv::cvtColor(imgR, grayR, CV_BGR2GRAY);

  cv::Ptr<cv::StereoBM> sbm = cv::StereoBM::create(); 
  sbm->setBlockSize(17);
  sbm->setNumDisparities(32);

  sbm->compute(grayL, grayR, disp);
  cv::normalize(disp, disp, 0, 255, CV_MINMAX, CV_8U);
}

void reconstruction(cv::Mat img, cv::Mat &Q, cv::Mat &disp, cv::Mat &rectified, cv::Mat &XYZ){
  cv::Mat mtxLeft = (cv::Mat_<double>(3, 3) <<
    350.6847, 0, 332.4661,
    0, 350.0606, 163.7461,
    0, 0, 1);
  cv::Mat distLeft = (cv::Mat_<double>(5, 1) << -0.1674, 0.0158, 0.0057, 0, 0);
  cv::Mat mtxRight = (cv::Mat_<double>(3, 3) <<
    351.9498, 0, 329.4456,
    0, 351.0426, 179.0179,
    0, 0, 1);
  cv::Mat distRight = (cv::Mat_<double>(5, 1) << -0.1700, 0.0185, 0.0048, 0, 0);
  cv::Mat R = (cv::Mat_<double>(3, 3) <<
    0.9997, 0.0015, 0.0215,
    -0.0015, 1, -0.00008,
    -0.0215, 0.00004, 0.9997);
  //cv::transpose(R, R);
  cv::Mat T = (cv::Mat_<double>(3, 1) << -119.1807, 0.1532, 1.1225);

  cv::Size stdSize = cv::Size(640, 360);
  int width = img.cols;
  int height = img.rows;
  cv::Mat imgL(img, cv::Rect(0, 0, width/2, height));
  cv::Mat imgR(img, cv::Rect(width/2, 0, width/2, height));

  cv::resize(imgL, imgL, stdSize);
  cv::resize(imgR, imgR, stdSize);

  //std::cout << imgR.size() <<std::endl;

  cv::Mat R1, R2, P1, P2;
  cv::Rect validRoI[2];
  cv::stereoRectify(mtxLeft, distLeft, mtxRight, distRight, stdSize, R, T, R1, R2, P1, P2, Q,
    cv::CALIB_ZERO_DISPARITY, 0.0, stdSize, &validRoI[0], &validRoI[1]);

  cv::Mat rmap[2][2];
  cv::initUndistortRectifyMap(mtxLeft, distLeft, R1, P1, stdSize, CV_16SC2, rmap[0][0], rmap[0][1]);
  cv::initUndistortRectifyMap(mtxRight, distRight, R2, P2, stdSize, CV_16SC2, rmap[1][0], rmap[1][1]);
  cv::remap(imgL, imgL, rmap[0][0], rmap[0][1], cv::INTER_LINEAR);
  cv::remap(imgR, imgR, rmap[1][0], rmap[1][1], cv::INTER_LINEAR);

  //cv::imwrite("2_left.png", imgL);
  //cv::imwrite("2_right.png", imgR);

  blockMatching(disp, imgL, imgR);
  cv::namedWindow("disp", cv::WINDOW_NORMAL);
  cv::imshow("disp", disp);
  cv::waitKey(0);

  rectified = imgL;

  cv::reprojectImageTo3D(disp, XYZ, Q);
  XYZ *= 0.001;
}


int main( int argc, char** argv )
{
  int start = std::stoi(argv[1]);
  int end = std::stoi(argv[2]);

  for(int i = start; i <= end; i++)
  {
    std::string pathName, line, x, y, label;
    pathName = "result/"+std::to_string(i);
    std::cout << pathName << std::endl;
    std::ifstream csvPath(pathName+".csv");
    cv::Mat img = cv::imread(pathName+"_stereo.png"); 
    cv::Mat Q, disp, rectified, XYZ;
    reconstruction(img, Q, disp, rectified, XYZ);
    cv::Vec3f point3D;
    std::ofstream myfile;
    myfile.open(pathName+"_stereo.csv");
    while (std::getline(csvPath, line)) 
    {  
      std::stringstream liness(line);  
      std::getline(liness, x, ',');  
      std::getline(liness, y, ','); 
      std::getline(liness, label, ',');
      point3D = XYZ.at<cv::Vec3f>(cv::Point(std::stoi(x)*2, std::stoi(y)*2));
      std::cout << x << " " << y << " " << point3D << std::endl;
      myfile << x+","+y+","+label+","+std::to_string(point3D[0])+","+std::to_string(point3D[1])+","+std::to_string(point3D[2])+"\n"; 
    }
    myfile.close();
  } 
}