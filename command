g++ helloworld.cpp -o helloworld  
./helloworld

g++ -I tinydir-master/ multiview.cpp -std=c++14 -lpthread `pkg-config --libs --cflags opencv` -o multiview
g++ reconstruction_3D1.cpp -std=c++14 -lpthread `pkg-config --libs --cflags opencv` -o reconstruction_3D
g++ new3D.cpp -std=c++14 -lpthread `pkg-config --libs --cflags opencv` -o new3D
