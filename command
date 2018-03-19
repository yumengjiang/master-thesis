$ g++ helloworld.cpp -o helloworld  
$ ./helloworld

g++ -I tinydir-master/ multiview.cpp -std=c++14 -lpthread `pkg-config --libs --cflags opencv` -o multiview
