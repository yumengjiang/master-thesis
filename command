g++ main.cpp -std=c++14 -I ../tiny-dnn/ -lpthread `pkg-config --libs --cflags opencv` -lboost_system -lboost_filesystem -fopenmp -g -O2 -ltbb -o main
