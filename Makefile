default all:
	$(CXX) Example\ MNIST/example1.cpp -I . -o example1 -O3
	$(CXX) Example\ MNIST/example2.cpp -I . -o example2 -O3
