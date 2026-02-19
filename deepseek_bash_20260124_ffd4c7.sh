%%bash
cd /content/build

echo "Compiling tests..."
g++ -std=c++17 -O3 -march=native -I../include -I/usr/include/eigen3 \
    -I/usr/include -pthread ../test_vedic.cpp -o test_vedic

echo "Running tests..."
./test_vedic