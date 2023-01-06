#include <iostream>
#include <vector>
#include <opencv2/imgproc.hpp>
#include "HogDescriptor.h"

using namespace cv;
using namespace std;


int main(){
    std::vector<float> descriptors;
    Mat img(164, 164, 0, Scalar(9)),src;

    randu(img, 0, 66);
    img(Range(5, 55), Range(5, 55)).setTo(0);
    resize(img, src, Size(61, 61), 0, 0, INTER_LINEAR_EXACT);

    Ptr<HOGDescriptor> hog = new HOGDescriptor(Size(60, 60), Size(10, 10), Size(5, 5), Size(5, 5), 9);
    hog->compute(src, descriptors, Size(1, 1), Size(3, 3));

    for (auto d: descriptors) {
        cout << d << endl;
    }
}
