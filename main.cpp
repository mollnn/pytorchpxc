#include <iostream>
#include <cstdio>
#include <vector>
#include <cstring>
#include <sstream>
#include "pytorchpxc.h"
using namespace std;

int main()
{
    PytorchProxyC nn;
    nn.eInit();
    nn.tInit();
    for (int i = 1; i <= 100; i++)
    {
        std::stringstream ss;
        double x = i * 0.01;
        double r = rand() * 0.00003;
        double y = x * x;
        ss << x << " " << r << " " << x << " " << r << " " << x << " " << y;
        nn.tPutData(ss.str());
    }
    nn.tFlushData();
    nn.tTrain();
    nn.tTrainPlus();
    nn.eLoad();
    cout << nn.eEval("1.0 0.0 1.0 0.0 1.0 0.0") << endl; // 1.0
    cout << nn.eEval("0.5 0.0 0.5 0.0 0.5 0.0") << endl; // 0.25
    cout << nn.eEval("0.0 0.0 0.0 0.0 0.0 0.0") << endl; // 0.0
    cout << nn.eEval("0.5 0.2 0.5 0.7 0.5 0.0") << endl; // 0.25
}