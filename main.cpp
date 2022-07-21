#include <bits/stdc++.h>
#include "pytorchpxc.h"
using namespace std;

int main()
{
    PytorchProxyC nn;
    nn.eInit();
    for (int i = 1; i <= 100; i++)
    {
        std::stringstream ss;
        double x = i * 0.01;
        double y = sin(x);
        ss << x << " " << y;
        nn.tPutData(ss.str());
    }
    nn.tFlushData();
    nn.tTrain();
    nn.tTrainPlus();
    nn.eLoad();
    cout << nn.eEval("0.2 0.0") << endl;
    cout << nn.eEval("0.7 0.0") << endl;
}