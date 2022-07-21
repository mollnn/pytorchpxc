#ifndef _PYTORCH_PXC_H_
#define _PYTORCH_PXC_H_

#include <bits/stdc++.h>
#include <Python.h>

class PytorchProxyC
{
public:
    PyObject *pName, *pModule, *pDict, *pFunc, *pArgs, *pValue, *pRet;
    std::vector<std::string> train_data_buffer;
    double res;

    void eInit()
    {
        Py_SetPythonHome(L"D:\\Anaconda3\\envs\\pytorch_gpu");
        Py_Initialize();
        PyRun_SimpleString("import sys");
        PyRun_SimpleString("sys.path.append('./')");
        std::string sr = "eval";
        pModule = PyImport_ImportModule(sr.c_str());
        pDict = PyModule_GetDict(pModule);
    }

    void eLoad()
    {
        pFunc = PyDict_GetItemString(pDict, "load");
        pArgs = PyTuple_New(0);
        pRet = PyObject_CallObject(pFunc, pArgs);
    }

    void eRelease()
    {
        Py_DECREF(pName);
        Py_DECREF(pArgs);
        Py_DECREF(pModule);
        Py_Finalize();
    }

    std::string eEval(const std::string &input)
    {
        pFunc = PyDict_GetItemString(pDict, "eval");
        pArgs = PyTuple_New(1);
        PyTuple_SetItem(pArgs, 0, Py_BuildValue("s", input.c_str()));
        pRet = PyObject_CallObject(pFunc, pArgs);
        PyArg_Parse(pRet, "d", &res);
        return std::to_string(res);
    }

    void tPutData(const std::string &train_data)
    {
        train_data_buffer.push_back(train_data);
    }

    void tFlushData()
    {
        std::ofstream ofs("data.txt");
        for (auto &i : train_data_buffer)
        {
            ofs << i << std::endl;
        }
        ofs.close();
        train_data_buffer.clear();
    }

    void tTrain()
    {
        system("python train.py");
    }

    void tTrainPlus()
    {
        system("python train1.py");
    }
};

#endif