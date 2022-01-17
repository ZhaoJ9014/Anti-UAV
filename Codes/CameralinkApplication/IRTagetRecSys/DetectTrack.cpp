#include "DetectTrack.h"

#pragma once
/*
DroneTrack::DroneTrack()
{
	
	Py_SetPythonHome(L"C:\\Users\\uavproject\\Anaconda3");
	Py_Initialize();
	import_array();
	pModule = PyImport_ImportModule("detective2_vs_test2");
	pDict = PyModule_GetDict(pModule);

	//定义python函数
	//获得检测和追踪类
	//DetectClass = PyDict_GetItemString(pDict, "DroneDetection");
	//DetectIns = PyInstanceMethod_New(DetectClass);
	//TrackClass = PyDict_GetItemString(pDict, "DroneTracker");
	//TrackIns = PyInstanceMethod_New(TrackClass);

	//pFunc = PyDict_GetItemString(pDict, "global_init");
	//PyObject_CallObject(pFunc, NULL);
	
	//pFunc = PyDict_GetItemString(pDict, "recvimg1");
	
	pArgs = PyTuple_New(1);
	DetectClass = PyDict_GetItemString(pDict, "DroneDetection");
	DetectIns = PyInstanceMethod_New(DetectClass);
	pIns = PyObject_CallObject(DetectIns, nullptr);

	//PyObject* myint = Py_BuildValue("i", 22);
	//PyObject_CallMethod(pIns, "test_detection", nullptr);
}*/


/*
DroneTrack::~DroneTrack()
{
	Py_Finalize();
}

void DroneTrack::CvtPyObject(void* ImgBuff)
{
	//pArgs = PyTuple_New(1);
	npy_intp Dim[2] = {IMG_IR_HEIGHT, IMG_IR_WIDTH};
	pMat = PyArray_SimpleNewFromData(2, Dim, NPY_UBYTE, ImgBuff);
	PyTuple_SetItem(pArgs,0, pMat);    
	//PyObject_CallObject(pFunc, pArgs);
	//DetectClass = PyDict_GetItemString(pDict, "DroneDetection");
	//DetectIns = PyInstanceMethod_New(DetectClass);
	//PyObject *pIns = PyObject_CallObject(DetectIns, nullptr);
	//PyObject* myint = Py_BuildValue("i", 22);
	PyObject_CallMethod(pIns, "test_frame", "O", pArgs);
    //PyObject_CallMethod(pIns, "forward","O", pArgs);
	
}

void DroneTrack::RecvSignal()
{

}
*/
