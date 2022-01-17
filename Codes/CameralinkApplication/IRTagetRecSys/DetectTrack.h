//#include <Python.h>
#include <WinSock2.h>
#include <iostream>
#include <string>
#include <Windows.h>
#include <stdio.h>
#include <tchar.h>
//#include <numpy/arrayobject.h>

#pragma once

#define IMG_IR_HEIGHT 512
#define IMG_IR_WIDTH 640

#define IMG_RGB_HEIGHT 1080
#define IMG_RGB_WIDTH 1920
//定义发送结构体
typedef struct _GFKDTrackAimInfo
{
	int msgCode;
	int nAimType;
	int nAimX;    //目标左上角
	int nAimY;
	int nAimW;	  //目标高宽
	int nAinH;
	int nTrackType;
	int nState;
}GFKDTrackAimInfo;
//定义接受结构体
typedef struct _GFKDTrackCMD
{
	int msgCode;
	int bIrOrCCD;
	int nFieldAngle;
	int nVal;
	int bStartOrStop;
	int nCenterY;
	int nCenterX;
}GFKDTrackCMD;

/*
class DroneTrack
{
public:
	GFKDTrackAimInfo AimInfo;
	GFKDTrackCMD TrackCMD;
	PyObject *pModule, *pDict;
	PyObject *pMat,*pArgs, *pFunc;
	PyObject *DetectClass, *TrackClass;
	PyObject *DetectIns, *TrackIns;
	
	PyObject *myint;

public:
	PyObject *pIns;
	//构造函数用于自动创建python运行环境，析构函数用于释放python环境
	DroneTrack();
	~DroneTrack();
	void RecvSignal();         //接受结构体信号
	void SendCoord();          //发送结构体信息
	void CvtPyObject(void* ImgBuff);        //C++图像转换成Python对象
	void DroneDetection();	   //检测线程函数，调用python检测
	void DroneTracking();	   //追踪线程函数，调用python追踪
};
*/