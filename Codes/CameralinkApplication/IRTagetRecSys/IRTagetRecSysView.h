
// IRTagetRecSysView.h : CIRTagetRecSysView 类的接口
//

#include "TrackWnd.h"
#include "TrackCtrlDlg.h"


//Dalsa CameraLink图像采集库支持
#include "SapClassBasic.h"
//#include "SapClassGui.h"
#pragma comment(lib,"SapClassBasic.lib")
//#pragma comment(lib,"SapClassGui.lib")
#pragma once


//采集缓冲区个数////////////////////////////////////////////////////////////////////////////////////////
#define MAXBUFFNUM  10

class CIRTagetRecSysView : public CView
{
protected: // 仅从序列化创建
	CIRTagetRecSysView();
	DECLARE_DYNCREATE(CIRTagetRecSysView)

// 特性，窗口变量
public:
	CTrackWnd         m_wndTrack;  //跟踪图像显示区
	CMFCTabCtrl       m_wndCtrlTab;      //控制Tab窗口
 	CTrackCtrlDlg     m_dlgTrackCtrl;    //控制页面

//特性，标志变量
public:
	BYTE            m_bRunning;
	double          m_fCapFrameTime;    //帧响应时间

//特性，图像变量
public:
	//IR图像缓冲区
	UINT   m_nCurIRCaptureNo;
	BYTE*  m_pbyIrBuffs[MAXBUFFNUM]; //跟踪8bit数据，MAXBUFFNUM帧，640*512
	BYTE*  m_pbyIr;            //跟踪8bit数据，单帧，已垂直镜像

	//RGB图像缓冲区
	UINT   m_nCurRGBCaptureNo;
	BYTE*  m_pbyRGBBuffs[MAXBUFFNUM]; //跟踪8bit数据，MAXBUFFNUM帧，640*480
	BYTE*  m_pbyRGB; 




public:
#ifndef HARDWARE_OK
	static ULONG WINAPI IRCapThread(PVOID);//创建仿真采集线程（无硬件时模拟）
#endif
	static ULONG WINAPI IRProcessThread(PVOID);//算法及显示处理自定义线程
	static ULONG WINAPI RECVMessage(PVOID);
	static void  IRCapCallback(SapXferCallbackInfo *pInfo);
	static void  RGBCapCallback(SapXferCallbackInfo *pInfo);
public:
	void SetWndSize();

	//Dalsa CameraLink热像采集卡硬件操作++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	bool            m_bIRChanLinkOk;
	SapAcquisition* m_pSapAcqIR;     //Track接口采集句柄
	SapBuffer*      m_pBuffersIR;    //Track接口采集buffer
	SapTransfer*    m_pXferIR;       //Track接口传输句柄
	static void     SignalCallbackIR(SapAcqCallbackInfo *pInfo);

    bool            m_bRGBChanLinkOk;
	SapAcquisition* m_pSapAcqRGB;     //Track接口采集句柄
	SapBuffer*      m_pBuffersRGB;    //Track接口采集buffer
	SapTransfer*    m_pXferRGB;       //Track接口传输句柄
	static void     SignalCallbackRGB(SapAcqCallbackInfo *pInfo);

	
	BOOL            InitSapCom(CString strIRConfig,CString strRGBConfig);
	BOOL            CreateSapObjects(); //创建句柄
	BOOL            DestroySapObjects();//销毁句柄
	void            GetSignalStatus();  //获取连接状态

// 重写
public:
	virtual void OnDraw(CDC* pDC);  // 重写以绘制该视图
	virtual BOOL PreCreateWindow(CREATESTRUCT& cs);
protected:

// 实现
public:
	virtual ~CIRTagetRecSysView();
#ifdef _DEBUG
	virtual void AssertValid() const;
	virtual void Dump(CDumpContext& dc) const;
#endif

public:
	void DrawSearchRuler();

// 生成的消息映射函数
protected:
	DECLARE_MESSAGE_MAP()
public:
	afx_msg int OnCreate(LPCREATESTRUCT lpCreateStruct);
	virtual void OnInitialUpdate();
	afx_msg void OnSize(UINT nType, int cx, int cy);
	BOOL   m_bExe;
	//afx_msg LRESULT  PyMeaasgeFun(WPARAM wParam, LPARAM lParam);
	//void pyfunc(void* imgBuff);
	afx_msg BOOL OnEraseBkgnd(CDC* pDC);
	void ButtonCMD();
	afx_msg LRESULT  TrackCMDMessage(WPARAM wParam, LPARAM lParam);
};

#ifndef _DEBUG  // IRTagetRecSysView.cpp 中的调试版本
inline CIRTagetRecSysDoc* CIRTagetRecSysView::GetDocument() const
   { return reinterpret_cast<CIRTagetRecSysDoc*>(m_pDocument); }
#endif



