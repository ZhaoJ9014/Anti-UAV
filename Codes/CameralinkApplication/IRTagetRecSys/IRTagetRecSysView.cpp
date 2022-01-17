
// RGBTagetRecSysView.cpp : CIRTagetRecSysView 类的实现
//

#include "stdafx.h"
// SHARED_HANDLERS 可以在实现预览、缩略图和搜索筛选器句柄的
// ATL 项目中进行定义，并允许与该项目共享文档代码。
#ifndef SHARED_HANDLERS
#include "IRTagetRecSys.h"
			#endif
#include "IRTagetRecSysDoc.h"
#include "IRTagetRecSysView.h"
#include "DetectTrack.h"
//#include <Python.h>
#include <iostream>
#include <WinSock2.h>

#pragma comment(lib,"ws2_32.lib")
#include <opencv2/opencv.hpp>
#include <core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#ifdef _DEBUG
#define new DEBUG_NEW
#endif
#define OPENCVMESSAGE WM_USER+100

using namespace std;
int udpEndSignal(char ipAddr[] , int Port);
// CIRTagetRecSysView

//定义动态全局的DetectTrack
//static DroneTrack* drone_det = new DroneTrack;
//接收 结构体
GFKDTrackCMD trackCMD;
BOOL g_bOccupied = FALSE;

//数据指针
BYTE *IMGData = NULL;

IMPLEMENT_DYNCREATE(CIRTagetRecSysView, CView)

BEGIN_MESSAGE_MAP(CIRTagetRecSysView, CView)
	ON_WM_CONTEXTMENU()
	ON_WM_RBUTTONUP()
	ON_WM_CREATE()
	ON_WM_SIZE()
	ON_WM_ERASEBKGND()
	//ON_MESSAGE(OPENCVMESSAGE,PyMeaasgeFun)
	ON_MESSAGE(OPENCVMESSAGE,TrackCMDMessage)
END_MESSAGE_MAP()

// CIRTagetRecSysView 构造/析构

/*
void CIRTagetRecSysView::ButtonCMD()
{	
		CIRTagetRecSysView* pView = (CIRTagetRecSysView*)(GetParent()->GetParent());
#ifdef HARDWARE_OK
	if(trackCMD.bIrOrCCD == 1){
		pView->m_wndTrack.m_dibBMP.DeleteObject();
		pView->m_wndTrack.m_nImageWidth  =IR_IMAGE_COLS;  //跟踪图像显示宽度
		pView->m_wndTrack.m_nImageHeight =IR_IMAGE_ROWS;  //跟踪图像显示高度
		pView->m_wndTrack.m_dibBMP.CreateDIB(pView->m_wndTrack.m_nImageWidth,pView->m_wndTrack.m_nImageHeight,0,8);
	}
	else{
		pView->m_wndTrack.m_dibBMP.DeleteObject();
		pView->m_wndTrack.m_nImageWidth  =RGB_IMAGE_COLS;  //跟踪图像显示宽度
		pView->m_wndTrack.m_nImageHeight =RGB_IMAGE_ROWS;  //跟踪图像显示高度
		pView->m_wndTrack.m_dibBMP.CreateDIB(pView->m_wndTrack.m_nImageWidth,pView->m_wndTrack.m_nImageHeight,0,24);
	}
#endif

	if (!trackCMD.bStartOrStop)//退出跟踪
	{   
#ifdef HARDWARE_OK	
		if(((CButton*)GetDlgItem(IDC_RADIO1))->GetCheck())
		{
			pView->m_pXferIR->Freeze();	
		}
		else
		{
			pView->m_pXferRGB->Freeze();	
		}
#endif
		//GetDlgItem(IDC_RADIO1)->EnableWindow(TRUE);
		//GetDlgItem(IDC_RADIO2)->EnableWindow(TRUE);
		pView->m_bRunning =FALSE;
		//SetDlgItemText(IDC_BTN_WORK,"开始跟踪");
	}
	else //开始跟踪
	{
#ifdef HARDWARE_OK	
		//CameraLink采集卡跟踪通道启动采集
		//UDP 协议控制
		if(trackCMD.bIrOrCCD == 1)
		{
			pView->m_pXferIR->Grab();	
		}
		else //if(trackCMD.bIrOrCCD == 2)
		{
			pView->m_pXferRGB->Grab();	
		}
#endif
		//GetDlgItem(IDC_RADIO1)->EnableWindow(FALSE);
		//GetDlgItem(IDC_RADIO2)->EnableWindow(FALSE);
		
		//SetDlgItemText(IDC_BTN_WORK,"停止跟踪");
		//pView->m_wndCtrlTab.SetActiveTab(0);
		//Sleep(500);
		
		pView->m_bExe=FALSE;
		pView->m_bRunning =TRUE;	
	}
}
//*/

void CIRTagetRecSysView::ButtonCMD()
{
	CIRTagetRecSysView* pView = (CIRTagetRecSysView*)(GetParent()->GetParent());
	CString strText; 
	GetDlgItemText(IDC_BTN_WORK,strText);
	pView->GetParent()->SetWindowText("红外目标跟踪识别系统 - CETC");

	if (!trackCMD.bStartOrStop)//退出跟踪
	{   
#ifdef HARDWARE_OK	
		if(((CButton*)GetDlgItem(IDC_RADIO1))->GetCheck())
		{
	
			pView->m_pXferIR->Freeze();	
		}
		else
		{
			pView->m_pXferRGB->Freeze();	
		}
#endif
		GetDlgItem(IDC_RADIO1)->EnableWindow(TRUE);
		GetDlgItem(IDC_RADIO2)->EnableWindow(TRUE);
		pView->m_bRunning =FALSE;
		SetDlgItemText(IDC_BTN_WORK,"开始跟踪");
	}
	else //开始跟踪
	{
#ifdef HARDWARE_OK	
		//CameraLink采集卡跟踪通道启动采集
		//if(trackCMD==0)
		//UDP 协议控制
		if(trackCMD.bIrOrCCD == 1)
		{
			pView->m_pXferIR->Grab();	
		}
		else //if(trackCMD.bIrOrCCD == 2)
		{
			pView->m_pXferRGB->Grab();	
		}
#endif
		GetDlgItem(IDC_RADIO1)->EnableWindow(FALSE);
		GetDlgItem(IDC_RADIO2)->EnableWindow(FALSE);
		
		SetDlgItemText(IDC_BTN_WORK,"停止跟踪");
		pView->m_wndCtrlTab.SetActiveTab(0);
		//Sleep(500);
		
		pView->m_bExe=FALSE;
		pView->m_bRunning =TRUE;	
	}
}
CIRTagetRecSysView::CIRTagetRecSysView()
{
	// TODO: 在此处添加构造代码
	m_fCapFrameTime=0.0;
	m_bRunning =FALSE;
	m_bExe = FALSE;
	m_pSapAcqIR  = NULL;
	m_pXferIR    = NULL;
	m_pBuffersIR = NULL;
	m_bIRChanLinkOk =false;


	m_pSapAcqRGB  = NULL;
	m_pXferRGB    = NULL;
	m_pBuffersRGB = NULL;
	m_bRGBChanLinkOk =false;
}

CIRTagetRecSysView::~CIRTagetRecSysView()
{
}

BOOL CIRTagetRecSysView::PreCreateWindow(CREATESTRUCT& cs)
{
	// TODO: 在此处通过修改
	//  CREATESTRUCT cs 来修改窗口类或样式
	//cs.style   &=~WS_BORDER;
	return CView::PreCreateWindow(cs);
}

// CIRTagetRecSysView 绘制

void CIRTagetRecSysView::OnDraw(CDC* /*pDC*/)
{
	
	// TODO: 在此处为本机数据添加绘制代码
}
// CIRTagetRecSysView 诊断

#ifdef _DEBUG
void CIRTagetRecSysView::AssertValid() const
{
	CView::AssertValid();
}

void CIRTagetRecSysView::Dump(CDumpContext& dc) const
{
	CView::Dump(dc);
}
#endif //_DEBUG

// CIRTagetRecSysView 消息处理程序
void CIRTagetRecSysView::OnSize(UINT nType, int cx, int cy)
{
	CView::OnSize(nType, cx, cy);

	// TODO: Add your message handler code here
	SetWndSize();

}
void CIRTagetRecSysView::SetWndSize()
{
	CRect rect;
	GetClientRect(rect);
	if (m_wndTrack.m_hWnd)
	{		
		int nTrackWndHeight =rect.bottom-4;
		int nTrackWndWidth = int(nTrackWndHeight*1.333);
		m_wndTrack.SetWindowPos(NULL,rect.left,2,nTrackWndWidth,nTrackWndHeight,SWP_NOACTIVATE | SWP_NOZORDER);	
		m_wndCtrlTab.SetWindowPos(NULL,rect.left+nTrackWndWidth+2,2,rect.right-nTrackWndWidth-4,nTrackWndHeight,SWP_NOACTIVATE | SWP_NOZORDER);	

	}
}
BOOL CIRTagetRecSysView::OnEraseBkgnd(CDC* pDC)
{
	// TODO: Add your message handler code here and/or call default
	CRect rect;
	GetClientRect(rect);
	pDC->FillSolidRect(0,0,rect.right,rect.bottom,RGB(0,110,0));

	CString strText;
	pDC->SetBkMode(TRANSPARENT);	
	pDC->SetTextColor(RGB(168,168,168));

	return TRUE;// CView::OnEraseBkgnd(pDC);
}
int CIRTagetRecSysView::OnCreate(LPCREATESTRUCT lpCreateStruct)
{
	if (CView::OnCreate(lpCreateStruct) == -1)
		return -1;

	// TODO:  Add your specialized creation code here	
	//创建图像内存缓冲区，内存分配按照小视场(120000)的开，一次开到位
	m_pbyIr       =new BYTE[IR_IMAGE_COLS*IR_IMAGE_ROWS];    
	m_pbyRGB      =new BYTE[RGB_IMAGE_COLS*RGB_IMAGE_ROWS*3];
	for(int i=0;i<2;i++)//只分配前2个内存区
	{
		m_pbyIrBuffs[i]  = new BYTE[IR_IMAGE_COLS*IR_IMAGE_ROWS];   
		if (m_pbyIrBuffs[i]==NULL)
		{
			MessageBox("内存分配失败！",_T("提示"),MB_OK|MB_ICONWARNING);
		}

		m_pbyRGBBuffs[i]  = new BYTE[RGB_IMAGE_COLS*RGB_IMAGE_ROWS*3];  
		if (m_pbyRGBBuffs[i]==NULL)
		{
			MessageBox("内存分配失败！",_T("提示"),MB_OK|MB_ICONWARNING);
		}
	}
	for(int i=2;i<MAXBUFFNUM;i++)//后面的缓冲区不再分配内存，与前2个共用
	{
		m_pbyIrBuffs[i]  = m_pbyIrBuffs[i%2];
		m_pbyRGBBuffs[i]  =m_pbyRGBBuffs[i%2];
	}
	m_nCurIRCaptureNo=0;
	m_nCurRGBCaptureNo=0;

	m_wndTrack.Create(NULL, _T(""), WS_CHILD | WS_VISIBLE, CRect(0,0,0,0), this, 3);
	m_wndCtrlTab.Create(CMFCTabCtrl::STYLE_3D_ONENOTE,CRect(0,0,0,0), this, 0,CMFCTabCtrl::LOCATION_TOP, FALSE);
	m_wndCtrlTab.EnableAutoColor (FALSE);
	m_wndCtrlTab.EnableTabSwap(FALSE);//不允许移动Tab之间位置
	m_dlgTrackCtrl.Create(IDD_TRACKCTRL,&m_wndCtrlTab);
	m_wndCtrlTab.AddTab(&m_dlgTrackCtrl,"跟踪控制",0, FALSE);

	return 0;
}

//追踪消息
/*
void CIRTagetRecSysView::pyfunc(void* imgBuff)
{
	//可见光
	//if(m_dlgTrackCtrl.m_byImageType==1)
	if(trackCMD.bIrOrCCD == 2)
	{
		npy_intp Dim[3] = {IMG_RGB_HEIGHT, IMG_RGB_WIDTH, 3};
		drone_det->pMat = PyArray_SimpleNewFromData(3, Dim, NPY_UBYTE, imgBuff);
		PyTuple_SetItem(drone_det->pArgs,0, drone_det->pMat);   
		PyObject_CallObject(drone_det->pFunc, drone_det->pArgs);
	}
	//红外图像
	else
	{
		npy_intp Dim[2] = {IMG_IR_HEIGHT, IMG_IR_WIDTH};
		drone_det->pMat = PyArray_SimpleNewFromData(2, Dim, NPY_UBYTE, imgBuff);
		PyTuple_SetItem(drone_det->pArgs,0, drone_det->pMat);   
		PyObject_CallObject(drone_det->pFunc, drone_det->pArgs);
	}

}
*/
/*
LRESULT  CIRTagetRecSysView::PyMeaasgeFun(WPARAM wParam, LPARAM lParam)
{
	m_bExe =TRUE;
	//....
	//pyfunc((void*)IMGData);
	//ButtonCMD();
	m_bExe =FALSE;
	return 0;
}
*/
LRESULT  CIRTagetRecSysView::TrackCMDMessage(WPARAM wParam, LPARAM lParam)
{
	ButtonCMD();
	//pyfunc((void*)IMGData);
	return 0;
}//*/

void CIRTagetRecSysView::OnInitialUpdate()
{

	CView::OnInitialUpdate();
	/*
	Py_SetPythonHome(L"C:\\Users\\aaa\\Anaconda3");
	Py_Initialize();
	//PyEval_InitThreads();
	import_array();
	drone_det->pModule = PyImport_ImportModule("detective2_vs_test2");
	drone_det->pDict = PyModule_GetDict(drone_det->pModule);
	drone_det->pFunc = PyDict_GetItemString(drone_det->pDict, "global_init");
	PyObject_CallObject(drone_det->pFunc, NULL);
	drone_det->pFunc = PyDict_GetItemString(drone_det->pDict, "recvimg1");
	drone_det->pArgs = PyTuple_New(1);
	*/
	// TODO: Add your specialized code here and/or call the base class
#ifdef HARDWARE_OK
	//初始化热像仪Camera采集卡,注意后面几段顺序不能反
	CString strConfigSearch;
	if(!InitSapCom(IR_CONFIG_FILENAME,RGB_CONFIG_FILENAME))
	{
		MessageBox("连接热像仪采集卡失败！",_T("提示"),MB_OK|MB_ICONWARNING);
	}
	//设置图像采集回调函数
	m_pXferIR  = new SapAcqToBuf(m_pSapAcqIR, m_pBuffersIR, IRCapCallback, this);//IR采集接口回调; 
	m_pXferRGB  = new SapAcqToBuf(m_pSapAcqRGB, m_pBuffersRGB, RGBCapCallback, this);//RGB采集接口回调; 
	if (!CreateSapObjects())//创建CL采集卡相关资源
	{  
		MessageBox("设置热像仪采集卡失败！",_T("提示"),MB_OK|MB_ICONWARNING);
		return; 
	}

	//end

#endif
	//创建所有相关线程
#ifndef HARDWARE_OK
	//创建仿真采集线程
	::CreateThread (NULL, 0, IRCapThread, this, NULL, NULL);
#endif
	//创建接收数据线程
	::CreateThread (NULL, 0, RECVMessage, this, NULL, NULL);
	//创建主处理线程

	::CreateThread (NULL, 0, IRProcessThread, this, NULL, NULL);

	//////
// 	trackCMD.bIrOrCCD = 1;
// 	trackCMD.bStartOrStop = 1;
}
//初始化CL采集卡,加载采集卡配置
BOOL CIRTagetRecSysView::InitSapCom(CString strIRConfig,CString strRGBConfig)
{
	DestroySapObjects();
	
	//获取当前.exe文件的主目录
	CHAR   FullPath[MAX_PATH];
	CString strMainDir;
	GetModuleFileName(NULL,FullPath,MAX_PATH); 
	strMainDir = (CString)FullPath; 
	int position = strMainDir.ReverseFind('\\'); 
	strMainDir  = strMainDir.Left(position); 

	//IR模式,采用端口2图像
	strIRConfig= strMainDir+strIRConfig;//配置文件2

	SapLocation loc("Xcelera-CL_PX4_1",1);//端口2
	SapAcquisition IRAcq;
	IRAcq.SetLocation(loc);
	IRAcq.SetConfigFile(strIRConfig);
	m_pSapAcqIR  = new SapAcquisition(IRAcq);
	m_pBuffersIR = new SapBufferWithTrash(2,m_pSapAcqIR,SapBuffer::TypeScatterGather);//根据配置信息分配IR采集通道buffer;

	//RGB模式,采用端口1图像
	strRGBConfig= strMainDir+strRGBConfig;//配置文件1
	SapLocation loc2("Xcelera-CL_PX4_1",0);//端口1
	SapAcquisition RGBAcq;
	RGBAcq.SetLocation(loc2);
	RGBAcq.SetConfigFile(strRGBConfig);
	m_pSapAcqRGB  = new SapAcquisition(RGBAcq);
	m_pBuffersRGB = new SapBufferWithTrash(2,m_pSapAcqRGB,SapBuffer::TypeScatterGather);//根据配置信息分配RGB采集通道buffer;

	//GetSignalStatus();   //得到当前信号状态;
	return TRUE;
}
void CIRTagetRecSysView::GetSignalStatus()
{
	SapAcquisition::SignalStatus signalStatus;
	if (m_pSapAcqIR)
	{
		if(m_pSapAcqIR->IsSignalStatusAvailable())
		{
			m_pSapAcqIR->GetSignalStatus(&signalStatus, SignalCallbackIR, this);//设置信号1状态回调;
		}	
	}
	if (m_pSapAcqRGB)
	{
		if(m_pSapAcqRGB->IsSignalStatusAvailable())
		{
			m_pSapAcqRGB->GetSignalStatus(&signalStatus, SignalCallbackRGB, this);//设置信号1状态回调;
		}	
	}
}
void CIRTagetRecSysView::SignalCallbackIR(SapAcqCallbackInfo *pInfo)
{
	CIRTagetRecSysView *pView = (CIRTagetRecSysView *) pInfo->GetContext();
	SapAcquisition::SignalStatus signalStatus = pInfo->GetSignalStatus();
	//pView->m_bIRChanLinkOk =(signalStatus != SapAcquisition::SignalNone);
}
void CIRTagetRecSysView::SignalCallbackRGB(SapAcqCallbackInfo *pInfo)
{
	CIRTagetRecSysView *pView = (CIRTagetRecSysView *) pInfo->GetContext();
	SapAcquisition::SignalStatus signalStatus = pInfo->GetSignalStatus();
	//pView->m_bRGBChanLinkOk =(signalStatus != SapAcquisition::SignalNone);
}

BOOL CIRTagetRecSysView::CreateSapObjects()
{
	CWaitCursor wait;

	//采集卡接口1
	// Create acquisition object 
	if (m_pSapAcqIR && !*m_pSapAcqIR && !m_pSapAcqIR->Create())
	{
		DestroySapObjects();//释放资源
		return FALSE;
	}
	// Create buffer object
	if (m_pBuffersIR&& !*m_pBuffersIR && !m_pBuffersIR->Create())
	{
		DestroySapObjects();
		return FALSE;
	}
	// Create transfer object
	if (m_pXferIR && !*m_pXferIR && !m_pXferIR->Create())
	{
		DestroySapObjects();
		return FALSE;
	}
		//采集卡接口2
	// Create acquisition object 
	if (m_pSapAcqRGB && !*m_pSapAcqRGB && !m_pSapAcqRGB->Create())
	{
		DestroySapObjects();//释放资源
		return FALSE;
	}
	// Create buffer object
	if (m_pBuffersRGB&& !*m_pBuffersRGB && !m_pBuffersRGB->Create())
	{
		DestroySapObjects();
		return FALSE;
	}
	// Create transfer object
	if (m_pXferRGB && !*m_pXferRGB && !m_pXferRGB->Create())
	{
		DestroySapObjects();
		return FALSE;
	}
	return TRUE;
}
BOOL CIRTagetRecSysView::DestroySapObjects()  
{
	if (m_pXferIR && *m_pXferIR) 
	{
		m_pXferIR->Destroy();
		m_pXferIR=NULL;
	}
	if (m_pBuffersIR && *m_pBuffersIR) 
	{
		m_pBuffersIR->Destroy();
		m_pBuffersIR=NULL;
	}
	if (m_pSapAcqIR && *m_pSapAcqIR)
	{
		m_pSapAcqIR->Destroy();
		m_pSapAcqIR=NULL;
	}

	if (m_pXferRGB && *m_pXferRGB) 
	{
		m_pXferRGB->Destroy();
		m_pXferRGB=NULL;
	}
	if (m_pBuffersRGB && *m_pBuffersRGB) 
	{
		m_pBuffersRGB->Destroy();
		m_pBuffersRGB=NULL;
	}
	if (m_pSapAcqRGB && *m_pSapAcqRGB)
	{
		m_pSapAcqRGB->Destroy();
		m_pSapAcqRGB=NULL;
	}
	return TRUE;
}
//IR采集通道回调函数
void CIRTagetRecSysView::IRCapCallback(SapXferCallbackInfo *pInfo)
{	
	CIRTagetRecSysView *pView= (CIRTagetRecSysView*) pInfo->GetContext();
	//static int frame_num = 0;
	//回调抓取过慢
	if (pInfo->IsTrash())
	{
		CString str;
		str.Format("Number of missed frames : %d", pInfo->GetEventCount());
		//AfxMessageBox("XferCallback1");
	}
	else
	{
		//抓取到一帧IR图像，往下走
		BYTE* pbyIRImgBuffer=NULL;
		pView->m_pBuffersIR->GetAddress((void**)&pbyIRImgBuffer);//获得采集接口1的图像缓冲区指针
		
		//拷贝
		memcpy(pView->m_pbyIrBuffs[pView->m_nCurIRCaptureNo],pbyIRImgBuffer, IR_IMAGE_COLS*IR_IMAGE_ROWS);

		static int startTime = GetTickCount();
		int endTime = GetTickCount();
		pView->m_fCapFrameTime = (endTime-startTime);	
		startTime =endTime;
		if (pView->m_nCurIRCaptureNo==(MAXBUFFNUM-1))
		{
		    pView->m_nCurIRCaptureNo=0;
	   	}
		else
		{
			pView->m_nCurIRCaptureNo++;
		}	
		pView->m_nCurRGBCaptureNo=0;
	}
}
//RGB采集通道回调函数
void CIRTagetRecSysView::RGBCapCallback(SapXferCallbackInfo *pInfo)
{	
	CIRTagetRecSysView *pView= (CIRTagetRecSysView*) pInfo->GetContext();
	//static int frame_num = 0;
	if (pInfo->IsTrash())
	{
		CString str;
		str.Format("Number of missed frames : %d", pInfo->GetEventCount());
		//AfxMessageBox("XferCallback1");
	}
	else
	{
		//抓取到一帧yuv图像，往下走
		BYTE* pbyYUV422ImgBuffer=NULL;
		pView->m_pBuffersRGB->GetAddress((void**)&pbyYUV422ImgBuffer);//获得采集接口2的图像缓冲区指针
		
		BYTE* pbyRGB=pView->m_pbyRGBBuffs[pView->m_nCurRGBCaptureNo];
		for(int i=0;i<RGB_IMAGE_COLS*RGB_IMAGE_ROWS/2;i++)
		{
			BYTE  Y0 = pbyYUV422ImgBuffer[4*i];
			BYTE  Cr =  pbyYUV422ImgBuffer[4*i+1];
			BYTE  Y1 = pbyYUV422ImgBuffer[4*i+2];
			BYTE  Cb =  pbyYUV422ImgBuffer[4*i+3];		
			
			//pixel1
			double r=1.164*(Y0-16) +1.596*(Cr-128);//R;
			pbyRGB[3*(2*i)+2]=r;
			if(r>=255.0)
			{
				pbyRGB[3*(2*i)+2]=255;
			}
			else if(r<=0.0)
			{
				pbyRGB[3*(2*i)+2]=0;
			}
			double g=1.164*(Y0-16) -0.813*(Cr-128)-0.392*(Cb-128);//G
			pbyRGB[3*(2*i)+1]=g;
		    if(g>=255.0)
			{
				pbyRGB[3*(2*i)+1]=255;
			}
			else if(g<=0.0)
			{
				pbyRGB[3*(2*i)+1]=0;
			}
			double b= 1.164*(Y0-16)+2.017*(Cb-128);//B
			pbyRGB[3*(2*i)]=b;
		    if(b>=255.0)
			{
				pbyRGB[3*(2*i)]=255;
			}
			else if(b<=0.0)
			{
				pbyRGB[3*(2*i)]=0;
			}

			//pixel2
			r=1.164*(Y1-16) +1.596*(Cr-128);//R;
			pbyRGB[3*(2*i+1)+2]=r;
			if(r>=255.0)
			{
				pbyRGB[3*(2*i+1)+2]=255;
			}
			else if(r<=0.0)
			{
				pbyRGB[3*(2*i+1)+2]=0;
			}
			g=1.164*(Y1-16) -0.813*(Cr-128)-0.392*(Cb-128);//G
			pbyRGB[3*(2*i+1)+1]=g;
		    if(g>=255.0)
			{
				pbyRGB[3*(2*i+1)+1]=255;
			}
			else if(g<=0.0)
			{
				pbyRGB[3*(2*i+1)+1]=0;
			}
			b= 1.164*(Y1-16)+2.017*(Cb-128);//B
			pbyRGB[3*(2*i+1)]=b;
		    if(b>=255.0)
			{
				pbyRGB[3*(2*i+1)]=255;
			}
			else if(b<=0.0)
			{
				pbyRGB[3*(2*i+1)]=0;
			}
		}
		static int startTime = GetTickCount();
		int endTime = GetTickCount();
		pView->m_fCapFrameTime = (endTime-startTime);	
		startTime =endTime;
		if (pView->m_nCurRGBCaptureNo==(MAXBUFFNUM-1))
		{
		    pView->m_nCurRGBCaptureNo=0;
	   	}
		else
		{
			pView->m_nCurRGBCaptureNo++;
		}	
		pView->m_nCurIRCaptureNo=0;
	}
}
//备注，下面这个线程仅在无硬件时仿真调用
#ifndef HARDWARE_OK
ULONG CIRTagetRecSysView::IRCapThread(LPVOID lpParam) 
{
	CIRTagetRecSysView* pView = (CIRTagetRecSysView*)lpParam;
	UINT nStartTime = GetTickCount();
	UINT nCurTime   = GetTickCount();
	UINT nDataPackNum=0;
	int nImageW=0;
	int nImageH=0;
	BYTE* pbyIRImgBuffer=new BYTE[TRACK_PACKET_COLS*IR_IMAGE_ROWS];  //跟踪8bit数据
	while(TRUE)
	{
		nStartTime = GetTickCount();
		if (pView->m_bRunning==false&&pView->m_bRunning==false)
		{
			Sleep(100);
			continue;
		}
			for(int i=0;i<IR_IMAGE_ROWS;i++)
			{
				for(int j=0;j<TRACK_PACKET_COLS;j++)
			    {
				    pbyIRImgBuffer[j*IR_IMAGE_ROWS+i] =rand()%256;
		     	}
			}
		    //图像垂直镜像（出来的是倒立的）
		    for(int i=0;i<IR_IMAGE_ROWS;i++)
		    {
			    memcpy(pView->m_pbyIrBuffs[pView->m_nCurIRCaptureNo]+(IR_IMAGE_ROWS-1-i)*IR_IMAGE_COLS,pbyIRImgBuffer+i*TRACK_PACKET_COLS, IR_IMAGE_COLS);
		    }
		    static int startTime = GetTickCount();
	    	int endTime = GetTickCount();
	    	pView->m_fCapFrameTime = (endTime-startTime);	
			startTime =endTime;
			if (pView->m_nCurIRCaptureNo==(MAXBUFFNUM-1))
			{
				pView->m_nCurIRCaptureNo=0;
			}
			else
			{
				pView->m_nCurIRCaptureNo++;
			}	
			Sleep(100);
		 }
	return 0;
}
#endif

ULONG CIRTagetRecSysView::RECVMessage(LPVOID lpParam)
{
	CIRTagetRecSysView* pView = (CIRTagetRecSysView*)lpParam;
	//创建套接字信息结构体
	WSADATA wsadata;
	//设置window socket版本号为2.2
	WORD sockVersion = MAKEWORD(2, 2);
	//启动构建windos socket 
	if (WSAStartup(sockVersion, &wsadata) != 0)
	{
		printf("WSAStartup failed \n");
		return 0;
	}
	SOCKET sClient = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
	//创建对方地址(服务器)
	sockaddr_in serverAddr;
	//创建本地地址（客户端）
	sockaddr_in clientAddr;
	//设置本地端口地址
	clientAddr.sin_family = AF_INET;
	//现场注意端口设置
	clientAddr.sin_port = htons(9921);
	clientAddr.sin_addr.S_un.S_addr = INADDR_ANY;
	if (SOCKET_ERROR == sClient)
	{
		printf("socket failed !\n");
		return 0;
	}
	//绑定本机
	if (bind(sClient, (SOCKADDR *)&clientAddr, sizeof(clientAddr)))
	{
		//system("pause");
		return 0;
	}
	//获取双方地址长度
	int slen = sizeof(serverAddr);
	int clen = sizeof(clientAddr);
	//设置缓冲区大小并对缓冲区清零,一共开辟1024个字节
	char buffer[1024];
	memset(buffer, 0, sizeof(buffer));
	int iRcv = 0;
	//全局结构体
	int N=0;

	char ipAddr[] = "127.0.0.1";
	int Port = 9999;
	int endSuccess;

	while(TRUE)
	{
		Sleep(10);
		/*
		if (pView->m_bRunning==FALSE)
		{

			Sleep(100);
			continue;
		}//*/
		//插入接收信号
		TRACE("WAITING");
		//接收数据
		iRcv = recvfrom(sClient, buffer, sizeof(buffer), 0, (SOCKADDR*)&serverAddr, &slen);
		memcpy(&trackCMD, buffer, 28);
		TRACE("irccd=%d---start=%d\n", trackCMD.bIrOrCCD, trackCMD.bStartOrStop);
		if (iRcv == SOCKET_ERROR)
		{
			printf("recvfrom failed:%d\n", WSAGetLastError());
		}
		else
		{	
			printf("SenderIP  :%s\n", inet_ntoa(serverAddr.sin_addr));
		}
		//CIRTagetRecSysView::ButtonCMD();
		//::SendMessage(pView->GetSafeHwnd(),OPENCVMESSAGE,NULL,NULL);

		//接收到python发送的信号之后进行处理
		//::PostMessageA(pView->GetSafeHwnd(),WM_COMMAND,MAKEWPARAM(IDC_BTN_WORK,OnBnClickedBtnWork),NULL);
		pView->m_dlgTrackCtrl.PerformClick();
		
		//if(!trackCMD.bStartOrStop){
	    //	
		//	endSuccess = udpEndSignal(ipAddr, Port);
		//	TRACE("endsig:%d\n",endSuccess);
		//}
		
	}
	
	closesocket(sClient);
	WSACleanup();
	//system("pause");
	return 0;
}

SOCKET createSocket() { //返回一个socket, 其类型为ipv4, UDP, 并没有将其绑定到特定端口
	//声明调用不同的Winsock版本。
	WORD version = MAKEWORD(2, 2);
	//一种数据结构。这个结构被用来存储被WSAStartup函数调用后返回的Windows Sockets数据。
	WSADATA wsadata;
	/*WSAStartup必须是应用程序或DLL调用的第一个Windows Sockets函数。
	它允许应用程序或DLL指明Windows Sockets API的版本号及获得特定Windows Sockets实现的细节。
	应用程序或DLL只能在一次成功的WSAStartup()调用之后才能调用进一步的Windows Sockets API函数。
    */
	if (WSAStartup(version, &wsadata))
	{
		cout << "WSAStartup failed " << endl;
		cout << "2s后控制台将会关闭！" << endl;
		Sleep(2000);
		exit(0);
	}
	//判断版本
	if (LOBYTE(wsadata.wVersion) != 2 || HIBYTE(wsadata.wVersion) != 2)
	{
		cout << "wVersion not 2.2" << endl;
		cout << "2s后控制台将会关闭！" << endl;
		Sleep(2000);
		exit(0);
	}
	//创建客户端UDP套接字
	SOCKET client;
	client = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
	if (SOCKET_ERROR == client)
	{
		cout << "socket failed" << endl;
		cout << "2s后控制台将会关闭！" << endl;
		Sleep(2000);
		exit(0);
	}
	else {
		return client;
	}
}


int udpEndSignal(char ipAddr[] , int Port)
{

	SOCKET client = createSocket();
	sockaddr_in caddr; //define an addr to use in recvfrom(), which is not specified
	sockaddr_in saddr;
	int slen = sizeof(saddr);
	int clen = sizeof(caddr);
	saddr.sin_family = AF_INET;
	saddr.sin_port = htons(Port);
	saddr.sin_addr.S_un.S_addr = inet_addr(ipAddr);
	char end[] = "STOP";
	int iSend = 0;
	iSend = sendto(client, end, strlen(end), 0, (SOCKADDR*)&saddr, slen);

	closesocket(client);
	WSACleanup();

	return iSend;

}

int udpSend(char ipAddr[] , int Port, BYTE * IMG, int length, int frameSize)
{
	SOCKET client = createSocket();
	
	sockaddr_in caddr; //define an addr to use in recvfrom(), which is not specified
	sockaddr_in saddr;

	//接收端地址长度
	int slen = sizeof(saddr);
	int clen = sizeof(caddr);
	//目的地址
	//设置服务器地址
	saddr.sin_family = AF_INET;
    saddr.sin_port = htons(Port);
	saddr.sin_addr.S_un.S_addr = inet_addr(ipAddr);
	
	/* //uncomment to bind client socket, which is unnecessary
	caddr.sin_family = AF_INET;
	caddr.sin_port = htons(Port-5);
	caddr.sin_addr.S_un.S_addr = inet_addr(ipAddr);
	/*
	int ret = bind(client, (SOCKADDR *)&caddr, clen);
	if (ret == SOCKET_ERROR) {  
			cout << "client bind failed！" << endl;   
			cout << "2s后退出控制台！" << endl;
			closesocket(client);
			WSACleanup();
			Sleep(2000);
			return -4;
	}
	//*/
	char data[100] = { 0 };                    //接受一些短字节的数据缓冲区
	char begin[] = "I BEGIN";       //发送图片前的确认信息
	//char end[] = "STOP";         //完成图片发送的通知信息

	int iSend = 0;                             //发送函数的状态
	int iRecv = 0;
		
	//发送图片前先和服务器打个招呼，欲准备状态，判断信息发送是否成功，若不成功，则服务器处于关闭状态
	iSend = sendto(client, begin, strlen(begin), 0, (SOCKADDR*)&saddr, slen);

	if (iSend == SOCKET_ERROR) {  
			cout << "服务器处于关闭状态，请稍后重试！" << endl;   
			cout << "20s后退出控制台！" << endl;
			closesocket(client);
			WSACleanup();
			Sleep(20000);
			return -4;
	}
	iRecv = recvfrom(client, data, sizeof(data), 0, (SOCKADDR*)&caddr, &clen);
	//cout << "Client: " << begin << endl;

	memset(data, 0, sizeof(data));

	//cout << "Img length: " << length << endl;

	iSend = sendto(client, (char *)&length, sizeof(int), 0, (SOCKADDR*)&saddr, slen); //首先发送图片大小(单位byte)给接收端
	if (iSend == SOCKET_ERROR) { 
			cout << "文件长度信息发送失败！" << endl;   
			cout << "10s后退出控制台！" << endl;
			closesocket(client);
			WSACleanup();
			Sleep(10000);
			return -4;
	}
	iRecv = recvfrom(client, data, sizeof(data), 0, (SOCKADDR*)&caddr, &clen);
	//cout << "····BEGIN SEND PICTURE····" << endl;
	//int i = 0;

	iSend = sendto(client, (char*)IMG, length, 0, (SOCKADDR*)&saddr, slen);
	//* 是否将图片分段发送(每段frameSize Bytes)
	while (length > 0) {
			//cout << i << endl;

			//发送图片的一部分，发送成功，则图片总长度减去当前发送的图片片断长度
			if (length >= frameSize){
				iSend = sendto(client, (char*)IMG, frameSize, 0, (SOCKADDR*)&saddr, slen);
				IMG += frameSize;
			}
			else{
				iSend = sendto(client, (char*)IMG, length, 0, (SOCKADDR*)&saddr, slen);
				IMG += length;
			}

			if (iSend == SOCKET_ERROR) {
				cout << "发送图片出错" << endl;
				cout << "2s后退出控制台！" << endl;
				closesocket(client);
				WSACleanup();
				Sleep(2000);
				return -8;
			}
            iRecv = recvfrom(client, data, sizeof(data), 0, (SOCKADDR*)&caddr, &clen);
			length -= frameSize;
			//i++;
	}
	//iRecv = recvfrom(client, data, sizeof(data), 0, (SOCKADDR*)&caddr, &clen);
    //*/
	closesocket(client);
	WSACleanup();
	//cout<< "SEND FINISH" << endl;
	return 0;
}

ULONG CIRTagetRecSysView::IRProcessThread(LPVOID lpParam) 
{
	CIRTagetRecSysView* pView = (CIRTagetRecSysView*)lpParam;
	UINT nStartTime = GetTickCount();
	UINT nCurTime   = GetTickCount();
	UINT nDataPackNum=0;
	int nImageW=0;
	int nImageH=0;
	CString strText;
	UINT nCurIRCaptureNo=MAXBUFFNUM-1;
	UINT nCurRGBCaptureNo=MAXBUFFNUM-1;
	char ipAddr[] = "127.0.0.1";
	int Port = 9999;
	int frameSize = 8192;
	int sendSucessFlag;
	int endSuccess=-100;

	while(TRUE)
	{
		Sleep(0);
		nStartTime   = GetTickCount();
		if (pView->m_bRunning==FALSE)
		{
			nCurIRCaptureNo =MAXBUFFNUM-1;
			nCurRGBCaptureNo=MAXBUFFNUM-1;
			
			if(endSuccess==-100){
				endSuccess = udpEndSignal(ipAddr, Port);
			    //TRACE("endsig:%d\n",endSuccess);
			}
			Sleep(100);
			continue;
		}
		//+++++++++++++++++++++++++++++++++++++++++++跟踪模式图像处理+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
		pView->m_bRunning = TRUE;
		if (pView->m_bRunning==TRUE)
		{	endSuccess=-100;		
			//if(pView->m_dlgTrackCtrl.m_byImageType==0)//红外
			if(trackCMD.bIrOrCCD == 1)
			{
				int nCurIRCaptureNoNew=MAXBUFFNUM-1;
				if (pView->m_nCurIRCaptureNo>0)
				{
					nCurIRCaptureNoNew =pView->m_nCurIRCaptureNo-1;
				}
				//判断当前帧是否已经处理过,未处理过则处理
				if (nCurIRCaptureNo!=nCurIRCaptureNoNew)
				{
					nCurIRCaptureNo=nCurIRCaptureNoNew;
					//拷贝8bit数据至跟踪区处理区		
					
					memcpy(pView->m_pbyIr,pView->m_pbyIrBuffs[nCurIRCaptureNo], IR_IMAGE_COLS*IR_IMAGE_ROWS);	
					BYTE* pImage =(BYTE*)pView->m_wndTrack.m_dibBMP.GetBits();
					BYTE* pImagSOURCE=pView->m_pbyIrBuffs[nCurIRCaptureNo];
					for(int i=0;i<IR_IMAGE_COLS;i++)
					{
						for(int j=0;j<IR_IMAGE_ROWS;j++)
						{
							pImage[(IR_IMAGE_ROWS-1-j)*IR_IMAGE_COLS+i] = pImagSOURCE[j*IR_IMAGE_COLS+i];	
						}
					}		
					//+++++++++++++++++++++跟踪算法处理在此添加++++++++++++++++++++++++++++++++++++++++++++++++++	
					//。。。。。待处理的内存区为m_pbyIr

					if(pView->m_bExe == false)
					{
						//每次只处理最新帧
						IMGData = pView->m_pbyIr;
						if(trackCMD.bStartOrStop)
						{
                            sendSucessFlag = udpSend(ipAddr, Port, IMGData, 640*512, frameSize);
							//::SendMessage(pView->GetSafeHwnd(),OPENCVMESSAGE,NULL,NULL);
						}
							
					}
					//drone_det->CvtPyObject((void*)pView->m_pbyIr);
					//+++++++++++++++++++++跟踪图显示++++++++++++++++++++++++++++++++++++++++++++++++	

// 					pView->m_wndTrack.m_bRunning =pView->m_bRunning;
// 					pView->m_wndTrack.m_fCapFrameTime =pView->m_fCapFrameTime;
// 					pView->m_wndTrack.DrawWnd();
				}
			}
			else
			{
				int nCurRGBCaptureNoNew=MAXBUFFNUM-1;
				if (pView->m_nCurRGBCaptureNo>0)
				{
					nCurRGBCaptureNoNew =pView->m_nCurRGBCaptureNo-1;
				}
				//判断当前帧是否已经处理过,未处理过则处理
				if (nCurRGBCaptureNo!=nCurRGBCaptureNoNew)
				{
					nCurRGBCaptureNo=nCurRGBCaptureNoNew;
					//拷贝8bit数据至跟踪区处理区						
					memcpy(pView->m_pbyRGB,pView->m_pbyRGBBuffs[nCurRGBCaptureNo], RGB_IMAGE_COLS*RGB_IMAGE_ROWS*3);	
					//+++++++++++++++++++++跟踪算法处理在此添加++++++++++++++++++++++++++++++++++++++++++++++++++	
					//。。。。。待处理的内存区为m_pbyIr

					if(pView->m_bExe == false)
					{
						//每次只处理最新帧
						IMGData = pView->m_pbyRGB;
						
						cv::Mat rgbImage(cv::Size(1920,1080),CV_8UC3,IMGData);
						//TRACE("GET ORG MAT");
						cv::Mat ResImg(cv::Size(640,384),CV_8UC3);
						//TRACE("GET RES MAT");
						cv::resize(rgbImage,ResImg, cv::Size(640,384),CV_INTER_CUBIC);
						//TRACE("RESIZE");
						BYTE* rgbImageDate = ResImg.data;
						//TRACE("PASS PTR");
						//cv::imwrite("wodeshixiong.jpg", ResImg);
						if(trackCMD.bStartOrStop){
							sendSucessFlag = udpSend(ipAddr, Port, rgbImageDate, 640*384*3, frameSize);
						}
							//::SendMessage(pView->GetSafeHwnd(),OPENCVMESSAGE,NULL,NULL);
					}
					//drone_det->CvtPyObject((void*)pView->m_pbyIr);

					//+++++++++++++++++++++彩色图显示++++++++++++++++++++++++++++++++++++++++++++++++	
					/*
					BYTE* pImage =(BYTE*)pView->m_wndTrack.m_dibBMP.GetBits();
					for(int i=0;i<RGB_IMAGE_COLS;i++)
					{
						for(int j=0;j<RGB_IMAGE_ROWS;j++)
						{
							pImage[3*((RGB_IMAGE_ROWS-1-j)*RGB_IMAGE_COLS+i)] = pView->m_pbyRGB[3*(j*RGB_IMAGE_COLS+i)];
							pImage[3*((RGB_IMAGE_ROWS-1-j)*RGB_IMAGE_COLS+i)+1] = pView->m_pbyRGB[3*(j*RGB_IMAGE_COLS+i)+1];
							pImage[3*((RGB_IMAGE_ROWS-1-j)*RGB_IMAGE_COLS+i)+2] = pView->m_pbyRGB[3*(j*RGB_IMAGE_COLS+i)+2];
							
						}
					}
					pView->m_wndTrack.m_bRunning =pView->m_bRunning;
					pView->m_wndTrack.m_fCapFrameTime =pView->m_fCapFrameTime;
					pView->m_wndTrack.DrawWnd();
					*/
				}
			}
		}
	}
	return 0;
}
