
#include "stdafx.h"
#include "DIB.h"

IMPLEMENT_DYNAMIC(CDIB, CGdiObject);
//////////////////////////////////////////////////////////////////////
//
// CDIB
//
// 设备无关位图类(版本1.0)
//
// 完成功能:
//
// 设备无关位图的创建,显示,读入,保存
//
//////////////////////////////////////////////////////////////////////

//构造函数
CDIB::CDIB() : CGdiObject()
{
}

//析购函数
CDIB::~CDIB()
{
	DeleteObject();
}


//////////////////////////////////////////////////////////////////
// 
// CreateDIB(int cx, int cy, UINT ibitcount, const void* lpBits) 
// 
// 完成功能:
//     创建DIB位图
//
// 输入参数:
//	   图像宽度 cx
//     图像高度 cy
//     图像位数 ibitcount 
//     图像数据 lpBits 
//
// 返回参数:
//	   是否成功
//
//////////////////////////////////////////////////////////////////

//创建DIB位图
BOOL CDIB::CreateDIB(int cx, int cy, int Palette,UINT ibitcount, const void* lpBits) 
{   //声明ibitcount仅能为1，4，8，16，24，32（即：位图信息头的biBitCount)
	ASSERT((ibitcount == 1) || (ibitcount == 4) || 
			(ibitcount == 8) || (ibitcount == 16) 
			|| (ibitcount == 24) ||(ibitcount == 32))	;

	// Create a BITMAPINFOHEADER structure to describe the DIB
	//创建信息头
    int iSize = sizeof(BITMAPINFOHEADER) + 256*sizeof(RGBQUAD);
	BITMAPINFO* pBMI;//位图信息头的指针
	BYTE *pByte;//指向位图像信息的指针
	switch(ibitcount)
	{
		case 8://256色
		case 1://单色
		case 4://16色
		case 16://16位色
		case 24://24位色
		case 32://32位色
			break;
		default:
			break;
	}

	pByte = new BYTE[iSize];
	pBMI = (BITMAPINFO*) pByte;
    memset(pBMI, 0, iSize);
	pBMI->bmiHeader.biSize        = sizeof(BITMAPINFOHEADER);
	pBMI->bmiHeader.biWidth       = cx;
	pBMI->bmiHeader.biHeight      = cy;
	pBMI->bmiHeader.biPlanes      = 1;
	pBMI->bmiHeader.biBitCount    = ibitcount;
	pBMI->bmiHeader.biClrUsed =   256;
	pBMI->bmiHeader.biCompression = BI_RGB; // 非压缩格式
	if(Palette==0)
	{
		for(int i=0;i<256;i++)
		{
			pBMI->bmiColors[i].rgbBlue=i;
			pBMI->bmiColors[i].rgbGreen=i;
			pBMI->bmiColors[i].rgbRed=i;
		} 
	}
	BOOL bRet = CreateDIBIndirect(pBMI, lpBits);
	delete[] pByte;
//	delete pBMI;
	return(bRet);
}


//////////////////////////////////////////////////////////////////
// 
// CreateDIBIndirect(LPBITMAPINFO pBMI, const void* lpBits)
// 
// 完成功能:
//     创建DIB位图
//
// 输入参数:
//	   位图信息结构指针 pBMI
//     图像数据 lpBits 
//
// 返回参数:
//	   是否成功
//
//////////////////////////////////////////////////////////////////

//创建DIB位图
BOOL CDIB::CreateDIBIndirect(LPBITMAPINFO pBMI, const void* lpBits)
{
	//摧毁原对象
	if (m_hObject != NULL)
	{
	DeleteObject();
	delete pBMI;
	}

	// Create the DIB section.
	//创建
	CDC *pDC = new CDC;
	pDC->CreateCompatibleDC(NULL);
	LPVOID pBits;
	HBITMAP hDIB = ::CreateDIBSection(pDC->GetSafeHdc(),
							pBMI,
							DIB_RGB_COLORS,
                           	&pBits,
                           	NULL,
                           	0);
//		::SetDIBits(pDC->GetSafeHdc(),hDIB,0,GetHeight(),LpBits,pBMI,DIB_RGB_COLORS);
	delete pDC;
	DeleteObject();
	ASSERT(hDIB);
	ASSERT(pBits);
	Attach(hDIB);

	//拷贝图像数据
	SetDIBBits(GetWidthBytes() * GetHeight(), lpBits);
//	::DeleteObject(hDIB);
//	delete pBMI;

	return TRUE;
}


//////////////////////////////////////////////////////////////////
// 
// SetDIBBits(DWORD dwCount,const void * lpBits)
// 
// 完成功能:
//     设置图像数据,将 lpBits 的数据拷贝至图像
//
// 输入参数:
//	   位图数据大小 dwCount
//     图像数据 lpBits 
//
// 返回参数:
//	   拷贝的数据大小
//
//////////////////////////////////////////////////////////////////

// Set DIB's bits 
// 设置图像数据
DWORD CDIB::SetDIBBits(DWORD dwCount,const void * lpBits)
{
	if(lpBits != NULL)
	{
		LPVOID pBits = GetBits();
		memcpy(pBits,lpBits,dwCount);
		return dwCount;
	//	::DeleteObject(pBits);
	}
	return 0;
}


//////////////////////////////////////////////////////////////////
// 
// GetDIBBits(DWORD dwCount, LPVOID lpBits)
// 
// 完成功能:
//     得到图像数据.
//     如果 lpBits 为空,则返回图像数据指针;
//	   如果 lpBits 不为空,
//             则将图像数据拷贝至该指针,
//             并返回图像数据指针;
//
// 输入参数:
//	   拷贝的位图数据大小 dwCount
//     图像数据 lpBits 
//
// 返回参数:
//	   图像数据指针
//
//////////////////////////////////////////////////////////////////

// Get DIB's bits
// 得到图像数据
LPVOID CDIB::GetDIBBits(DWORD dwCount, LPVOID lpBits)
{
	LPVOID pBits = GetBits();
	if(lpBits != NULL){
		memcpy(lpBits,pBits,dwCount);
		return pBits;
	}
	else{
		return pBits;
	}
}

//////////////////////////////////////////////////////////////////
// 
// operator = (CDIB& copy)
// 
// 完成功能:
//     重载赋值符
//
// 输入参数:
//	   要拷贝的位图 copy
//
// 返回参数:
//	   新图像数据
//
//////////////////////////////////////////////////////////////////

//重载赋值符
CDIB& CDIB::operator = (CDIB& copy)
{
 	DIBSECTION DibSection;
	//得到原图像信息
	copy.GetDibSection(&DibSection);
	int nSize = DibSection.dsBmih.biClrUsed*sizeof(RGBQUAD) + sizeof(BITMAPINFOHEADER);

	//申请新图像信息头内存
	BYTE *pByte = new BYTE[nSize];
	//拷贝信息
	
	memcpy(pByte, &(DibSection.dsBmih), sizeof(BITMAPINFOHEADER));

	CDC *pdc = copy.GetDC();
	//得到调色板信息
	::GetDIBColorTable(pdc->GetSafeHdc(), 0, DibSection.dsBmih.biClrUsed,
						(RGBQUAD*)(pByte+sizeof(BITMAPINFOHEADER)));
	copy.ReleaseDC(pdc);

	//创建新位图
	BITMAPINFO *pBMI = (BITMAPINFO*)pByte;
	CreateDIBIndirect(pBMI);

	//拷贝图像信息
	int nTotalSize = copy.GetWidthBytes() * copy.GetHeight();
	memcpy(GetBits(), copy.GetBits(), nTotalSize);

	delete[] pByte;
	return(*this);
}


//////////////////////////////////////////////////////////////////
// 
// SetPalette(UINT uStartIndex, UINT cEntries, CONST RGBQUAD *pColors)
// 
// 完成功能:
//     设置调色板
//
// 输入参数:
//	   调色板开始索引 uStartIndex
//     调色板入口 cEntries
//     颜色数据 pColors
//
// 返回参数:
//	   无
//
//////////////////////////////////////////////////////////////////

// Set the color table in the DIB section.
// 设置调色板
void CDIB::SetPalette(UINT uStartIndex, UINT cEntries, CONST RGBQUAD *pColors)
{
	HDC hdc = ::CreateCompatibleDC(NULL);
	HBITMAP hOld = (HBITMAP)::SelectObject(hdc, m_hObject);

	::SetDIBColorTable(hdc, uStartIndex, cEntries, pColors);
	
	::SelectObject(hdc, hOld);
	::DeleteObject(hdc);
}


//////////////////////////////////////////////////////////////////
// 
// SetPalette(CPalette* pPal)
// 
// 完成功能:
//     设置调色板
//
// 输入参数:
//	   调色板结构指针 pPal
//
// 返回参数:
//	   无
//
//////////////////////////////////////////////////////////////////

// 设置调色板
void CDIB::SetPalette(CPalette* pPal)
{
    ASSERT(pPal);

    // get the colors from the palette
    int iColors = 0;
    pPal->GetObject(sizeof(iColors), &iColors);
    ASSERT(iColors > 0);
    PALETTEENTRY* pPE = new PALETTEENTRY[iColors];
    pPal->GetPaletteEntries(0, iColors, pPE);

    // Build a table of RGBQUADS
    RGBQUAD* pRGB = new RGBQUAD[iColors];
    ASSERT(pRGB);
    for (int i = 0; i < iColors; i++) {
        pRGB[i].rgbRed = 0;//pPE[i].peRed;
        pRGB[i].rgbGreen =0; //pPE[i].peGreen;
        pRGB[i].rgbBlue =0;// pPE[i].peBlue;
        pRGB[i].rgbReserved = 0;
    }

	SetPalette(0, iColors, pRGB);

    delete [] pRGB;
    delete [] pPE;
}


//////////////////////////////////////////////////////////////////
// 
// GetDC(void)
// 
// 完成功能:
//     得到与位图相关的设备
//
// 输入参数:
//	   无
//
// 返回参数:
//	   与位图相关的设备指针
//
//////////////////////////////////////////////////////////////////

//得到与位图相关的设备
CDC* CDIB::GetDC(void)
{
	CDibDC* pdc = new CDibDC;
	if(pdc == NULL)
		return(NULL);
	pdc->CreateCompatibleDC(NULL);
	pdc->m_hOld = (HBITMAP)::SelectObject(pdc->GetSafeHdc(), GetSafeHandle());

	return(pdc);
}


//////////////////////////////////////////////////////////////////
// 
// ReleaseDC(CDC *pdc)
// 
// 完成功能:
//     得到与位图相关的设备
//
// 输入参数:
//	   与位图相关的设备
//
// 返回参数:
//	   是否成功
//
//////////////////////////////////////////////////////////////////

//释放得到的与位图相关的设备
BOOL CDIB::ReleaseDC(CDC *pdc)
{
	ASSERT(pdc != NULL);
	if(pdc->IsKindOf(RUNTIME_CLASS(CDibDC))){
		delete pdc;
		return(TRUE);
	}
	return(FALSE);
}

#ifdef _DEBUG
void CDIB::Dump(CDumpContext& dc) const
{
	CGdiObject::Dump(dc);

	if (m_hObject == NULL)
		return;

	BITMAP bm;
	VERIFY(GetObject(sizeof(bm), &bm));
	dc << _T("bm.bmType = ") << bm.bmType;
	dc << _T("\nbm.bmHeight = ") << bm.bmHeight;
	dc << _T("\nbm.bmWidth = ") << bm.bmWidth;
	dc << _T("\nbm.bmWidthBytes = ") << bm.bmWidthBytes;
	dc << _T("\nbm.bmPlanes = ") << bm.bmPlanes;
	dc << _T("\nbm.bmBitsPixel = ") << bm.bmBitsPixel;

	dc << _T("\n");
}
#endif
//////////////////////////////////////////////////////////////////
// 
// GetColorUsed()
// 
// 完成功能:
//     得到使用的颜色数
//
// 输入参数:
//	   无
//
// 返回参数:
//	   颜色数
//
//////////////////////////////////////////////////////////////////

//得到使用的颜色数
int CDIB::GetColorUsed()
{
	LPBITMAPINFOHEADER pBMIH;
	DIBSECTION DibSection;
	GetDibSection(&DibSection);
	pBMIH = &DibSection.dsBmih;
	return pBMIH->biClrUsed;
}

//////////////////////////////////////////////////////////////////
// 
// GetPalette()
// 
// 完成功能:
//     得到使用的调色板
//
// 输入参数:
//	   无
//
// 返回参数:
//	   调色板指针,用完应释放
//
//////////////////////////////////////////////////////////////////

//得到使用的调色板
CPalette * CDibDC::GetPalette()
{
	LOGPALETTE * pLogPal = (LOGPALETTE *)new char[sizeof(LOGPALETTE) +
		 256 * sizeof(PALETTEENTRY)];

	pLogPal->palVersion = 0x300;
	pLogPal->palNumEntries = 256;

	HDC hdc = GetSafeHdc();
	RGBQUAD pRGB[256];
	::GetDIBColorTable(hdc, 0, 256,pRGB);
	
	for(int i = 0 ; i < 256 ; i ++)
	{
		pLogPal->palPalEntry[i].peRed = pRGB[i].rgbRed;
		pLogPal->palPalEntry[i].peGreen = pRGB[i].rgbGreen;
		pLogPal->palPalEntry[i].peBlue = pRGB[i].rgbBlue;
		pLogPal->palPalEntry[i].peFlags = 0;
	}
	
	CPalette * pPal = NULL;
	pPal = new CPalette;	
	pPal->CreatePalette(pLogPal);

	delete pLogPal;
	return pPal;
}

//////////////////////////////////////////////////////////////////
// 
// GetBitmapInfo(void)
// 
// 完成功能:
//     得到位图信息
//
// 输入参数:
//	   无
//
// 返回参数:
//	   位图信息指针,用完应释放
//
//////////////////////////////////////////////////////////////////

//得到位图信息
LPBITMAPINFO CDIB::GetBitmapInfo(void)
{
	DIBSECTION DibSection;
	GetDibSection(&DibSection);
	int nSize = DibSection.dsBmih.biClrUsed*sizeof(RGBQUAD) + sizeof(BITMAPINFOHEADER);
	
	BYTE *pByte = new BYTE[nSize];
	memcpy(pByte, &(DibSection.dsBmih), sizeof(BITMAPINFOHEADER));
	CDC *pdc = GetDC();
	::GetDIBColorTable(pdc->GetSafeHdc(), 0, DibSection.dsBmih.biClrUsed,
						(RGBQUAD*)(pByte+sizeof(BITMAPINFOHEADER)));
	ReleaseDC(pdc);
	BITMAPINFO *pBMI = (BITMAPINFO*)pByte;
	return(pBMI);
}
//////////////////////////////////////////////////////////////////
// 
// CaptureDIB(CWnd * pWnd, const CRect& capRect)
// 
// 完成功能:
//     捕捉窗口图象
//
// 输入参数:
//	   窗口指针 pWnd
//     捕捉的大小 capRect
//
// 返回参数:
//	   是否成功
//
//////////////////////////////////////////////////////////////////

//捕捉窗口图象
BOOL CDIB::CaptureDIB(CWnd * pWnd, const CRect& capRect)
{
	BOOL ret = false;

	if(pWnd == NULL)
		return false;

	CDC * pPlayDc = pWnd->GetDC();

	if(pPlayDc == NULL)
		return false;

	CRect Rect;
	if(capRect.IsRectEmpty())
		pWnd->GetClientRect(Rect);
	else
		Rect = capRect;

	//得到图像颜色数
	UINT nBitCount = pPlayDc->GetDeviceCaps(BITSPIXEL);

	//创建位图
	if(CreateDIB(Rect.Width(), Rect.Height(), 0,nBitCount))
	{
		CDC * pCopyDc = GetDC();
		
		if(pCopyDc == NULL)
		{
			pWnd->ReleaseDC(pPlayDc);		
			return false;
		}

		pWnd->ShowWindow(SW_SHOW);
		//捕捉
		if(pCopyDc->BitBlt(0, 0, Rect.Width(), Rect.Height(), pPlayDc, 0, 0, SRCCOPY))
			ret = true;
		
		ReleaseDC(pCopyDc);
	}

	pWnd->ReleaseDC(pPlayDc);		
	return ret;
}


////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////
//
// CDibDC
//
// 设备无关位图设备类(版本1.0)
//
// 完成功能:
//
// 与设备无关位图的相关联
//
//////////////////////////////////////////////////////////////////////

IMPLEMENT_DYNAMIC(CDibDC, CDC);

CDibDC::CDibDC()
{
	m_hOld = NULL;
}

CDibDC::~CDibDC()
{
	if(m_hOld != NULL){
		::SelectObject(GetSafeHdc(), m_hOld);
	}
}
LPVOID CDIB::GetMemImgData(const void *LpMem)
{   DWORD i,Ipoint,Mpoint;
	LPVOID LpImg;
    LpImg=(LPVOID)new char[640*480*3];
    for(i=0,Ipoint=0,Mpoint=1;i<640*480;i++,Ipoint+=3,Mpoint+=4)
	     memcpy((LPSTR)LpImg+Ipoint,(LPSTR)LpMem+Mpoint,3);
	return LpImg;
}


