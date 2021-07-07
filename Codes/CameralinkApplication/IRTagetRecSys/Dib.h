#ifndef __DIB_H
#define __DIB_H
//
// CDIB -
// CDibDC -
//
// Implements a simple encapsulation of a DIB section and a DC.
//
//

//////////////////////////////////////////////////////////////////////
//
// CDIB
//
// 设备无关位图类
//
// 完成功能:
//
// 设备无关位图的创建,显示,读入,保存
//
//////////////////////////////////////////////////////////////////////


class CDIB;
class CDibDC;

class CDIB : public CGdiObject
{
	DECLARE_DYNAMIC(CDIB)

public:
	//由句柄创建位图
	static CDIB* PASCAL FromHandle(HBITMAP hBitmap);

// Constructors
	CDIB();

	//创建位图
	BOOL CreateDIB(int nWidth, int nHeight,int Palette,UINT nBitcount, const void* lpBits=NULL);

	//创建位图
	BOOL CreateDIBIndirect(LPBITMAPINFO lpBitmap, const void* lpBits=NULL);

	//捕捉窗口图像
	BOOL CaptureDIB(CWnd * pWnd, const CRect& capRect = CRect(0,0,0,0));

// Attributes
	//得到位图
	operator HBITMAP() const;

	//拷贝位图
	CDIB& operator = (CDIB& copy);

// Operations
	//设置图像数据
	DWORD SetDIBBits(DWORD dwCount, const void* lpBits);
	//得到图像数据
	LPVOID GetDIBBits(DWORD dwCount = 0, LPVOID lpBits = NULL);

// Implementation
public:
	virtual ~CDIB();
#ifdef _DEBUG
	virtual void Dump(CDumpContext& dc) const;
#endif

// Newly added functions
public:
	//得到使用的颜色数
	int GetColorUsed();

	//设置调色板
	void SetPalette(UINT uStartIndex, UINT cEntries, CONST RGBQUAD *pColors);
	//设置调色板
	void SetPalette(CPalette* pPal);
	
	//得到设备
	CDC* GetDC(void);
	//释放设备
	static BOOL ReleaseDC(CDC *pdc);

	//得到位图
	int GetBitmap(BITMAP* pBitMap);
	//得到DibSection
	int GetDibSection(DIBSECTION* pDibSection);
	//得到固定内存的图像数据
	LPVOID GetMemImgData(const void *pMem);
	//得到宽度
	int GetWidth();
	//得到像元字节数
	int CDIB::GetBytesPixel(void);
	//得到高度
	int GetHeight();
	//得到尺寸
	SIZE GetSize();
	//得到每行图像字节数
	int GetWidthBytes();
	//得到图像位数
	int GetBitCount();
	//得到图像数据
	LPVOID GetBits();
	//得到图像信息头
	LPBITMAPINFO GetBitmapInfo(void);

	int  GetDataFromIR(WORD* pIRData,BYTE nType,WORD wMaxLimit,WORD wMinLimit); //转换为灰度图
	int  Get2bitDataFromIR(WORD* pIRData,WORD wTh);
};

////////////////////////////////////////////////////////////////////
// inline functions

//////////////////////////////////////////////////////////////////
// 
// GetBitmap(BITMAP* pBitMap)
// 
// 完成功能:
//     得到位图
//
// 输入参数:
//	   位图指针 pBitMap
//
// 返回参数:
//	   是否成功
//
//////////////////////////////////////////////////////////////////

//得到位图
inline int CDIB::GetBitmap(BITMAP* pBitMap)
{
	return(::GetObject(m_hObject, sizeof(BITMAP), pBitMap));
}


//////////////////////////////////////////////////////////////////
// 
// GetDibSection(DIBSECTION *pDibSection)
// 
// 完成功能:
//     得到DibSection
//
// 输入参数:
//	   DibSection指针 pDibSection
//
// 返回参数:
//	   是否成功
//
//////////////////////////////////////////////////////////////////

//得到DibSection
inline int CDIB::GetDibSection(DIBSECTION *pDibSection)
{
	return(::GetObject(m_hObject, sizeof(DIBSECTION), pDibSection));
}


//////////////////////////////////////////////////////////////////
// 
// HBITMAP()
// 
// 完成功能:
//     得到位图句柄
//
// 输入参数:
//	   无
//
// 返回参数:
//	   位图句柄
//
//////////////////////////////////////////////////////////////////

//得到位图句柄
inline CDIB::operator HBITMAP() const
{
	return (HBITMAP)(this == NULL ? NULL : m_hObject);
}


//////////////////////////////////////////////////////////////////
// 
// FromHandle(HBITMAP hDib)
// 
// 完成功能:
//     从位图句柄得到位图类
//
// 输入参数:
//	   位图句柄 hDib
//
// 返回参数:
//	   位图类
//
//////////////////////////////////////////////////////////////////

//从位图句柄得到位图类
inline CDIB* PASCAL CDIB::FromHandle(HBITMAP hDib)
{
	return((CDIB*)CGdiObject::FromHandle(hDib));
}


//////////////////////////////////////////////////////////////////
// 
// GetWidth(void)
// 
// 完成功能:
//     得到宽度
//
// 输入参数:
//	   无
//
// 返回参数:
//	   宽度
//
//////////////////////////////////////////////////////////////////

//得到宽度
inline int CDIB::GetWidth(void)
{
	BITMAP bmp;
	GetBitmap(&bmp);
	return(bmp.bmWidth);
}

//////////////////////////////////////////////////////////////////
// 
// GetBitsPixel(void)
// 
// 完成功能:
//     得到像元字节数
//
// 输入参数:
//	   无
//
// 返回参数:
//	   像元字节数
//
//////////////////////////////////////////////////////////////////

//得到像元字节数
inline int CDIB::GetBytesPixel(void)
{
	BITMAP bmp;
	GetBitmap(&bmp);
	return(bmp.bmBitsPixel/8);
}
//////////////////////////////////////////////////////////////////
// 
// GetHeight(void)
// 
// 完成功能:
//     得到高度
//
// 输入参数:
//	   无
//
// 返回参数:
//	   高度
//
//////////////////////////////////////////////////////////////////

//得到高度
inline int CDIB::GetHeight(void)
{
	BITMAP bmp;
	GetBitmap(&bmp);
	return(bmp.bmHeight);
}


//////////////////////////////////////////////////////////////////
// 
// GetSize(void)
// 
// 完成功能:
//     得到尺寸
//
// 输入参数:
//	   无
//
// 返回参数:
//	   尺寸
//
//////////////////////////////////////////////////////////////////

//得到尺寸
inline SIZE CDIB::GetSize(void)
{
	BITMAP bmp;
	GetBitmap(&bmp);
	CSize size(bmp.bmWidth, bmp.bmHeight);
	return(size);
}


//////////////////////////////////////////////////////////////////
// 
// GetWidthBytes(void)
// 
// 完成功能:
//     得到每行字节数
//
// 输入参数:
//	   无
//
// 返回参数:
//	   每行字节数
//
//////////////////////////////////////////////////////////////////

//得到每行字节数
inline int CDIB::GetWidthBytes(void)
{
	BITMAP bmp;
	GetBitmap(&bmp);
	return(bmp.bmWidthBytes);
}


//////////////////////////////////////////////////////////////////
// 
// GetBitCount(void)
// 
// 完成功能:
//     得到图像位数
//
// 输入参数:
//	   无
//
// 返回参数:
//	   图像位数
//
//////////////////////////////////////////////////////////////////

//得到图像位数
inline int CDIB::GetBitCount(void)
{
	BITMAP bmp;
	GetBitmap(&bmp);
	return(bmp.bmBitsPixel);
}


//////////////////////////////////////////////////////////////////
// 
// GetBits(void)
// 
// 完成功能:
//     得到图像数据
//
// 输入参数:
//	   无
//
// 返回参数:
//	   图像数据
//
//////////////////////////////////////////////////////////////////

//得到图像数据
inline LPVOID CDIB::GetBits(void)
{
	BITMAP bmp;
	GetBitmap(&bmp);
	return(bmp.bmBits);
}
inline int CDIB::GetDataFromIR(WORD* pIRData,BYTE nType,WORD wMaxLimit,WORD wMinLimit) //转换为灰度图,0:直方图;1：直接映射
{
	int w= GetWidth();
	int h =GetHeight();		
	BYTE* pImage =(BYTE*)GetBits();
	int kk =0;

	if (nType==0)
	{
		UCHAR  m_pbyHist[65536];
		UCHAR  m_pLUT[65536];

		BYTE* pImage =(BYTE*)GetBits();
		DWORD sum=0,HistSum=0;
		DWORD HistSumH=0,HistSumL=0,HistMean=0;

		memset(m_pLUT,0,65536*sizeof(UCHAR));
		memset(m_pbyHist,0,65536*sizeof(UCHAR));

		int kk =0;	
		while (kk<(h*w))
		{
			WORD wIrValue =pIRData[kk];
			if(wIrValue>=wMaxLimit)
			{
				wIrValue=wMaxLimit;
			}
			if(wIrValue<=wMinLimit)
			{
				wIrValue=wMinLimit;
			}
			if(m_pbyHist[wIrValue]<255) 
			{
				m_pbyHist[wIrValue]++;
			}
			kk++;
		}
		HistSum=0;
		for (int  i=0;i<65536;i++)
		{
			if (m_pbyHist[i]<3)
			{
				m_pbyHist[i]=0;
			}
			HistSum+=m_pbyHist[i];
		}
		sum=0;
		int nTemp;
		for (int i=0;i<65536;i++)
		{
			sum+=m_pbyHist[i];
			if (HistSum!=0)
			{
				m_pLUT[i]=255.00*sum/HistSum;
				nTemp=m_pLUT[i];
				if(nTemp<0)
				{
					nTemp=0;
				}
				else if(nTemp>255)
				{
					nTemp=255;
				}
				m_pLUT[i]=nTemp;
			}
		}
		for(int i =0;i<w;i++)
		{
			for(int j =0;j<h;j++)
			{
				if (pIRData[j*w+i]==65535)
				{
					pImage[((h-1-j)*(w)+i)] =255;
				}
				else if (pIRData[j*w+i]==0)
				{
					pImage[((h-1-j)*(w)+i)] =0;
				}
				else
				{
					pImage[((h-1-j)*(w)+i)] = m_pLUT[pIRData[j*w+i]];
				}				
			}
		}
	}
	else
	{
		WORD maxV=0,minV=65536;
		while (kk<(h*w))
		{
			if(pIRData[kk]>=maxV)
			{
				maxV=pIRData[kk];
			}
			if(minV>=pIRData[kk])
			{
				minV=pIRData[kk];
			}
			kk++;
		}
		if(maxV>=wMaxLimit)
		{
			maxV=wMaxLimit;
		}
		if(minV<=wMinLimit)
		{
			minV=wMinLimit;
		}
		int nsample = 0; 
		float span = (float)(maxV - minV+ 1);
		for(int i =0;i<w;i++)
		{
			for(int j =0;j<h;j++)
			{
				if (pIRData[j*w+i]==65535)
				{
					nsample =255;
				}
				else if (pIRData[j*w+i]==0)
				{
					nsample =0;
				}
				else if (pIRData[j*w+i]>maxV)
				{
					nsample=255;
				}
				else if (pIRData[j*w+i]<minV)
				{
					nsample=0;
				}
				else
				{
					nsample =(int)(((pIRData[j*w+i]-minV)*1.0/span)*255);
				}
				pImage[((h-1-j)*(w)+i)] = nsample;
			}
		}
	}
	return TRUE;
}
inline int CDIB::Get2bitDataFromIR(WORD* pIRData,WORD wTh) //转换为灰度二值图
{
	int w= GetWidth();
	int h =GetHeight();		
	BYTE* pImage =(BYTE*)GetBits();
	int kk =0;
	for(int i =0;i<w;i++)
	{
		for(int j =0;j<h;j++)
		{
			if (pIRData[j*w+i]>wTh)
			{
				pImage[((h-1-j)*(w)+i)] = 255;
			}
			else
			{
				pImage[((h-1-j)*(w)+i)]=0;
			}
		}
	}
	return TRUE;
}
//////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////
//
// CDibDC
//
// 设备无关位图设备类(版本1.0)
//
// 完成功能:
//     与设备无关位图的相关联
//
//////////////////////////////////////////////////////////////////////


class CDibDC : public CDC
{
	DECLARE_DYNAMIC(CDibDC)

// Constructors
public:
	CDibDC();

// Attributes
protected:
	HBITMAP m_hOld;

// Implementation
public:
	CPalette * GetPalette();
	virtual ~CDibDC();
	
	friend class CDIB;
};

#endif //__DIB_H