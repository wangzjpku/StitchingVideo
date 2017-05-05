#include <iostream>
#include <fstream>
#include <string>
#include "opencv2/opencv_modules.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/stitching/detail/autocalib.hpp"
#include "opencv2/stitching/detail/blenders.hpp"
#include "opencv2/stitching/detail/camera.hpp"
#include "opencv2/stitching/detail/exposure_compensate.hpp"
#include "opencv2/stitching/detail/matchers.hpp"
#include "opencv2/stitching/detail/motion_estimators.hpp"
#include "opencv2/stitching/detail/seam_finders.hpp"
#include "opencv2/stitching/detail/util.hpp"
#include "opencv2/stitching/detail/warpers.hpp"
#include "opencv2/stitching/warpers.hpp"
#include "stitching_sharemem.h"

using namespace std;
using namespace cv;
using namespace cv::detail;

int Stitching_sharemem::ShareMemPre()  
{  
    hMutexSti = OpenMutex(MUTEX_ALL_ACCESS,  
        FALSE,  
        OpMutex);  
    if (NULL == hMutexSti)  
    {  
        if (ERROR_FILE_NOT_FOUND == GetLastError())  
        {  
            cout << "OpenMutex fail: file not found!" << endl;  
			return -1; 
        }  
        else  
        {  
            cout << "OpenMutex fail:" << GetLastError() << endl;  
			return -1; 
        }  
    }  
  
    if (WaitForSingleObject(hMutexSti, 5000) != WAIT_OBJECT_0)//hMutex 一旦互斥对象处于有信号状态，则该函数返回  
    {  
        DWORD dwErr = GetLastError();  
        return -1; 
    }  
  
    //open share memory  
    hFileMapping = OpenFileMapping(FILE_MAP_ALL_ACCESS,  
        FALSE,  
        OpShareMem);  
    if (NULL == hFileMapping)  
    {  
        cout << "OpenFileMapping" << GetLastError() << endl;  
        return -1;  
    }  
  
    lpShareMemory = MapViewOfFile(hFileMapping,  
        FILE_MAP_ALL_ACCESS,  
        0,  
        0,  
        0);  
    if (NULL == lpShareMemory)  
    {  
        cout << "MapViewOfFile" << GetLastError() << endl;  
        return -1;  
    }  
    //read and write data  
    hServerWriteOver = CreateEvent(NULL,  
        TRUE,  
        FALSE,  
        OpServerOver);  
    hClientReadOver = CreateEvent(NULL,  
        TRUE,  
        FALSE,  
        OpClientOver);  
    if (NULL == hServerWriteOver ||  
        NULL == hClientReadOver)  
    {  
        cout << "CreateEvent" << GetLastError() << endl;  
       	return -1;  
    }   
	return 0;
}  

 int Stitching_sharemem::SharememClose() 
{
    //release share memory  
    if (NULL != hServerWriteOver)   CloseHandle(hServerWriteOver);  
    if (NULL != hClientReadOver)    CloseHandle(hClientReadOver);  
    if (NULL != lpShareMemory)      UnmapViewOfFile(lpShareMemory);  
    if (NULL != hFileMapping)       CloseHandle(hFileMapping);  
    if (NULL != hMutexSti)          ReleaseMutex(hMutexSti);  
    return 0;  
}

int Stitching_sharemem::GetMemData( Mat &pCvMat)
{
    char p = 0;  
    char* q = (char*)lpShareMemory;  

    if (!SetEvent(hClientReadOver))   
	{
		return -1;
	} 
  
    if (WaitForSingleObject(hServerWriteOver, INFINITE) != WAIT_OBJECT_0)   
	{
		return -1;
	}   
 
	memcpy( &Sharepacket , q , sizeof(DatePacket));
    if (pCvMat.empty())
		pCvMat.create(cv::Size(Sharepacket.width, Sharepacket.height),CV_8UC3 );	

	memcpy(pCvMat.data, q + Sharepacket.buferPointer + 1 , Sharepacket.flag);

    if (!ResetEvent(hServerWriteOver))   
	{
		return -1;
	}

	return 0;  
}
