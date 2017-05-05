
#include "stitching_sharemem.h"

using namespace std;
using namespace cv;
using namespace cv::detail;

struct DatePacket{
    char filePath[120];
	int width;
	int height;
    int flag;
    int buferPointer;
}mempacket1,mempacket2,mempacket3,mempacket4,mempacket5;

HANDLE hMutexSti        = NULL;  
HANDLE hFileMapping     = NULL;  
LPVOID lpShareMemory    = NULL;  
HANDLE hServerWriteOver = NULL;  
HANDLE hClientReadOver  = NULL;   

HANDLE hMutexSti2        = NULL;  
HANDLE hFileMapping2     = NULL;  
LPVOID lpShareMemory2    = NULL;  
HANDLE hServerWriteOver2 = NULL;  
HANDLE hClientReadOver2  = NULL;   

int ShareMemPre()  
{  
    hMutexSti = OpenMutex(MUTEX_ALL_ACCESS,  
        FALSE,  
        L"SM_MutexRTSPData1");  
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
        L"ShareMemoryRTSPData1");  
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
        L"ServerWriteOver1");  
    hClientReadOver = CreateEvent(NULL,  
        TRUE,  
        FALSE,  
        L"ClientReadOver1");  
    if (NULL == hServerWriteOver ||  
        NULL == hClientReadOver)  
    {  
        cout << "CreateEvent" << GetLastError() << endl;  
       	return -1;  
    }   
	return 0;
}  


int ShareMemPre2()  
{  
    hMutexSti2 = OpenMutex(MUTEX_ALL_ACCESS,  
        FALSE,  
        L"SM_MutexRTSPData2");  
    if (NULL == hMutexSti2)  
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
  
    if (WaitForSingleObject(hMutexSti2, 5000) != WAIT_OBJECT_0)//hMutex 一旦互斥对象处于有信号状态，则该函数返回  
    {  
        DWORD dwErr = GetLastError();  
        return -1; 
    }  
  
    //open share memory  
    hFileMapping2 = OpenFileMapping(FILE_MAP_ALL_ACCESS,  
        FALSE,  
        L"ShareMemoryRTSPData2");  
    if (NULL == hFileMapping2)  
    {  
        cout << "OpenFileMapping" << GetLastError() << endl;  
        return -1;  
    }  
  
    lpShareMemory2 = MapViewOfFile(hFileMapping2,  
        FILE_MAP_ALL_ACCESS,  
        0,  
        0,  
        0);  
    if (NULL == lpShareMemory2)  
    {  
        cout << "MapViewOfFile" << GetLastError() << endl;  
        return -1;  
    }  
    //read and write data  
    hServerWriteOver2 = CreateEvent(NULL,  
        TRUE,  
        FALSE,  
        L"ServerWriteOver2");  
    hClientReadOver2 = CreateEvent(NULL,  
        TRUE,  
        FALSE,  
        L"ClientReadOver2");  
    if (NULL == hServerWriteOver2 ||  
        NULL == hClientReadOver2)  
    {  
        cout << "CreateEvent" << GetLastError() << endl;  
       	return -1;  
    }   
	return 0;
}  

int ShareMemClose() 
{
    //release share memory  
    if (NULL != hServerWriteOver)   CloseHandle(hServerWriteOver);  
    if (NULL != hClientReadOver)    CloseHandle(hClientReadOver);  
    if (NULL != lpShareMemory)      UnmapViewOfFile(lpShareMemory);  
    if (NULL != hFileMapping)       CloseHandle(hFileMapping);  
    if (NULL != hMutexSti)          ReleaseMutex(hMutexSti);  

    //release share memory  
    if (NULL != hServerWriteOver2)   CloseHandle(hServerWriteOver2);  
    if (NULL != hClientReadOver2)    CloseHandle(hClientReadOver2);  
    if (NULL != lpShareMemory2)      UnmapViewOfFile(lpShareMemory2);  
    if (NULL != hFileMapping2)       CloseHandle(hFileMapping2);  
    if (NULL != hMutexSti2)          ReleaseMutex(hMutexSti2);  

    return 0;  
}

int SendMemData( Mat &pCvMat)
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
 
	memcpy( &mempacket1 , q , sizeof(DatePacket));
    if (pCvMat.empty())
		pCvMat.create(cv::Size(mempacket1.width, mempacket1.height),CV_8UC3 );	

	memcpy(pCvMat.data, q + mempacket1.buferPointer + 1 , mempacket1.flag);

    if (!ResetEvent(hServerWriteOver))   
	{
		return -1;
	}

	return 0;  
}

int SendMemData2( Mat &pCvMat)
{
    char p = 0;  
    char* q = (char*)lpShareMemory2;  

	//namedWindow("MyPicture3",0);
	//moveWindow("MyPicture3", 300,300);  
	//resizeWindow("MyPicture3" , 1000 , 300  );
	//FILE *fp_rgb = fopen("output.rgb", "wb+");
	//do   
    //{  
        if (!SetEvent(hClientReadOver2))   
		{
			return -1;
		} 
  
        if (WaitForSingleObject(hServerWriteOver2, INFINITE) != WAIT_OBJECT_0)   
		{
			return -1;
		}   
 
		memcpy( &mempacket2 , q , sizeof(DatePacket));
    	if (pCvMat.empty())
		    pCvMat.create(cv::Size(mempacket2.width, mempacket2.height),CV_8UC3 );	

		memcpy(pCvMat.data, q + mempacket2.buferPointer + 1 , mempacket2.flag);
		//fwrite(pCvMat->data,1,mempacket.flag,fp_rgb);  //write BGR data;

		//imshow( "MyPicture3", pCvMat  );
		//printf( "This rtsp file path is %d  %d \n" , mempacket1.width , mempacket1.height ); 
		//waitKey(30) ;
        if (!ResetEvent(hServerWriteOver2))   
		{
			return -1;
		}
    //} while (p != '\n');   

	return 0;  
}

int ReceivMemData()
{
	return 0;  
}
