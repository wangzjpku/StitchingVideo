 
#include <iostream> 
#include <Windows.h> 

using namespace std;
using namespace cv;
using namespace cv::detail;

class Stitching_sharemem
{
public:	
	Stitching_sharemem( wchar_t *l1 ,wchar_t *l2,wchar_t *l3,wchar_t *l4  )
	{
		hMutexSti        =NULL;  
		hFileMapping     =NULL;  
		lpShareMemory    =NULL;  
		hServerWriteOver =NULL;  
		hClientReadOver  =NULL;  

		OpMutex	 = l1;
		OpShareMem = l2;
		OpServerOver	 = l3;
		OpClientOver	 = l4;
	}

	int ShareMemPre();
	int GetMemData( Mat &pCvMat);
	int SharememClose();
private:
	HANDLE hMutexSti        ;  
	HANDLE hFileMapping     ;  
	LPVOID lpShareMemory    ;  
	HANDLE hServerWriteOver ;  
	HANDLE hClientReadOver  ;   
	wchar_t *OpMutex		;
	wchar_t *OpShareMem		;
	wchar_t *OpServerOver	;
	wchar_t *OpClientOver	;

	struct DatePacket{
    char filePath[120];
	int width;
	int height;
    int flag;
    int buferPointer;
	}Sharepacket;

};








