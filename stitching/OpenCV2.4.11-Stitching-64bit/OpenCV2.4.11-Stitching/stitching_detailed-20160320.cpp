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
#include  <windows.h> 

//#include "opencv2/imgproc/imgproc.hpp"

using namespace std;
using namespace cv;
using namespace cv::detail;

HANDLE hMutex;
//HANDLE hMutexRemap;
HANDLE hMutexMessg;
//排序图片 从1到n 必须从做到有 1 到 N 
char outbuf[50];
char testfile[50];
int num_images = 3 ;// camera numbers 
/////////////////////////////////////////////////////////////////////////////公用变量
//第一图像宽高和img mask
vector<VideoCapture> capture(num_images);
vector<Mat>	xmapstitch(num_images), ymapstitch(num_images);
vector<Mat>	xmap1(num_images), ymap1(num_images); 
vector<Mat> mapEye1(num_images), mapEye2(num_images);
int loopNum = 0;
////////////////////////////////////////////////////////////////////////////预处理变量
Mat result1, result_mask1; 
Mat  xmapdet, ymapdet,indxmap;
vector<Rect> dst_roi(num_images);
vector<Point> corners(num_images);
Rect dst_roi_1 ;
int srctype = 0;
////////////////////////////////////////////////////////////////////////////拼接变量
Mat ResultStitch;
Mat Result_mask_stitch;
Mat  xmapdetstitch, ymapdetstitch,indxmapstitch;
vector<Mat> img4(num_images);
////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////
int gEventPre = 0;
int gEventSting = 0;

//int	pMessageUpdate = 0;
//int pMessageEnd = 0;
////////////////////////////////////////////////////////////////////////////
#define PRESTI 0
#define STITCH 1

#define CONTUN 0
#define RUNSTI 1
#define UPDATE 2
#define ENDOUT 3
#define WAITIN 4

int HandlePreMessage = CONTUN;
int HandleStiMessage = WAITIN;

int SendMessage( int message , int type )
{
    WaitForSingleObject( hMutexMessg , INFINITE  );    
    if (type == PRESTI)
        HandlePreMessage = message;
    else if (type == STITCH)
        HandleStiMessage = message;
    ReleaseMutex(hMutexMessg);
    return 0;
}
int RecvMessage( int &message , int type)
{
    WaitForSingleObject( hMutexMessg , INFINITE  );
    if (type == PRESTI)
        message = HandlePreMessage;
    else if (type == STITCH)
        message = HandleStiMessage;
    ReleaseMutex(hMutexMessg);
    return 0;
}

//得到参数拼一个，继续其他图片的拼接计算

void Set4By1(const cv::Mat& src,cv::Mat& dst,
	cv::Point srcPt,cv::Point dstPt,
	int nHalfWidthSrc,int nHalfHeightSrc,
	int nHalfWidthDst,int nHalfHeightDst)
{
	int u = srcPt.x;
	int v = srcPt.y;
	int x = dstPt.x;
	int y = dstPt.y;
	Point srcPoint[4] = 
		{Point(u+nHalfWidthSrc,v+nHalfHeightSrc),	//右上)
		Point(u+nHalfWidthSrc,nHalfHeightSrc-v),
		Point(nHalfWidthSrc-u,nHalfHeightSrc-v),
		Point(nHalfWidthSrc-u,nHalfHeightSrc+v)};

	Point dstPoint[4] = 
		{Point(x+nHalfWidthDst,y+nHalfHeightDst),	//右上)
		Point(x+nHalfWidthDst,nHalfHeightDst-y),
		Point(nHalfWidthDst-x,nHalfHeightDst-y),
		Point(nHalfWidthDst-x,nHalfHeightDst+y)};

	bool bGray = src.channels()==3 ? false : true;

	for (int i=0; i<4; ++i)
	{
		if (srcPoint[i].x<src.cols && srcPoint[i].y<src.rows &&
			srcPoint[i].x>=0 && srcPoint[i].y>=0 /*&&
			dstPoint[i].x<dst.cols && dstPoint[i].y<dst.rows &&
			dstPoint[i].x>=0 && dstPoint[i].y>=0*/)
		{
			if (bGray)
			{
				dst.at<uchar>(dstPoint[i]) = src.at<uchar>(srcPoint[i]);
			}
			else
			{
				dst.at<Vec3b>(dstPoint[i]) = src.at<Vec3b>(srcPoint[i]);
			}
		}
	}
}

Rect resultRoiSize(const vector<Point> &corners, const vector<Size> &sizes)
{
    CV_Assert(sizes.size() == corners.size());
    Point tl(numeric_limits<int>::max(), numeric_limits<int>::max());
    Point br(numeric_limits<int>::min(), numeric_limits<int>::min());
    for (size_t i = 0; i < corners.size(); ++i)
    {
        tl.x = min(tl.x, corners[i].x);
        tl.y = min(tl.y, corners[i].y);
        br.x = max(br.x, corners[i].x + sizes[i].width);
        br.y = max(br.y, corners[i].y + sizes[i].height);
    }
    return Rect(tl, br);
}

void prepareSize(Rect dst_roi , Mat &dst_ , Mat &dst_mask_ , Rect &dst_roi_)
{
    dst_.create(dst_roi.size(), CV_16SC3);
    dst_.setTo(Scalar::all(0));
    dst_mask_.create(dst_roi.size(), CV_8U);
    dst_mask_.setTo(Scalar::all(0));
    dst_roi_ = dst_roi;

	xmapdet.create(dst_roi.size(), CV_32F);
	xmapdet.setTo(Scalar::all(0));
	ymapdet.create(dst_roi.size(), CV_32F);
	ymapdet.setTo(Scalar::all(0));
	indxmap.create(dst_roi.size(), CV_32F);
	indxmap.setTo(Scalar::all(0));

}

void feedSize(const Mat &img, const Mat &mask, Point tl , Rect &dst_roi_ ,Mat &dst_ , Mat &dst_mask_ ,int imgidx )
{
    CV_Assert(img.type() == CV_16SC3);
    CV_Assert(mask.type() == CV_8U);
    int dx = tl.x - dst_roi_.x;
    int dy = tl.y - dst_roi_.y;

    for (int y = 0; y < img.rows ; ++y) //img.rows
    {
        const Point3_<short> *src_row = img.ptr<Point3_<short> >(y);
        Point3_<short> *dst_row = dst_.ptr<Point3_<short> >(dy + y);
        const uchar *mask_row = mask.ptr<uchar>(y);
        uchar *dst_mask_row = dst_mask_.ptr<uchar>(dy + y);

		float *dst_x = xmapdet.ptr<float>(dy + y);
		float *dst_y = ymapdet.ptr<float>(dy + y);
		float *dst_ix = indxmap.ptr<float>(dy + y);

        for (int x = 0; x < img.cols; ++x)  //img.cols
        {
            if (mask_row[x])
			{
                dst_row[dx + x] = src_row[x];            

				dst_x[dx + x] = (float)x;
				dst_y[dx + x] = (float)y;
				dst_ix[dx + x] = (float)imgidx;				
			}	
			dst_mask_row[dx + x] |= mask_row[x];
        }
    }
}

void feedSizeRemap(const vector<Mat> &img, Mat &dst_ , Mat &dst_mask_  )
{
    for (int y = 0; y < dst_.rows ; ++y)  
    {
        Point3_<short> *dst_row = dst_.ptr<Point3_<short> >( y );
        uchar *dst_mask_row = dst_mask_.ptr<uchar>( y );

		float *dst_x = xmapdetstitch.ptr<float>( y );
		float *dst_y = ymapdetstitch.ptr<float>( y );
		float *dst_z = indxmapstitch.ptr<float>( y );

        for (int x = 0; x < dst_.cols; ++x)   
        {
			if (dst_mask_row[x])
			{					
					//LOGLN("xyz map " << dst_z[x] << ".."<< dst_y[x] <<".." << dst_x[x] << "\n");	
					//LOGLN("Row Cols map " << (img[dst_z[x]].cols) << ".."<< img[dst_z[x]].rows <<"\n");	
					//if (( dst_y[x] < img[dst_z[x]].cols) && (  dst_x[x] < img[dst_z[x]].rows ))
					//{
						//int xx = (int)dst_y[x];
						//int yy = (int)dst_x[x];
						dst_row[x] = img[dst_z[x]].at< Point3_<short> >(  (int)dst_y[x] , (int)dst_x[x]  ) ;						
					//}				
			}
        }
    }

}

void blendSize(Mat &dst, Mat &dst_mask , Mat &dst_ , Mat &dst_mask_ )
{
    dst_.setTo(Scalar::all(0), dst_mask_ == 0);
    dst = dst_;
    dst_mask = dst_mask_;
//    dst_.release();
//    dst_mask_.release();
}

Rect warpsize( RotationWarper * warper  , Mat &src,  Mat &K, const Mat &R, int interp_mode, int border_mode,
                                  Mat &dst , Mat &xmap, Mat &ymap )
{

    Rect dst_roi = warper->buildMaps(src.size(), K, R, xmap, ymap);

    dst.create(dst_roi.height + 1, dst_roi.width + 1, src.type());
    remap(src, dst, xmap, ymap, interp_mode, border_mode);

    return dst_roi;
}

/*
////////////////////////////////////////////////////////////////////////////////////////////////////////////
#define        _I(img,x,y) ((unsigned char*)((img)->imageData + (img)->widthStep*(y)))[(x)]
//亚像素级灰度值
#define        _IF(image,x,y)    ( ((int)(x+1)-(x))*((int)(y+1)-(y))*_I((image),(int)(x),(int)(y)) + ((int)(x+1)-(x))*((y)-(int)(y))*_I((image),(int)(x),(int)(y+1)) + ((x)-(int)(x))*((int)(y+1)-(y))*_I((image),(int)(x+1),(int)(y)) + ((x)-(int)(x))*((y)-(int)(y))*_I((image),(int)(x+1),(int)(y+1)) )//插值后的像素值(IN表示interpolation),x、y可以为小数

// 镜头校正查找表    
CvPoint CorrInd[480][640];      
// 镜头径向畸变校正初始化    
//void LensCorrectInit( double k1 = -0.00000035, double k2 = -0.00000035 );      
   
// 镜头径向畸变校正初始化    
void LensCorrectInit( double k1, double k2 , int imgWidth , int imgHeight)   
{   
    //int imgWidth = camera.GetWidth();   // 获得摄像头图像尺寸    
    //int imgHeight = camera.GetHeight();   
   
    int imgWd2 = imgWidth>>1;   
    int imgHd2 = imgHeight>>1;   
   
    int it;   
    int jt;   
    int it2;   
    int jt2;   
   
    for( int i=0; i<imgHeight; i++ )   
    {   
        it = ( i + 1 ) - imgHd2; // (i+1) 是其真正的图像坐标    
        for( int j=0; j<imgWidth; j++ )   
        {   
            double temp;   
   
            jt = ( j + 1 ) - imgWd2; // (j+1) 是其真正的图像坐标    
            temp = 1 + k1*it*it + k2*jt*jt;   
            it2 = (int)(temp * it);   
            jt2 = (int)(temp * jt);   
   
            it2 = it2 + imgHd2 + 1;   
            jt2 = jt2 + imgWd2 + 1;   
   
            CorrInd[i][j].x = jt2;   
            CorrInd[i][j].y = it2;   
        }   
    }   
}      
   
   
// 镜头径向畸变校正    
void LensCorrect( IplImage* src, IplImage* dst )   
{   
    int imgWidth = dst->width;   
    int imgHeight = dst->height;   
   
    int it2;   
    int jt2;   
   
    for( int i=0; i<imgHeight; i++ )   
    {   
        uchar *pdstdata = (uchar*)(dst->imageData) + i*dst->widthStep;   
   
        for( int j=0; j<imgWidth; j++ )   
        {   
            it2 = CorrInd[i][j].y;   
            jt2 = CorrInd[i][j].x;   
   
            uchar * psrcdata = cvPtr2D( src, it2, jt2 );   
   
            *(pdstdata++) = *(psrcdata++);   
            *(pdstdata++) = *(psrcdata++);   
            *(pdstdata++) = *(psrcdata++);   
        }   
    }   
}
*/
int RunTestEyeFishMap( Size sizedata , int img_index)
{
	Mat im1, D1;
	im1.create(3,3,CV_64FC1); 
	D1.create(1,4,CV_64FC1);

    double fc1,fc2,cc1,cc2,kc1,kc2,kc3,kc4;

	fc1 = 655.6;
    fc2 = 656.4;
    cc1 = 625.06;
    cc2 = 361.96;
    kc1 = -0.382;
    kc2 = 0.195;
    kc3 = -0.00363;
    kc4 = 0.00237;

	im1.at<double>(0, 0) = fc1;
	im1.at<double>(0, 1) = 0;
	im1.at<double>(0, 2) = cc1;
	im1.at<double>(1, 0) = 0;
	im1.at<double>(1, 1) = fc2;
	im1.at<double>(1, 2) = cc2;
	im1.at<double>(2, 0) = 0;
	im1.at<double>(2, 1) = 0;
	im1.at<double>(2, 2) = 1;
	D1.at<double>(0, 0) = kc1;
	D1.at<double>(0, 1) = kc2;
	D1.at<double>(0, 2) = kc3;
	D1.at<double>(0, 3) = kc4;

    initUndistortRectifyMap(im1, D1, Mat(),im1 , sizedata , CV_16SC2, mapEye1[img_index], mapEye2[img_index]);
    //remap( Img, ImgUndistort, mapEye1[img_index], mapEye2[img_index] , INTER_LINEAR );

	im1.release();
	D1.release();
	return 0;
}
/*
int RunTestEyeFish()
{
	const char* imagename = ".\\TestImg\\3.bmp";
    Mat img = imread(imagename);
	Mat result;
    if(img.empty())
    {
        fprintf(stderr, "Can not load image %s\n", imagename);
        return -1;
    }
	result = img.clone();
	IplImage* img1 = &IplImage(img);
	IplImage* Show1 = &IplImage(result);

	double *mi;
    double *md;

    mi = new double[3*3];
    md = new double[4];

    CvMat intrinsic_matrix,distortion_coeffs;

    //摄像机内参数
    cvInitMatHeader(&intrinsic_matrix,3,3,CV_64FC1,mi);
    
    //镜头畸变参数
    cvInitMatHeader(&distortion_coeffs,1,4,CV_64FC1,md);

    //参数由matlab获得
    double fc1,fc2,cc1,cc2,kc1,kc2,kc3,kc4;
    fc1 = 317.34;
    fc2 = 497.40;
    cc1 = 230.15;
    cc2 = 317.73;
    kc1 = -0.5672;
    kc2 = 0.29126;
    kc3 = -0.0248;
    kc4 = 0.00647;

    cvmSet(&intrinsic_matrix, 0, 0, fc1);
    cvmSet(&intrinsic_matrix, 0, 1, 0);
    cvmSet(&intrinsic_matrix, 0, 2, cc1);
    cvmSet(&intrinsic_matrix, 1, 0, 0);
    cvmSet(&intrinsic_matrix, 1, 1, fc2);
    cvmSet(&intrinsic_matrix, 1, 2, cc2);
    cvmSet(&intrinsic_matrix, 2, 0, 0);
    cvmSet(&intrinsic_matrix, 2, 1, 0);
    cvmSet(&intrinsic_matrix, 2, 2, 1);

    cvmSet(&distortion_coeffs, 0, 0, kc1);
    cvmSet(&distortion_coeffs, 0, 1, kc2);
    cvmSet(&distortion_coeffs, 0, 2, kc3);
    cvmSet(&distortion_coeffs, 0, 3, kc4);
   //矫正畸变(opencv)
    {
        Mat view, rview, map1, map2;
        //initUndistortRectifyMap(intrinsic_matrix, distortion_coeffs, Mat(),
        //                        getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, imageSize, 1, imageSize, 0),
        //                        imageSize, CV_16SC2, map1, map2);

        //for( i = 0; i < (int)imageList.size(); i++ )
        {
            //view = imread(imageList[i], 1);
            //if(!view.data)
            //    continue;
            ////undistort( view, rview, cameraMatrix, distCoeffs, cameraMatrix );
            remap(img, result, map1, map2, INTER_LINEAR);
            //imshow("Image View", rview);
            //int c = 0xff & waitKey();
            //if( (c & 255) == 27 || c == 'q' || c == 'Q' )
            //    break;
        }
    }

    //矫正畸变(opencv)
    for (int nx=0; nx< img.cols ; nx++)
    {
        for (int ny=0; ny< img.rows ; ny++)
        {
            double x=nx-50;
            double y=ny-50;
            double xx=(x-cc1)/fc1;
            double yy=(y-cc2)/fc2;
            double r2=pow(xx,2)+pow(yy,2);
            double r4=pow(r2,2);
            double xxx=xx*(1+kc1*r2+kc2*r4)+2*kc3*xx*yy+kc4*(r2+2*xx*xx);
            double yyy=yy*(1+kc1*r2+kc2*r4)+2*kc4*xx*yy+kc3*(r2+2*yy*yy);
            double xxxx = xxx*fc1+cc1;
            double yyyy = yyy*fc2+cc2;
            if (xxxx>0 && xxxx<320 && yyyy>0 && yyyy<240)
            {
                _I(Show1,nx,ny) = (int)_IF(img1,xxxx,yyyy);
            }
            else
            {
                _I(Show1,nx,ny) = 0;
            }

        }
	}
	//cvUndistort2
	//undistort（）;

	imwrite(".\\TestImg\\3-Fish.bmp", result);
    return 0;

}

*/

///////////////////////////////////////////////////////////////////////////


DWORD WINAPI GetPreStruct( LPVOID lpParameter )
{
    int64 app_start_time = getTickCount();
	int frmnum = 1;
    cv::setBreakOnError(true);

	// Default command line args
	vector<string> img_names;
	double work_megapix = 0.6; //如果是1会造成矫正时间过长，速度太大 	(*adjuster)(features, pairwise_matches, cameras);
	double seam_megapix = 0.1;
	double compose_megapix = -1;
	float conf_thresh = 1.0f;

	string ba_refine_mask = "xxxxx";
	bool do_wave_correct = true;
	WaveCorrectKind wave_correct = detail::WAVE_CORRECT_HORIZ;
	bool save_graph = false;
	std::string save_graph_to;
	string warp_type = "cylindrical";//"spherical"; //"cylindrical";//
	int expos_comp_type = ExposureCompensator::GAIN_BLOCKS;
	float match_conf = 0.3f;

	double work_scale = 1, seam_scale = 1, compose_scale = 1;
	bool is_work_scale_set = false; 
	bool is_seam_scale_set = false, is_compose_scale_set = false;

	LOGLN("Finding features...");
	int64 t = getTickCount();

	Ptr<FeaturesFinder> finder;
	finder = new SurfFeaturesFinder();

	vector<Mat> mask_warped1;
	vector<Mat> full_img_s(num_images); 
	vector<ImageFeatures> features(num_images);
	vector<Mat> images(num_images);
	vector<Size> full_img_sizes(num_images);
	double seam_work_aspect = 1;

	WaitForSingleObject( hMutex , INFINITE  );
	for (int i = 0; i < num_images ; ++i)
	{
		Mat iimg;
		capture[i].read( iimg );
		remap( iimg, full_img_s[i], mapEye1[i], mapEye2[i] , INTER_LINEAR );
	}
	ReleaseMutex(hMutex);

	for (int i = 0; i < num_images ; ++i)
	{	
		/////////////////////////////////////////////
		full_img_sizes[i] = full_img_s[i].size();

		if (full_img_s[i].empty())
		{
			LOGLN("Can't open image " << img_names[i]);
			return 1;
		}
		if (work_megapix < 0)
		{
			images[i] = full_img_s[i];
			work_scale = 1;
			is_work_scale_set = true;
		}
		else
		{
			if (!is_work_scale_set)
			{
				work_scale = min(1.0, sqrt(work_megapix * 1e6 / full_img_s[i].size().area()));
				is_work_scale_set = true;
			}
			resize(full_img_s[i], images[i], Size(), work_scale, work_scale);
		}
		if (!is_seam_scale_set)
		{
			seam_scale = min(1.0, sqrt(seam_megapix * 1e6 / full_img_s[i].size().area()));
			seam_work_aspect = seam_scale / work_scale;
			is_seam_scale_set = true;
		}

		(*finder)(images[i], features[i]);
		features[i].img_idx = i;
		LOGLN("Features in image #" << i+1 << ": " << features[i].keypoints.size());

		resize(full_img_s[i], images[i], Size(), seam_scale, seam_scale);
	}

	finder->collectGarbage();

	LOGLN("Finding features, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");
	LOG("Pairwise matching");
	t = getTickCount();

	vector<MatchesInfo> pairwise_matches;
	BestOf2NearestMatcher matcher( false , match_conf);
	matcher(features, pairwise_matches);
	matcher.collectGarbage();
	LOGLN("Pairwise matching, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");

	// Check if we should save matches graph
	if (save_graph)
	{
		LOGLN("Saving matches graph...");
		ofstream f(save_graph_to.c_str());
		f << matchesGraphAsString(img_names, pairwise_matches, conf_thresh);
	}

	HomographyBasedEstimator estimator;
	vector<CameraParams> cameras;
	estimator(features, pairwise_matches, cameras);
	for (size_t i = 0; i < cameras.size(); ++i)
	{
		Mat R;
		cameras[i].R.convertTo(R, CV_32F);
		cameras[i].R = R;
	}

	Ptr<detail::BundleAdjusterBase> adjuster = new detail::BundleAdjusterRay();
	adjuster->setConfThresh(conf_thresh);
	Mat_<uchar> refine_mask = Mat::zeros(3, 3, CV_8U);
	if (ba_refine_mask[0] == 'x') refine_mask(0,0) = 1;
	if (ba_refine_mask[1] == 'x') refine_mask(0,1) = 1;
	if (ba_refine_mask[2] == 'x') refine_mask(0,2) = 1;
	if (ba_refine_mask[3] == 'x') refine_mask(1,1) = 1;
	if (ba_refine_mask[4] == 'x') refine_mask(1,2) = 1;

	adjuster->setRefinementMask(refine_mask);
	(*adjuster)(features, pairwise_matches, cameras);

	// Find median focal length
	vector<double> focals;
	for (size_t i = 0; i < cameras.size(); ++i)
	{
		focals.push_back(cameras[i].focal);
	}
	sort(focals.begin(), focals.end());

	float warped_image_scale;
	if (focals.size() % 2 == 1)
		warped_image_scale = static_cast<float>(focals[focals.size() / 2]);
	else
		warped_image_scale = static_cast<float>(focals[focals.size() / 2 - 1] + focals[focals.size() / 2]) * 0.5f;
	// warve correct 
	if (do_wave_correct)
	{
		vector<Mat> rmats;
		for (size_t i = 0; i < cameras.size(); ++i)
			rmats.push_back(cameras[i].R);
		waveCorrect(rmats, wave_correct);
		for (size_t i = 0; i < cameras.size(); ++i)
			cameras[i].R = rmats[i];
	}

	LOGLN("Warping images (auxiliary)... ");
	t = getTickCount();

	vector<Mat> masks_warped(num_images);
	vector<Mat> images_warped(num_images);
	vector<Size> sizes(num_images);
	vector<Mat> masks(num_images);

	// Preapre images masks
	for (int i = 0; i < num_images; ++i)
	{
		masks[i].create(images[i].size(), CV_8U);
		masks[i].setTo(Scalar::all(255));
	}

	// Warp images and their masks
	Ptr<WarperCreator> warper_creator;
	if(warp_type == "cylindrical") warper_creator = new cv::CylindricalWarper();
	else if (warp_type == "spherical") warper_creator = new cv::SphericalWarper();
    
	if (warper_creator.empty())
	{
		cout << "Can't create the following warper '" << warp_type << "'\n";
		return 1;
	}

	Ptr<RotationWarper> warper = warper_creator->create(static_cast<float>(warped_image_scale * seam_work_aspect));

	for (int i = 0; i < num_images; ++i)
	{
		Mat_<float> K;
		cameras[i].K().convertTo(K, CV_32F);
		float swa = (float)seam_work_aspect;
		K(0,0) *= swa; K(0,2) *= swa;
		K(1,1) *= swa; K(1,2) *= swa;
		corners[i] = warper->warp(images[i], K, cameras[i].R, INTER_LINEAR, BORDER_REFLECT, images_warped[i]);
		sizes[i] = images_warped[i].size();

		warper->warp(masks[i], K, cameras[i].R, INTER_NEAREST, BORDER_CONSTANT, masks_warped[i]);
	}

	vector<Mat> images_warped_f(num_images);
	for (int i = 0; i < num_images; ++i)
		images_warped[i].convertTo(images_warped_f[i], CV_32F);

	LOGLN("Warping images, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");

	Ptr<ExposureCompensator> compensator = ExposureCompensator::createDefault(expos_comp_type);
	compensator->feed(corners, images_warped, masks_warped);

	Ptr<SeamFinder> seam_finder;
	seam_finder = new detail::GraphCutSeamFinder(GraphCutSeamFinderBase::COST_COLOR);
	if (seam_finder.empty())
	{
		cout << "Can't create the following seam finder " << "'\n";
		return 1;
	}

	seam_finder->find(images_warped_f, corners, masks_warped);

	// Release unused memory
	images.clear();
	images_warped.clear();
	images_warped_f.clear();
	masks.clear();

	LOGLN("Compositing...");
	t = getTickCount();

	Mat img;
	Mat img_warped, img_warped_s;
	Mat dilated_mask, seam_mask, mask, mask_warped;
	double compose_work_aspect = 1;

	vector<Mat> Kcamera;

	for (int img_idx = 0; img_idx < num_images; ++img_idx)
	{
		if (!is_compose_scale_set)
		{
			if (compose_megapix > 0)
				compose_scale = min(1.0, sqrt(compose_megapix * 1e6 / full_img_s[img_idx].size().area()));
			is_compose_scale_set = true;
			// Compute relative scales
			//compose_seam_aspect = compose_scale / seam_scale;
			compose_work_aspect = compose_scale / work_scale;

			// Update warped image scale
			warped_image_scale *= static_cast<float>(compose_work_aspect);
			warper = warper_creator->create(warped_image_scale);

			// Update corners and sizes
			for (int i = 0; i < num_images; ++i)
			{
				// Update intrinsics
				cameras[i].focal *= compose_work_aspect;
				cameras[i].ppx *= compose_work_aspect;
				cameras[i].ppy *= compose_work_aspect;
				// Update corner and size
				Size sz = full_img_sizes[i];
				if (std::abs(compose_scale - 1) > 1e-1)
				{
					sz.width = cvRound(full_img_sizes[i].width * compose_scale);
					sz.height = cvRound(full_img_sizes[i].height * compose_scale);
				}

				Mat K;
				cameras[i].K().convertTo(K, CV_32F);
				Rect roi = warper->warpRoi(sz, K, cameras[i].R);
				corners[i] = roi.tl();
				sizes[i] = roi.size();
			}
		}
		if (abs(compose_scale - 1) > 1e-1)
			resize(full_img_s[img_idx], img, Size(), compose_scale, compose_scale);
		else
			img = full_img_s[img_idx];
		srctype = img.type();
		full_img_s[img_idx].release();
		Size img_size = img.size();
		Mat Kt;
		cameras[img_idx].K().convertTo( Kt, CV_32F);
		Kcamera.push_back(Kt);

		// Warp the current image mask
		mask.create(img_size, CV_8U);
		mask.setTo(Scalar::all(255));
		warpsize( warper , mask, Kcamera.at(img_idx), cameras[img_idx].R, INTER_NEAREST, BORDER_CONSTANT, mask_warped  , xmap1[img_idx], ymap1[img_idx]);
		
		// Warp the current image	
		dst_roi[img_idx] = warpsize( warper , img,  Kcamera.at(img_idx), cameras[img_idx].R, INTER_LINEAR, BORDER_REFLECT, img_warped , xmap1[img_idx], ymap1[img_idx] );
		img.release();
		// Compensate exposure
		compensator->apply(img_idx, corners[img_idx], img_warped, mask_warped);
		img_warped.convertTo(img_warped_s, CV_16S);
		img_warped.release();
		mask.release();

		//img_warped_s  mask_warped
		dilate(masks_warped[img_idx], dilated_mask, Mat());
		resize(dilated_mask, seam_mask, mask_warped.size());
		mask_warped = seam_mask & mask_warped;
				
		mask_warped1.push_back(mask_warped);
		if( img_idx == 0 )
			prepareSize( resultRoiSize(corners, sizes) ,   result1, result_mask1 , dst_roi_1  );
		// Blend the current image
		feedSize(img_warped_s, mask_warped, corners[img_idx] , dst_roi_1 , result1, result_mask1 ,img_idx );
	}

	Mat result2, result_mask2;
	blendSize(result2, result_mask2 ,  result1, result_mask1   );
	LOGLN("Compositing, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");
	LOGLN("Finished, total time: " << ((getTickCount() - app_start_time) / getTickFrequency()) << " sec");		
	imwrite("prestitch.jpg", result1);
	return 0;
}

int UpdateMat()
{
	ResultStitch.create(dst_roi_1.size(), CV_16SC3);
	ResultStitch.setTo(Scalar::all(0));
	Result_mask_stitch = result_mask1.clone();
	for(int i = 0 ; i < num_images ; i++ )
	{
		xmapstitch[i] = xmap1[i].clone();
		ymapstitch[i] = ymap1[i].clone();
		img4[i].create(dst_roi[i].height + 1, dst_roi[i].width + 1, srctype );
	}
	xmapdetstitch = xmapdet.clone();
	ymapdetstitch = ymapdet.clone();
	indxmapstitch = indxmap.clone();
	return 0;
}

DWORD WINAPI StitchingAll( LPVOID lpParameter )
{
	float ave = 0;
	float t=0,t1=0;
	vector<Mat> img3(num_images);
	vector<Mat> img5(num_images);

	int frmnum = 1;
	int meg;
	for(  frmnum = 2 ; frmnum < 30*0.5 ;   frmnum ++ )
	{
        RecvMessage( meg , STITCH );
		if (meg == UPDATE) 
        {
            //update 矩阵参数
            UpdateMat();
            SendMessage( CONTUN  , STITCH );
            SendMessage( CONTUN  , PRESTI );
        }

		WaitForSingleObject( hMutex , INFINITE  );
		for (int i = 0; i < num_images ; ++i)
		{
			//capture[i].set(CV_CAP_PROP_POS_MSEC , frmnum * 100 ); //每1秒读取一次
			Mat iimg;
			capture[i].read( iimg  );
			remap( iimg, img3[i], mapEye1[i], mapEye2[i] , INTER_LINEAR );
			if (img3[i].data == NULL )
					return -1;
		}
		ReleaseMutex(hMutex);
		
		for (int img_idx = 0; img_idx < num_images ; ++img_idx)
		{
			t = getTickCount();
			Size img_size = img3[img_idx].size();
			//LOGLN("Read Frame, time:  idx " << frmnum << " " << ((getTickCount() - t) / getTickFrequency()) << " sec");

			t = getTickCount();
			remap(img3[img_idx], img4[img_idx],  xmapstitch[img_idx], ymapstitch[img_idx],  INTER_LINEAR, BORDER_REFLECT );		
			img4[img_idx].convertTo( img5[img_idx]  , CV_16S  );	
			t1 += ((getTickCount() - t) / getTickFrequency()) ;	

		}

		t = getTickCount();
		feedSizeRemap( img5 ,  ResultStitch , Result_mask_stitch  );
		ave += ((getTickCount() - t) / getTickFrequency()) ;
		sprintf(outbuf, ".\\output\\img-%d.jpg", frmnum );
		imwrite(outbuf, ResultStitch);
	}

	LOGLN("Stitching ave Frame, time: "  << ave/frmnum << " sec"<< "\n"<< "'\n");
	LOGLN("Stitching t1 Frame, time: "  << t1/frmnum << " sec"<< "\n"<< "'\n");
	LOGLN("Stitching ave + t1 Frame, time: "  << (ave + t1)/frmnum << " sec"<< "\n"<< "'\n");
	return 0;
}


DWORD WINAPI GetPreStructw( LPVOID lpParameter )
{	
	int ret = 0;
    int meg = 0;
	while(1)
	{
        RecvMessage( meg , PRESTI );
        if( meg == ENDOUT)
        {
            break;
        }
        else if (meg == CONTUN) 
        {
			loopNum++;
			LOGLN("Stitching Statistic Per is : "  << loopNum << "\n");
    		ret = GetPreStruct( lpParameter );
    		if( ( ret == 0 )&&(gEventSting == 0))
    		{
    			LOGLN("Stitching PreStich is finished: "  << ret << "\n");
                SendMessage( UPDATE  , STITCH );
                SendMessage( WAITIN  , PRESTI );
    		}
        } 
        else 
        {
            Sleep( 30000);
        }

	}
	result1.release();
	result_mask1.release();
	gEventPre = 1; //set end message
	return 0;
}

DWORD WINAPI StitchingAllw( LPVOID lpParameter )
{
	int ret = 0;  
    int meg;
	while(1) 
	{
        RecvMessage( meg , STITCH );
        if (meg == CONTUN) 
        {            
    		ret = StitchingAll( lpParameter );	
    		if ( ret == 0 )
			{
				SendMessage( ENDOUT  , PRESTI );
    			break;
			}
			else if( ret == -1 )
    			LOGLN("Stitching is error : "  << ret << "\n");
        }
        else if (meg == UPDATE) 
        {
            //update 矩阵参数
            UpdateMat();
            SendMessage( CONTUN  , STITCH );
            SendMessage( CONTUN  , PRESTI );
        }
        else // meg == CONTUN
        {
            Sleep(2000);
        }
	}
    SendMessage( ENDOUT  , PRESTI );
	gEventSting = 1; //set end message 
	return 0;
}

int OpenVideo()
{
	vector<string> filename;
#if 0
	filename.push_back(".\\TestVideo\\1.mp4");
	filename.push_back(".\\TestVideo\\2.mp4");
	filename.push_back(".\\TestVideo\\3.mp4");
	filename.push_back(".\\TestVideo\\4.mp4");
	filename.push_back(".\\TestVideo\\5.mp4");
	filename.push_back(".\\TestVideo\\6.mp4");
#else
	filename.push_back(".\\TestNew\\1.mp4");
	filename.push_back(".\\TestNew\\2.mp4");
	filename.push_back(".\\TestNew\\3.mp4");
	filename.push_back(".\\TestNew\\4.mp4");

#endif
	for( int i = 0 ; i < num_images ; i++ )
	{
		capture[i].open( filename[i] );	
	}
	if(!capture[0].isOpened())
	{
		cout<<"fail to open video file!"<<endl;
		return -1;
	}

	Mat temp;
	for( int i = 0 ; i < num_images ; i++ )
	{
		capture[i].set(CV_CAP_PROP_POS_MSEC ,  1200 );
		capture[i].read( temp );
		RunTestEyeFishMap( temp.size()  , i);  //筒形畸形矫正
	}
	return 0;
}

int main(int argc, char* argv[])
{

	LPVOID lpData = NULL;

	//RunTestEyeFish1();

	hMutex = CreateMutex( NULL , TRUE , NULL ); //读取文件保护
	if ( OpenVideo() != 0 )
	{	
		return 0;
	}
	ReleaseMutex(hMutex);

	//hMutexRemap = CreateMutex( NULL , FALSE , NULL ); //读取文件保护
	hMutexMessg = CreateMutex( NULL , FALSE , NULL ); //读取文件保护

	HANDLE handlePreStruct = CreateThread( NULL , 0 ,  GetPreStructw , NULL , 0 , NULL ); 
	HANDLE handleStitching = CreateThread( NULL , 0 ,  StitchingAllw , NULL , 0 , NULL ); 

	CloseHandle(handlePreStruct);
	CloseHandle(handleStitching);

	//gEventSting =1;
	while( ( gEventPre == 0 ) )
	{
		Sleep(2000);
	}

	LOGLN("Stitching Statistic is : "  << loopNum << "\n");
	return 0;
}





