/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//
//M*/

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
#include  <process.h> 
#define NUM 6

using namespace std;
using namespace cv;
using namespace cv::detail;

// Default command line args
vector<string> img_names;
double work_megapix = -1;
double seam_megapix = 0.1;
double compose_megapix = -1;
float conf_thresh = 1.0f;
int num_images = 6 ;// camera numbers 
string ba_refine_mask = "xxxxx";
bool do_wave_correct = true;
WaveCorrectKind wave_correct = detail::WAVE_CORRECT_HORIZ;
bool save_graph = false;
std::string save_graph_to;
string warp_type = "cylindrical";//"spherical"; //"cylindrical";//
int expos_comp_type = ExposureCompensator::GAIN_BLOCKS;
float match_conf = 0.3f;
int blend_type = Blender::MULTI_BAND;


//排序图片 从1到n 必须从做到有 1 到 N 

char outbuf[50];
char testfile[50];
//////////////////////////////////////////////////////////////////////////////////
//第一图像宽高和img mask
vector<VideoCapture> capture(num_images);
vector<Mat>	xmap1(num_images), ymap1(num_images),mask_warped1; 
Mat result1, result_mask1; 

vector<Rect> dst_roi(num_images);
vector<Point> corners(num_images);

Mat  xmapdet, ymapdet,indxmap;
Rect dst_roi_1 ;

vector<Mat> img3(num_images);
vector<Mat> img4(num_images);
vector<Mat> img5(num_images);
vector<Mat> img6(num_images);
/////////////////////////////////////////////////////////////////////////////
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
	//CV_Assert(dst_.depth() != sizeof(uchar));  
    for (int y = 0; y < dst_.rows ; ++y)  
    {
        Point3_<short> *dst_row = dst_.ptr<Point3_<short> >( y );
        uchar *dst_mask_row = dst_mask_.ptr<uchar>( y );

		float *dst_x = xmapdet.ptr<float>( y );
		float *dst_y = ymapdet.ptr<float>( y );
		float *dst_z = indxmap.ptr<float>( y );

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
int warpsizeT( RotationWarper * warper  , Mat &src,  Mat &K, const Mat &R, int interp_mode, int border_mode,
                                  Mat &dst , Mat &xmap, Mat &ymap )
{
    Rect dst_roi = warper->buildMaps(src.size(), K, R, xmap, ymap);
    dst.create(dst_roi.height + 1, dst_roi.width + 1, src.type());
    //remap(src, dst, xmap, ymap, interp_mode, border_mode);
	LOGLN("src.type() " << src.type()<< ".."<<"\n");	

    for (int y = 0; y < dst.rows; ++y )
    {
        Point3_<short> *dst_row = dst.ptr<Point3_<short> >( y );
		float *dst_y = ymap.ptr<float>( y );
		float *dst_x = xmap.ptr<float>( y );
        for (int x = 0; x < dst.cols; ++x )
        {
			int xx = dst_x[ x ];
			int yy = dst_y[ x ];
			LOGLN("src map " << xx << ".."<< yy << "\n");	
			LOGLN("Row map " << dst.rows << ".."<< dst.cols <<"\n");	
			dst_row[ x ] = src.at< Point3_<short> >( dst.cols/2 - yy , dst.rows / 2 - xx  ) ;
        }
    }
    return 0;
}
*/

/*
inline void AdjustWidth(cv::Mat& src)
{
	if (src.cols*src.channels()%4 != 0)
	{
		int right = (src.cols+3)/4*4 - src.cols;
		copyMakeBorder(src,src,0,0,0,right,BORDER_REPLICATE);
	}
}

int FishEyeRectify(const cv::Mat& src,cv::Mat& dst,int z)
{
	if (src.empty() ||
		(src.data == dst.data))
	{
		return -1;
	}

	int rd = (int)sqrt(double(src.cols*src.cols + src.rows*src.rows)) / 2;
	double tempq = tan(double(rd*1.0/z));
	int temp =(int)( 2 * z * tempq);
	int nWidth =(int)( temp * src.cols / 2.0 / rd);
	int nHeight =(int)( temp * src.rows / 2.0 / rd);

	if (nHeight > 10000 || nWidth > 10000)
	{
		return -1;
	}

	const int nHalfWidthDst = nWidth / 2;
	const int nHalfHeightDst = nHeight / 2;
	const int nHalfWidthSrc = src.cols / 2;
	const int nHalfHeightSrc = src.rows / 2;

	dst = Mat::zeros(nHeight,nWidth,src.type());

	for (int v=0; v<nHalfHeightDst; ++v)
	{
		for (int u=0; u<nHalfWidthDst; ++u)
		{
			double ru = sqrt(double(u*u+v*v));
			double temp = z * atan(ru / z) / ru;
			int x =(int)( u * temp);
			int y =(int)( v * temp);
			Set4By1(src,dst,Point(x,y),Point(u,v),
				nHalfWidthSrc,nHalfHeightSrc,
				nHalfWidthDst,nHalfHeightDst);

		}
	}

	AdjustWidth(dst);
	return 0;

}
*/


int GetStruct( )
{
    int64 app_start_time = getTickCount();
	int frmnum = 1;
    //cv::setBreakOnError(true);

	vector<string> filename;
	filename.push_back(".\\TestVideo\\1.mp4");
	filename.push_back(".\\TestVideo\\2.mp4");
	filename.push_back(".\\TestVideo\\3.mp4");
	filename.push_back(".\\TestVideo\\4.mp4");
	filename.push_back(".\\TestVideo\\5.mp4");
	filename.push_back(".\\TestVideo\\6.mp4");


	for( int i = 0 ; i < num_images ; i++ )
	{
		capture[i].open( filename[i] );	
	}
	if(!capture[0].isOpened())
	{
		cout<<"fail to open video file!"<<endl;
		return 0;
	}

	for( int i = 0 ; i < num_images ; i++ )
	{
		capture[i].set(CV_CAP_PROP_POS_MSEC ,  1000 );
	}
	double work_scale = 1, seam_scale = 1, compose_scale = 1;
	bool is_work_scale_set = false; 
	bool is_seam_scale_set = false, is_compose_scale_set = false;

	LOGLN("Finding features...");
	int64 t = getTickCount();

	Ptr<FeaturesFinder> finder;
	finder = new SurfFeaturesFinder();

	//Mat full_img,img, first_img;
	vector<Mat> full_img_s(num_images);//
	vector<ImageFeatures> features(num_images);
	vector<Mat> images(num_images);
	vector<Size> full_img_sizes(num_images);
	double seam_work_aspect = 1;

	for (int i = 0; i < num_images ; ++i)
	{			
		capture[i].read( full_img_s[i] );
		full_img_sizes[i] = full_img_s[i].size();

		if (full_img_s[i].empty())
		{
			LOGLN("Can't open image " << img_names[i]);
			return -1;
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

	//read next frame
	sprintf(outbuf, ".\\output\\img-%d.jpg", frmnum );
	for( int i = 0 ; i < num_images ; i++ )
	{
		capture[i].set(CV_CAP_PROP_POS_MSEC , frmnum * 1000 );
	}
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
		full_img_s[img_idx].release();//
		Size img_size = img.size();
		Mat Kt;
		cameras[img_idx].K().convertTo( Kt, CV_32F);
		Kcamera.push_back(Kt);

		// Warp the current image mask
		mask.create(img_size, CV_8U);
		mask.setTo(Scalar::all(255));


		warpsize( warper , mask, Kcamera.at(img_idx), cameras[img_idx].R, INTER_NEAREST, BORDER_CONSTANT, mask_warped  , xmap1[img_idx], ymap1[img_idx]);
		
		//sprintf(testfile, ".\\output1\\img-%d.jpg", img_idx );
		//imwrite(testfile, mask_warped);
		
		// Warp the current image	
		dst_roi[img_idx] = warpsize( warper , img,  Kcamera.at(img_idx), cameras[img_idx].R, INTER_LINEAR, BORDER_REFLECT, img_warped , xmap1[img_idx], ymap1[img_idx] );
		
		img4[img_idx].create(dst_roi[img_idx].height + 1, dst_roi[img_idx].width + 1, img.type());

		//sprintf(testfile, ".\\output1\\warp-%d.jpg", img_idx );
		//imwrite(testfile, img_warped);

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
		
		//sprintf(testfile, ".\\output1\\warped-%d.jpg", img_idx );
		//imwrite(testfile, mask_warped);					
		mask_warped1.push_back(mask_warped);

		if( img_idx == 0 )
			prepareSize( resultRoiSize(corners, sizes) ,   result1, result_mask1 , dst_roi_1  );
		// Blend the current image
		feedSize(img_warped_s, mask_warped, corners[img_idx] , dst_roi_1 , result1, result_mask1 ,img_idx );

	}

	Mat result2, result_mask2;
	blendSize(result2, result_mask2 ,  result1, result_mask1   );
	LOGLN("Compositing, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");
	imwrite(outbuf, result2);
	LOGLN("Finished, total time: " << ((getTickCount() - app_start_time) / getTickFrequency()) << " sec");	
	return 0;
}

int StitchingAll()
{
	int64 ave = 0;
	int64 t=0,t1=0;
	int frmnum = 1;
	for(  frmnum = 2 ; frmnum < 60 ;   frmnum ++ )
	{
		t = getTickCount();
		sprintf(outbuf, ".\\output\\img-%d.jpg", frmnum );
		for( int i = 0 ; i < num_images ; i++ )
		{
			capture[i].set(CV_CAP_PROP_POS_MSEC , frmnum * 1000 );
		}

		for (int img_idx = 0; img_idx < num_images ; ++img_idx)
		{
			t = getTickCount();
			capture[img_idx].read( img3[img_idx] );
			Size img_size = img3[img_idx].size();
			LOGLN("Read Frame, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");

			t = getTickCount();
			//warpsizeT( warper ,  img4[img_idx],  K[img_idx], cameras[img_idx].R, INTER_LINEAR, BORDER_REFLECT, img5[img_idx] , xmap1[img_idx], ymap1[img_idx] );
			remap(img3[img_idx], img4[img_idx],  xmap1[img_idx], ymap1[img_idx],  INTER_LINEAR, BORDER_REFLECT );		
			img4[img_idx].convertTo( img5[img_idx]  , CV_16S  );	
			t1 += ((getTickCount() - t) / getTickFrequency()) ;	

			//sprintf(testfile, ".\\output1\\re-img-%d.jpg", img_idx );
			//imwrite(testfile, img5[img_idx]  );	
		}

		t = getTickCount();
		feedSizeRemap( img5 ,  result1 , result_mask1  );
		ave += ((getTickCount() - t) / getTickFrequency()) ;
		imwrite(outbuf, result1);
	}

	LOGLN("Stitching ave Frame, time: "  << ave/58 << " sec"<< "\n"<< "'\n");
	LOGLN("Stitching t1 Frame, time: "  << t1/58 << " sec"<< "\n"<< "'\n");
	LOGLN("Stitching ave + t1 Frame, time: "  << (ave + t1)/58 << " sec"<< "\n"<< "'\n");

	result1.release();
	result_mask1.release();
	return 0;
}

int main(int argc, char* argv[])
{
	GetStruct( );
	StitchingAll( );
/*
    pthread_t t1,t2;
    pthread_create(&t1,NULL,GetStruct,(void *)"hello,");
    pthread_create(&t2,NULL,StitchingAll,(void *)"world!\n");
   
    pthread_join(t1,NULL);
    pthread_join(t2,NULL);
*/
	return 0;
}

