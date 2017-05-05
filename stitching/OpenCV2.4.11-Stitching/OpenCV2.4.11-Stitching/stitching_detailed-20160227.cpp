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

using namespace std;
using namespace cv;
using namespace cv::detail;

// Default command line args
vector<string> img_names;
//bool preview = false;
//bool try_gpu = false;
double work_megapix = -1;
double seam_megapix = 0.1;
double compose_megapix = -1;
float conf_thresh = 1.0f;
//string features_type = "surf";
//string ba_cost_func = "ray";
string ba_refine_mask = "xxxxx";
bool do_wave_correct = true;
WaveCorrectKind wave_correct = detail::WAVE_CORRECT_HORIZ;
bool save_graph = false;
std::string save_graph_to;
string warp_type = "cylindrical";//"spherical"; //"cylindrical";//
int expos_comp_type = ExposureCompensator::GAIN_BLOCKS;
float match_conf = 0.3f;
//string seam_find_type = "gc_color";
int blend_type = Blender::MULTI_BAND;
//float blend_strength = 5;
//string result_name = "result.jpg";

static int parseCmdArgs(int argc, char** argv)
{
    if (argc == 1)
    {
        //printUsage();
        return -1;
    }
    for (int i = 1; i < argc; ++i)
    {
        if (string(argv[i]) == "--work_megapix")
        {
            work_megapix = atof(argv[i + 1]);
            i++;
        }
        else if (string(argv[i]) == "--seam_megapix")
        {
            seam_megapix = atof(argv[i + 1]);
            i++;
        }
        else if (string(argv[i]) == "--compose_megapix")
        {
            compose_megapix = atof(argv[i + 1]);
            i++;
        }
        else if (string(argv[i]) == "--match_conf")
        {
            match_conf = static_cast<float>(atof(argv[i + 1]));
            i++;
        }
        else if (string(argv[i]) == "--conf_thresh")
        {
            conf_thresh = static_cast<float>(atof(argv[i + 1]));
            i++;
        }
        else if (string(argv[i]) == "--wave_correct")
        {
            if (string(argv[i + 1]) == "no")
                do_wave_correct = false;
            else if (string(argv[i + 1]) == "horiz")
            {
                do_wave_correct = true;
                wave_correct = detail::WAVE_CORRECT_HORIZ;
            }
            else if (string(argv[i + 1]) == "vert")
            {
                do_wave_correct = true;
                wave_correct = detail::WAVE_CORRECT_VERT;
            }
            else
            {
                cout << "Bad --wave_correct flag value\n";
                return -1;
            }
            i++;
        }
        else if (string(argv[i]) == "--save_graph")
        {
            save_graph = true;
            save_graph_to = argv[i + 1];
            i++;
        }
        else
            img_names.push_back(argv[i]);
    }
    return 0;
}

typedef unsigned int uint;

void Set4By1(const cv::Mat& src,cv::Mat& dst,
	cv::Point srcPt,cv::Point dstPt,
	uint nHalfWidthSrc,uint nHalfHeightSrc,
	uint nHalfWidthDst,uint nHalfHeightDst)
{
	uint u = srcPt.x;
	uint v = srcPt.y;
	uint x = dstPt.x;
	uint y = dstPt.y;
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

	for (uint i=0; i<4; ++i)
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
}

void feedSize(const Mat &img, const Mat &mask, Point tl , Rect &dst_roi_ ,Mat &dst_ , Mat &dst_mask_  )
{
    CV_Assert(img.type() == CV_16SC3);
    CV_Assert(mask.type() == CV_8U);
    int dx = tl.x - dst_roi_.x;
    int dy = tl.y - dst_roi_.y;

    for (int y = 0; y < img.rows; ++y)
    {
        const Point3_<short> *src_row = img.ptr<Point3_<short> >(y);
        Point3_<short> *dst_row = dst_.ptr<Point3_<short> >(dy + y);
        const uchar *mask_row = mask.ptr<uchar>(y);
        uchar *dst_mask_row = dst_mask_.ptr<uchar>(dy + y);

        for (int x = 0; x < img.cols; ++x)
        {
            if (mask_row[x])
                dst_row[dx + x] = src_row[x];
            dst_mask_row[dx + x] |= mask_row[x];
        }
    }
}

void blendSize(Mat &dst, Mat &dst_mask , Mat &dst_ , Mat &dst_mask_ )
{
    dst_.setTo(Scalar::all(0), dst_mask_ == 0);
    dst = dst_;
    dst_mask = dst_mask_;
    dst_.release();
    dst_mask_.release();
}

Point warpsize( RotationWarper * warper  ,const Mat &src, const Mat &K, const Mat &R, int interp_mode, int border_mode,
                                  Mat &dst ,     Mat &xmap, Mat &ymap )
{

    Rect dst_roi = warper->buildMaps(src.size(), K, R, xmap, ymap);

    dst.create(dst_roi.height + 1, dst_roi.width + 1, src.type());
    remap(src, dst, xmap, ymap, interp_mode, border_mode);

    return dst_roi.tl();
}

inline void AdjustWidth(cv::Mat& src)
{
	if (src.cols*src.channels()%4 != 0)
	{
		uint right = (src.cols+3)/4*4 - src.cols;
		copyMakeBorder(src,src,0,0,0,right,BORDER_REPLICATE);
	}
}

int FishEyeRectify(const cv::Mat& src,cv::Mat& dst,uint z)
{
	if (src.empty() ||
		(src.data == dst.data))
	{
		return -1;
	}

	uint rd = sqrt(double(src.cols*src.cols + src.rows*src.rows)) / 2;
	double tempq = tan(double(rd*1.0/z));
	uint temp = 2 * z * tempq;
	uint nWidth = temp * src.cols / 2.0 / rd;
	uint nHeight = temp * src.rows / 2.0 / rd;

	if (nHeight > 10000 || nWidth > 10000)
	{
		return -1;
	}

	const uint nHalfWidthDst = nWidth / 2;
	const uint nHalfHeightDst = nHeight / 2;
	const uint nHalfWidthSrc = src.cols / 2;
	const uint nHalfHeightSrc = src.rows / 2;

	dst = Mat::zeros(nHeight,nWidth,src.type());

	for (uint v=0; v<nHalfHeightDst; ++v)
	{
		for (uint u=0; u<nHalfWidthDst; ++u)
		{
			double ru = sqrt(double(u*u+v*v));
			double temp = z * atan(ru / z) / ru;
			uint x = u * temp;
			uint y = v * temp;
	/*		double num = z * atan(double(rd*1.0/z));
			uint x = u * rd / num;
			uint y = v * rd / num;*/

			Set4By1(src,dst,Point(x,y),Point(u,v),
				nHalfWidthSrc,nHalfHeightSrc,
				nHalfWidthDst,nHalfHeightDst);

		}
	}

	AdjustWidth(dst);
	return 0;

}

//排序图片 从1到n 必须从做到有 1 到 N 

int main(int argc, char* argv[])
{
    int64 app_start_time = getTickCount();
    cv::setBreakOnError(true);
    int retval = parseCmdArgs(argc, argv);
    if (retval)
        return retval;
    // Check if have enough images
    int num_images = static_cast<int>(img_names.size());
    if (num_images < 2)
    {
        LOGLN("Need more images");
        return -1;
    }

	vector<string> filename;
	filename.push_back(".\\TestVideo\\1.mp4");
	filename.push_back(".\\TestVideo\\2.mp4");
	filename.push_back(".\\TestVideo\\3.mp4");
	filename.push_back(".\\TestVideo\\4.mp4");
	filename.push_back(".\\TestVideo\\5.mp4");
	filename.push_back(".\\TestVideo\\6.mp4");

	vector<VideoCapture> capture(6);
	capture[0].open( filename[0] );
	capture[1].open( filename[1] );
	capture[2].open( filename[2] );
	capture[3].open( filename[3] );	
	capture[4].open( filename[4] );	
	capture[5].open( filename[5] );
	if(!capture[0].isOpened())
	{
		cout<<"fail to open!"<<endl;
		return 0;
	}
	char outbuf[20];
	string ss;
	int frmnum = 1;
	for(  frmnum = 1 ; frmnum < 15  ;   frmnum ++ )
	{

		double work_scale = 1, seam_scale = 1, compose_scale = 1;
		bool is_work_scale_set = false; 
		bool is_seam_scale_set = false, is_compose_scale_set = false;

		LOGLN("Finding features...");
		int64 t = getTickCount();

		Ptr<FeaturesFinder> finder;
		finder = new SurfFeaturesFinder();

		Mat full_img,first_img, img;
		vector<Mat> full_img_s(num_images);//
		vector<ImageFeatures> features(num_images);
		vector<Mat> images(num_images);
		vector<Size> full_img_sizes(num_images);
		double seam_work_aspect = 1;


		sprintf(outbuf, ".\\output\\img-%d.jpg", frmnum );
		capture[0].set(CV_CAP_PROP_POS_MSEC , frmnum * 1000 );
		capture[1].set(CV_CAP_PROP_POS_MSEC , frmnum * 1000 );
		capture[2].set(CV_CAP_PROP_POS_MSEC , frmnum * 1000 );
		capture[3].set(CV_CAP_PROP_POS_MSEC , frmnum * 1000 );
		capture[4].set(CV_CAP_PROP_POS_MSEC , frmnum * 1000 );
		capture[5].set(CV_CAP_PROP_POS_MSEC , frmnum * 1000 );

		//for (int i = 0; i < num_images; ++i)
		for (int i = 0; i < 6 ; ++i)
		{
			
			capture[i].read( full_img );
		   // full_img = imread(img_names[i]);
		
			//namedWindow("MyPicture");
			// imshow("MyPicture",full_img);
			// waitKey(10);
		
			full_img_sizes[i] = full_img.size();
			full_img_s[i] = full_img.clone();//

			if (full_img.empty())
			{
				LOGLN("Can't open image " << img_names[i]);
				return -1;
			}
			if (work_megapix < 0)
			{
				img = full_img;
				work_scale = 1;
				is_work_scale_set = true;
			}
			else
			{
				if (!is_work_scale_set)
				{
					work_scale = min(1.0, sqrt(work_megapix * 1e6 / full_img.size().area()));
					is_work_scale_set = true;
				}
				resize(full_img, img, Size(), work_scale, work_scale);
			}
			if (!is_seam_scale_set)
			{
				seam_scale = min(1.0, sqrt(seam_megapix * 1e6 / full_img.size().area()));
				seam_work_aspect = seam_scale / work_scale;
				is_seam_scale_set = true;
			}

			(*finder)(img, features[i]);
			features[i].img_idx = i;
			LOGLN("Features in image #" << i+1 << ": " << features[i].keypoints.size());

			resize(full_img, img, Size(), seam_scale, seam_scale);
			images[i] = img.clone();
		}

		finder->collectGarbage();
		full_img.release();
		img.release();

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

		// Check if we still have enough images
		num_images = static_cast<int>(img_names.size());
		if (num_images < 2)
		{
			LOGLN("Need more images");
			return -1;
		}

		HomographyBasedEstimator estimator;
		vector<CameraParams> cameras;
		estimator(features, pairwise_matches, cameras);

		for (size_t i = 0; i < cameras.size(); ++i)
		{
			Mat R;
			cameras[i].R.convertTo(R, CV_32F);
			cameras[i].R = R;
			//LOGLN("Initial intrinsics #" << indices[i]+1 << ":\n" << cameras[i].K());
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
		   // LOGLN("Camera #" << indices[i]+1 << ":\n" << cameras[i].K());
			focals.push_back(cameras[i].focal);
		}

		sort(focals.begin(), focals.end());
		float warped_image_scale;
		if (focals.size() % 2 == 1)
			warped_image_scale = static_cast<float>(focals[focals.size() / 2]);
		else
			warped_image_scale = static_cast<float>(focals[focals.size() / 2 - 1] + focals[focals.size() / 2]) * 0.5f;

	// warve correct 
		if (0)
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

		vector<Point> corners(num_images);
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

		Mat img_warped, img_warped_s;
		Mat dilated_mask, seam_mask, mask, mask_warped;
		Ptr<Blender> blender;
		double compose_work_aspect = 1;

		Mat result1, result_mask1, xmap1, ymap1; 
		Rect dst_roi_1 ; 

		for (int img_idx = 0; img_idx < num_images; ++img_idx)
		{
			//LOGLN("Compositing image #" << indices[img_idx]+1);
  
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

			Mat K;
			cameras[img_idx].K().convertTo(K, CV_32F);
			// Warp the current image	
			warpsize( warper , img, K, cameras[img_idx].R, INTER_LINEAR, BORDER_REFLECT, img_warped , xmap1, ymap1 );

			// Warp the current image mask
			mask.create(img_size, CV_8U);
			mask.setTo(Scalar::all(255));
			warpsize( warper , mask, K, cameras[img_idx].R, INTER_NEAREST, BORDER_CONSTANT, mask_warped  , xmap1, ymap1);

			// Compensate exposure
			compensator->apply(img_idx, corners[img_idx], img_warped, mask_warped);
			img_warped.convertTo(img_warped_s, CV_16S);

			img_warped.release();
			img.release();
			mask.release();

			//img_warped_s  mask_warped
			dilate(masks_warped[img_idx], dilated_mask, Mat());
			resize(dilated_mask, seam_mask, mask_warped.size());
			mask_warped = seam_mask & mask_warped;

			if( img_idx == 0 )
				prepareSize( resultRoiSize(corners, sizes) ,   result1, result_mask1 , dst_roi_1  );

			// Blend the current image
			feedSize(img_warped_s, mask_warped, corners[img_idx] , dst_roi_1 , result1, result_mask1   );

		}

		Mat result2, result_mask2;
		blendSize(result2, result_mask2 ,  result1, result_mask1   );
		LOGLN("Compositing, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");
		imwrite(outbuf, result2);
		LOGLN("Finished, total time: " << ((getTickCount() - app_start_time) / getTickFrequency()) << " sec");
	}
	return 0;
}
