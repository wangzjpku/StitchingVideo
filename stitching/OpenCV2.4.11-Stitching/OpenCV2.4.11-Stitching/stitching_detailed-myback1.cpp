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

static void printUsage()
{
    cout <<
        "Rotation model images stitcher.\n\n"
        "stitching_detailed img1 img2 [...imgN] [flags]\n\n"
        "Flags:\n"
        "  --preview\n"
        "      Run stitching in the preview mode. Works faster than usual mode,\n"
        "      but output image will have lower resolution.\n"
        "  --try_gpu (yes|no)\n"
        "      Try to use GPU. The default value is 'no'. All default values\n"
        "      are for CPU mode.\n"
        "\nMotion Estimation Flags:\n"
        "  --work_megapix <float>\n"
        "      Resolution for image registration step. The default is 0.6 Mpx.\n"
        "  --features (surf|orb)\n"
        "      Type of features used for images matching. The default is surf.\n"
        "  --match_conf <float>\n"
        "      Confidence for feature matching step. The default is 0.65 for surf and 0.3 for orb.\n"
        "  --conf_thresh <float>\n"
        "      Threshold for two images are from the same panorama confidence.\n"
        "      The default is 1.0.\n"
        "  --ba (reproj|ray)\n"
        "      Bundle adjustment cost function. The default is ray.\n"
        "  --ba_refine_mask (mask)\n"
        "      Set refinement mask for bundle adjustment. It looks like 'x_xxx',\n"
        "      where 'x' means refine respective parameter and '_' means don't\n"
        "      refine one, and has the following format:\n"
        "      <fx><skew><ppx><aspect><ppy>. The default mask is 'xxxxx'. If bundle\n"
        "      adjustment doesn't support estimation of selected parameter then\n"
        "      the respective flag is ignored.\n"
        "  --wave_correct (no|horiz|vert)\n"
        "      Perform wave effect correction. The default is 'horiz'.\n"
        "  --save_graph <file_name>\n"
        "      Save matches graph represented in DOT language to <file_name> file.\n"
        "      Labels description: Nm is number of matches, Ni is number of inliers,\n"
        "      C is confidence.\n"
        "\nCompositing Flags:\n"
        "  --warp (plane|cylindrical|spherical|fisheye|stereographic|compressedPlaneA2B1|compressedPlaneA1.5B1|compressedPlanePortraitA2B1|compressedPlanePortraitA1.5B1|paniniA2B1|paniniA1.5B1|paniniPortraitA2B1|paniniPortraitA1.5B1|mercator|transverseMercator)\n"
        "      Warp surface type. The default is 'spherical'.\n"
        "  --seam_megapix <float>\n"
        "      Resolution for seam estimation step. The default is 0.1 Mpx.\n"
        "  --seam (no|voronoi|gc_color|gc_colorgrad)\n"
        "      Seam estimation method. The default is 'gc_color'.\n"
        "  --compose_megapix <float>\n"
        "      Resolution for compositing step. Use -1 for original resolution.\n"
        "      The default is -1.\n"
        "  --expos_comp (no|gain|gain_blocks)\n"
        "      Exposure compensation method. The default is 'gain_blocks'.\n"
        "  --blend (no|feather|multiband)\n"
        "      Blending method. The default is 'multiband'.\n"
        "  --blend_strength <float>\n"
        "      Blending strength from [0,100] range. The default is 5.\n"
        "  --output <result_img>\n"
        "      The default is 'result.jpg'.\n";
}


// Default command line args
vector<string> img_names;
bool preview = false;
bool try_gpu = false;
double work_megapix = 0.6;
double seam_megapix = 0.1;
double compose_megapix = -1;
float conf_thresh = 1.f;
string features_type = "surf";
string ba_cost_func = "ray";
string ba_refine_mask = "xxxxx";
bool do_wave_correct = true;
WaveCorrectKind wave_correct = detail::WAVE_CORRECT_HORIZ;
bool save_graph = false;
std::string save_graph_to;
string warp_type = "cylindrical";//"spherical"; //"cylindrical";//
int expos_comp_type = ExposureCompensator::GAIN_BLOCKS;
float match_conf = 0.3f;
string seam_find_type = "gc_color";
int blend_type = Blender::MULTI_BAND;
float blend_strength = 5;
string result_name = "result.jpg";

static int parseCmdArgs(int argc, char** argv)
{
    if (argc == 1)
    {
        printUsage();
        return -1;
    }
    for (int i = 1; i < argc; ++i)
    {
        if (string(argv[i]) == "--help" || string(argv[i]) == "/?")
        {
            printUsage();
            return -1;
        }
        else if (string(argv[i]) == "--preview")
        {
            preview = true;
        }
        else if (string(argv[i]) == "--try_gpu")
        {
            if (string(argv[i + 1]) == "no")
                try_gpu = false;
            else if (string(argv[i + 1]) == "yes")
                try_gpu = true;
            else
            {
                cout << "Bad --try_gpu flag value\n";
                return -1;
            }
            i++;
        }
        else if (string(argv[i]) == "--work_megapix")
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
        else if (string(argv[i]) == "--result")
        {
            result_name = argv[i + 1];
            i++;
        }
        else if (string(argv[i]) == "--features")
        {
            features_type = argv[i + 1];
            if (features_type == "orb")
                match_conf = 0.3f;
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
        else if (string(argv[i]) == "--ba")
        {
            ba_cost_func = argv[i + 1];
            i++;
        }
        else if (string(argv[i]) == "--ba_refine_mask")
        {
            ba_refine_mask = argv[i + 1];
            if (ba_refine_mask.size() != 5)
            {
                cout << "Incorrect refinement mask length.\n";
                return -1;
            }
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
        else if (string(argv[i]) == "--warp")
        {
            warp_type = string(argv[i + 1]);
            i++;
        }
        else if (string(argv[i]) == "--expos_comp")
        {
            if (string(argv[i + 1]) == "no")
                expos_comp_type = ExposureCompensator::NO;
            else if (string(argv[i + 1]) == "gain")
                expos_comp_type = ExposureCompensator::GAIN;
            else if (string(argv[i + 1]) == "gain_blocks")
                expos_comp_type = ExposureCompensator::GAIN_BLOCKS;
            else
            {
                cout << "Bad exposure compensation method\n";
                return -1;
            }
            i++;
        }
        else if (string(argv[i]) == "--seam")
        {
            if (string(argv[i + 1]) == "no" ||
                string(argv[i + 1]) == "voronoi" ||
                string(argv[i + 1]) == "gc_color" ||
                string(argv[i + 1]) == "gc_colorgrad" ||
                string(argv[i + 1]) == "dp_color" ||
                string(argv[i + 1]) == "dp_colorgrad")
                seam_find_type = argv[i + 1];
            else
            {
                cout << "Bad seam finding method\n";
                return -1;
            }
            i++;
        }
        else if (string(argv[i]) == "--blend")
        {
            if (string(argv[i + 1]) == "no")
                blend_type = Blender::NO;
            else if (string(argv[i + 1]) == "feather")
                blend_type = Blender::FEATHER;
            else if (string(argv[i + 1]) == "multiband")
                blend_type = Blender::MULTI_BAND;
            else
            {
                cout << "Bad blending method\n";
                return -1;
            }
            i++;
        }
        else if (string(argv[i]) == "--blend_strength")
        {
            blend_strength = static_cast<float>(atof(argv[i + 1]));
            i++;
        }
        else if (string(argv[i]) == "--output")
        {
            result_name = argv[i + 1];
            i++;
        }
        else
            img_names.push_back(argv[i]);
    }
    if (preview)
    {
        compose_megapix = 0.6;
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
		{Point(u+nHalfWidthSrc,v+nHalfHeightSrc),	//срио)
		Point(u+nHalfWidthSrc,nHalfHeightSrc-v),
		Point(nHalfWidthSrc-u,nHalfHeightSrc-v),
		Point(nHalfWidthSrc-u,nHalfHeightSrc+v)};

	Point dstPoint[4] = 
		{Point(x+nHalfWidthDst,y+nHalfHeightDst),	//срио)
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

    double work_scale = 1, seam_scale = 1, compose_scale = 1;
    bool is_work_scale_set = false; 
	bool is_seam_scale_set = false, is_compose_scale_set = false;

    LOGLN("Finding features...");

    int64 t = getTickCount();

    Ptr<FeaturesFinder> finder;
    if (features_type == "surf")
    {
            finder = new SurfFeaturesFinder();
    }
    else
    {
        cout << "Unknown 2D features type: '" << features_type << "'.\n";
        return -1;
    }

    Mat full_img,first_img, img;
    vector<ImageFeatures> features(num_images);
    vector<Mat> images(num_images);
    vector<Size> full_img_sizes(num_images);
    double seam_work_aspect = 1;

    for (int i = 0; i < num_images; ++i)
    {
        full_img = imread(img_names[i]);
		
        full_img_sizes[i] = full_img.size();

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
    BestOf2NearestMatcher matcher(try_gpu, match_conf);
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

    // Leave only images we are sure are from the same panorama
    vector<int> indices = leaveBiggestComponent(features, pairwise_matches, conf_thresh);
    vector<Mat> img_subset;
    vector<string> img_names_subset;
    vector<Size> full_img_sizes_subset;
    for (size_t i = 0; i < indices.size(); ++i)
    {
        img_names_subset.push_back(img_names[indices[i]]);
        img_subset.push_back(images[indices[i]]);
        full_img_sizes_subset.push_back(full_img_sizes[indices[i]]);
    }

    images = img_subset;
    img_names = img_names_subset;
    full_img_sizes = full_img_sizes_subset;

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
        LOGLN("Initial intrinsics #" << indices[i]+1 << ":\n" << cameras[i].K());
    }

    Ptr<detail::BundleAdjusterBase> adjuster;
    if (ba_cost_func == "reproj") adjuster = new detail::BundleAdjusterReproj();
    else if (ba_cost_func == "ray") adjuster = new detail::BundleAdjusterRay();
    else
    {
        cout << "Unknown bundle adjustment cost function: '" << ba_cost_func << "'.\n";
        return -1;
    }
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
        LOGLN("Camera #" << indices[i]+1 << ":\n" << cameras[i].K());
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
    {
        if(warp_type == "cylindrical") warper_creator = new cv::CylindricalWarper();
        else if (warp_type == "spherical") warper_creator = new cv::SphericalWarper();
    }

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
    if (seam_find_type == "no")
        seam_finder = new detail::NoSeamFinder();
    else if (seam_find_type == "gc_color")
    {
            seam_finder = new detail::GraphCutSeamFinder(GraphCutSeamFinderBase::COST_COLOR);
    }

    if (seam_finder.empty())
    {
        cout << "Can't create the following seam finder '" << seam_find_type << "'\n";
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
    //double compose_seam_aspect = 1;
    double compose_work_aspect = 1;

	Mat result1, result_mask1;
	Rect dst_roi_1 ; 
	Mat xmap1, ymap1; 

    for (int img_idx = 0; img_idx < num_images; ++img_idx)
    {
        LOGLN("Compositing image #" << indices[img_idx]+1);

		Mat f_img;
        // Read image and resize it if necessary
        full_img = imread(img_names[img_idx]);

		//full_img = f_img.clone();
		//const uint theta1 = 150;
		//uint z1 = first_img.cols / tan(CV_PI*theta1/360);
		//FishEyeRectify(first_img  , full_img  , 600 );
        
		if (!is_compose_scale_set)
        {
            if (compose_megapix > 0)
                compose_scale = min(1.0, sqrt(compose_megapix * 1e6 / full_img.size().area()));
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
            resize(full_img, img, Size(), compose_scale, compose_scale);
        else
            img = full_img;
        full_img.release();
        Size img_size = img.size();

        Mat K;
        cameras[img_idx].K().convertTo(K, CV_32F);

        // Warp the current image
       // warper->warp(img, K, cameras[img_idx].R, INTER_LINEAR, BORDER_REFLECT, img_warped);
		
		warpsize( warper , img, K, cameras[img_idx].R, INTER_LINEAR, BORDER_REFLECT, img_warped , xmap1, ymap1 );

        // Warp the current image mask
        mask.create(img_size, CV_8U);
        mask.setTo(Scalar::all(255));
       // warper->warp(mask, K, cameras[img_idx].R, INTER_NEAREST, BORDER_CONSTANT, mask_warped);

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
/*
        if (blender.empty())
        {
            blender = Blender::createDefault(blend_type, try_gpu);
            Size dst_sz = resultRoi(corners, sizes).size();
            float blend_width = sqrt(static_cast<float>(dst_sz.area())) * blend_strength / 100.f;
            if (blend_width < 1.f)
                blender = Blender::createDefault(Blender::NO, try_gpu);
            else if (blend_type == Blender::MULTI_BAND)
            {
                MultiBandBlender* mb = dynamic_cast<MultiBandBlender*>(static_cast<Blender*>(blender));
                mb->setNumBands(static_cast<int>(ceil(log(blend_width)/log(2.)) - 1.));
                LOGLN("Multi-band blender, number of bands: " << mb->numBands());
            }
            blender->prepare(corners, sizes);
        }
*/
		if( img_idx == 0 )
			prepareSize( resultRoiSize(corners, sizes) ,   result1, result_mask1 , dst_roi_1  );

        // Blend the current image
////        blender->feed(img_warped_s, mask_warped, corners[img_idx]);	

		feedSize(img_warped_s, mask_warped, corners[img_idx] , dst_roi_1 , result1, result_mask1   );

/*		char buf[10];
		string ss;
		sprintf(buf, "img%d.jpg", img_idx );
		ss = buf;
		imwrite( ss , img_warped_s);
		sprintf(buf, "mask%d.jpg", img_idx );
		ss = buf;
		imwrite( ss , mask_warped);		
*/
    }

//    Mat result, result_mask;
	Mat result2, result_mask2;
////    blender->blend(result, result_mask);
	
	blendSize(result2, result_mask2 ,  result1, result_mask1   );

    LOGLN("Compositing, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");

////    imwrite(result_name, result);

	imwrite("zmystitching.jpg", result2);

    LOGLN("Finished, total time: " << ((getTickCount() - app_start_time) / getTickFrequency()) << " sec");
    
	return 0;
}
