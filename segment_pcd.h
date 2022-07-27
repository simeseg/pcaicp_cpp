#pragma once
#define _CRT_SECURE_NO_WARNINGS
#include "utils.h"


namespace segmentpcd
{
	
	struct _outputClouds
	{
		std::vector<utils::PointCloud> output_clouds;
	};

	struct Pixel 
	{
		int r = 0, g = 0, b = 0;
		int grey;
		segmentpcd::Pixel() : grey((int)((r + g + b) / 3)) {};
		double Xc = 0, Yc = 0, Zc = 0;
		double Xw = 0, Yw = 0, Zw = 0;
	
	};

	struct Image
	{
		int width = 0;
		int height = 0;
		std::vector<Pixel> pixels;
	};

	struct Bbox
	{
		int index = 0;
		int x_left = 0, y_left = 0, size_x = 0, size_y = 0;
	};

	struct Bboxes
	{
		size_t num = 0;
		std::vector<Bbox> bboxes;
	};

	struct _masks
	{
		int thresh = 200;
		std::vector<Image> masks;
	};

	enum CoordType : int
	{
		Camera = 0,
		World,
		Local,
	};
	struct IntrinsicPara
	{
		double u0;
		double v0;
		double lambda;
		double fx;
		double fy;
		double skew;

		double k1;
		double k2;
		double p1;
		double p2;
		double k3;
		double k4;
		double p3;
		double p4;
	};
	struct ExtrinsicPara
	{
		double CameraToWorld_R[3][3];
		double CameraToWorld_T[3][1];
		double fLocalToWorld[6];
	};
	struct CalibData
	{
		int nImageWidth;
		int nImageHeight;
		IntrinsicPara IntrinsicPara;
		ExtrinsicPara ExtrinsicPara;
	};

	BOOL _SavePointCloudToPly(const utils::PointCloud& i_PointCloud, const CString& SavePath, BOOL bBinary);

	BOOL _LoadPly(char const* i_strPlyPath, utils::PointCloud& o_PointCloud);

	BOOL _LoadRGBImage(char const* i_strImagePath, segmentpcd::Image& o_image);

	void _project(const utils::PointCloud& i_PointCloud, const segmentpcd::CalibData& i_CalibData, segmentpcd::Image& o_Image);

	void _readBboxes(char const* i_strBboxPath, segmentpcd::Bboxes& bboxes);

	void _read_mask(char const* i_strMaskPath, segmentpcd::Image& mask);

	void _get_masks(segmentpcd::_masks& masks, size_t& num_bboxes, int& scene_id);

	void _clip_bolt_pcd(segmentpcd::Bboxes& bboxes, segmentpcd::Image& o_image, segmentpcd::_masks& masks, segmentpcd::_outputClouds& o_pointclouds);

}