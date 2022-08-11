#pragma once
#define _CRT_SECURE_NO_WARNINGS
#ifndef UTILS
#define UTILS
#include "utils.h"
#endif //!UTILS

namespace align
{
	struct param
	{
		double dist = .5;
		bool mode = 0;
		int iterations = 100;
	};
	
	void _shift_by_mean(const utils::PointCloud& in_pointcloud, utils::PointCloud& out_pointcloud);
	
	void _get_rotation_matrices(const utils::PointCloud& model, const utils::PointCloud& scene, Matrix::matrix R, Matrix::matrix R_eta);
}