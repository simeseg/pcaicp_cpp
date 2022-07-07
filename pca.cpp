#include "pca.h"

using namespace align;

void align::_shift_by_mean(const utils::PointCloud& in_pointcloud, utils::PointCloud& out_pointcloud)
{
	utils::point mean;
	utils::PointCloud_mean(in_pointcloud, mean);
	
	for (auto& pt : in_pointcloud.points)
	{
		out_pointcloud.points.push_back(pt - mean);
	}
}

void _get_rotation_matrices(const utils::PointCloud& model, const utils::PointCloud& scene, Matrix::matrix* R, Matrix::matrix* R_eta)
{
	R = Matrix::newMatrix(4, 4);
	R_eta = Matrix::newMatrix(4, 4);
}