#include "registration.h"

Matrix::matrix* Registration::principal_axis(Matrix::matrix* data)
{

	Matrix::matrix* covariance = product(transpose(data), data);
	//get largetst eigenpair using power method
	Matrix::matrix* v = Matrix::newMatrix(covariance->rows, 1); set(v, 3, 1, 1);
	*v = *product(power(covariance, 10), v); //10 iterations
	double vn = norm(v);

	if (vn == 0) return v;
	return scalar_prod(v, (1 / vn));
}

int Registration::getRotations(Matrix::matrix* model, Matrix::matrix* scene, Matrix::matrix* R, Matrix::matrix* R_eta)
{
	if (!R || !R_eta) return -1;

	Matrix::matrix* v1 = principal_axis(model);
	Matrix::matrix* v2 = principal_axis(scene);

	//rotation axis and matrix
	double theta = acos(MIN(dotProduct(v1, v2), 1.0));
	Matrix::matrix* axis = product(skewSymmetric3D(v2), v1);
	double axis_norm = norm(axis);
	if (axis_norm != 0) *axis = *scalar_prod(axis, axis_norm);
	Matrix::matrix* s_ax = skewSymmetric3D(axis);

	//rodrigues formula
	*R = *sum(Matrix::identity(3), scalar_prod(s_ax, sin(theta)));
	*R = *sum(R, scalar_prod(power(s_ax, 2), 1 - cos(theta)));

	//flipped pcd
	double eta = theta + M_PI;
	*R_eta = *sum(Matrix::identity(3), scalar_prod(s_ax, sin(eta)));
	*R_eta = *sum(R_eta, scalar_prod(power(s_ax, 2), 1 - cos(eta)));

	return 0;

}


Matrix::matrix* Registration::alignment(Matrix::matrix* cloud, Matrix::matrix* R, double vertical_shift)
{
	Matrix::matrix* out;
	Matrix::matrix* t = Matrix::newMatrix(3, 1); set(t, 3, 1, vertical_shift);
	out = Matrix::sum(Matrix::product(R, Matrix::transpose(cloud)), Matrix::product(R, t));
	return out;
}

int Registration::coarseAlign(utils::PointCloud* _model, utils::PointCloud* _scene, utils::PointCloud* scene_out)
{
	Registration::params Params = Registration::params(1, 1, 100, -5);

	utils::point staticMean, dynamicMean;
	shiftByMean(_model, staticMean);
	shiftByMean(_scene, dynamicMean);

	//matrix form
	Matrix::matrix* model = Matrix::newMatrix(_model->points.size(), 3);
	Matrix::matrix* scene = Matrix::newMatrix(_scene->points.size(), 3);
	Matrix::pcd2mat(_model, model); Matrix::pcd2mat(_scene, scene);

	//get model pcd and downsample and make tree and normals
	utils::computeNormals(_model, 10);

	//filter scene with statistical outlier filter

	//alignment
	Matrix::matrix* R = Matrix::identity(3);
	Matrix::matrix* R_eta = Matrix::identity(3);
	getRotations(model, scene, R, R_eta);
	print(R); print(R_eta);

	Matrix::matrix* aligned_left_up = alignment(scene, R, Params.vertical_shift);
	Matrix::matrix* aligned_right_up = alignment(scene, R_eta, Params.vertical_shift);
	Matrix::matrix* aligned_left_down = alignment(scene, R, -Params.vertical_shift);
	Matrix::matrix* aligned_right_down = alignment(scene, R_eta, -Params.vertical_shift);


	return 0;
}

void Registration::icp::getCorrespondences()
{
	correspondences.reserve(P.number_of_correspondences);

	//sample from dynamic cloud
	std::vector<int> dynamic_samples;
	{
		std::random_device rd; std::mt19937 g(rd());
		if (P.number_of_correspondences < _static_indexes.size())
		{
			std::shuffle(_dynamic_indexes.begin(), _dynamic_indexes.end(), g); 
			std::copy(_dynamic_indexes.begin(), _dynamic_indexes.begin() + P.number_of_correspondences, dynamic_samples.begin());
		}
		else
		{
			dynamic_samples = _dynamic_indexes;
		}
	}

	//get nearest neighbor indices in static scene


}

void Registration::icp::point2point()
{

}

void Registration::icp::point2plane()
{

}