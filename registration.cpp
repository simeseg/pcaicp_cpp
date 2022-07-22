#include "registration.h"

Matrix::matrix* Registration::principal_axis(Matrix::matrix* data)
{   /*
	Matrix::matrix* mean = Matrix::newMatrix(3,1);
	//#pragma omp parallel for
	for (int idx = 1; idx <= data->cols; idx++)
	{
		*mean = *sum(mean, Matrix::getColumn(data, idx));
	}
	mean = Matrix::scalar_prod(mean, (double) 1/data->cols);
	print(mean);
	Matrix::matrix* shifted = Matrix::newMatrix(data->rows, data->cols);
	//#pragma omp parallel for
	for (int idx = 1; idx <= data->cols; idx++)
	{
		Matrix::setColumn(shifted, diff(Matrix::getColumn(data, idx), mean), idx);
	}
	*/
	Matrix::matrix* covariance = Matrix::product(data, Matrix::transpose(data));
	
	//get largetst eigenpair using power method
	Matrix::matrix* v = Matrix::newMatrix(3, 1); set(v, 3, 1, 1);
	*v = *Matrix::product(Matrix::power(covariance, 10), v); //10 iterations
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
	if (axis_norm != 0) *axis = *scalar_prod(axis, 1/axis_norm);
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
	out = Matrix::sum(Matrix::product(R, cloud), Matrix::product(R, t));
	return out;
}

int Registration::Align(utils::PointCloud* _model, utils::PointCloud* _scene, utils::PointCloud* scene_out)
{
	Registration::params Params = Registration::params(1, 1, 100, -5);

	utils::point staticMean, dynamicMean;
	shiftByMean(_model, staticMean);
	shiftByMean(_scene, dynamicMean);

	//get model pcd and downsample and make tree and normals
	//make a kdtree

	utils::PointCloudAdaptor modelAdaptor(3, *_model, 10);
	modelAdaptor.index->buildIndex();

	//get normals
	int nn = 10;
	utils::computeNormals(_model, modelAdaptor, nn);

	//matrix form
	Matrix::matrix* model = Matrix::newMatrix(3, _model->points.size());
	Matrix::matrix* scene = Matrix::newMatrix(3, _scene->points.size());
	Matrix::pcd2mat(_model, model); Matrix::pcd2mat(_scene, scene);

	//filter scene with statistical outlier filter


	//coarse alignment
	Matrix::matrix* R = Matrix::identity(3); Matrix::matrix* R_eta = Matrix::identity(3);
	getRotations(model, scene, R, R_eta);
	print(R); print(R_eta);

	Matrix::matrix* aligned_left_up = alignment(scene, R, Params.vertical_shift);
	Matrix::matrix* aligned_right_up = alignment(scene, R_eta, Params.vertical_shift);

	//do icp
	Registration::transform transform1;
	Registration::icp icp1(model, &modelAdaptor, aligned_left_up, Params.dist, Params.mode, Params.iterations, Params.loss, &transform1);
	icp1.point2point();
	return 0;
}

void Registration::icp::getCorrespondences()
{
	correspondences.clear(); correspondences.reserve(P.number_of_correspondences);

	//sample from dynamic cloud
	std::vector<int> dynamic_samples(P.number_of_correspondences);
	{
		std::random_device rd; std::mt19937 g(rd());
		if (P.number_of_correspondences < _dynamic_indexes.size())
		{
			std::shuffle(_dynamic_indexes.begin(), _dynamic_indexes.end(), g); 
			std::copy(_dynamic_indexes.begin(), _dynamic_indexes.begin() + P.number_of_correspondences, dynamic_samples.begin());
		}
		else{ dynamic_samples = _dynamic_indexes;}
	}
	
	//get nearest neighbor indices and distance in static scene
	std::vector<int> static_neighbors(dynamic_samples.size());
	for (auto& idx : dynamic_samples) {
		
		Matrix::matrix* pt = getColumn(_dynamic, idx);
		double query_point[] = { Matrix::get(pt,1,1), Matrix::get(pt,2,1), Matrix::get(pt,3,1)};
		size_t out_indices[1];   
		double out_distances[1]; 

		nanoflann::KNNResultSet<double> resultSet(1);
		resultSet.init(&out_indices[0], &out_distances[0]);
		tree->index->findNeighbors(resultSet, &query_point[0], 1);
		std::tuple<int, int, double> corresp = { idx, (int)out_indices[0], (double)sqrt(out_distances[0]) };
		correspondences.push_back(corresp);
		delete pt;
	}
}

void Registration::icp::jacobian1(Matrix::matrix* rot_p, Matrix::matrix* J)
{
	Matrix::matrix* e;
	for (int i = 1; i <= 3; i++)
	{
		e = Matrix::newMatrix(3, 1); Matrix::set(e, i, 1, 1);
		Matrix::setColumn(J, Matrix::scalar_prod(Matrix::product(Matrix::skewSymmetric3D(e), rot_p), -1), i);
		Matrix::setColumn(J, e, i+3);
		delete e;
	}
}

void Registration::icp::point2point()
{

	//////Gauss-Newton Method//////
	int iter = 0;
	double error2 = 0;
	int inlier_count = 0;
	std::vector<int> _static_inliers;
	while (iter < P.iterations)
	{
		getCorrespondences();
		Matrix::matrix* H = Matrix::newMatrix(6, 6);
		Matrix::matrix* b = Matrix::newMatrix(6, 1);
		Matrix::matrix* J = Matrix::newMatrix(3, 6);
		Matrix::matrix* rot = update.rotation;
		Matrix::matrix* trans = update.translation;
//#pragma omp parallel for
		for (int i = 0; i < correspondences.size(); i++)
		{
			Matrix::matrix* p = Matrix::getColumn(_dynamic, std::get<0>(correspondences.at(i)));
			Matrix::matrix* x = Matrix::getColumn(_static, std::get<1>(correspondences.at(i)));
			double distance   = std::get<2>(correspondences.at(i));
			if (distance <= P.dist)
			{
				Matrix::matrix* rot_p = Matrix::product(rot, p);
				Matrix::matrix* e = diff(sum(rot_p, trans), x);
				double w = kernel(Matrix::norm(e), "huber");
				jacobian1(rot_p, J); Matrix::matrix* wJT = Matrix::scalar_prod(Matrix::transpose(J), w);
				*H = *sum(H, Matrix::product(wJT, J));
				*b = *sum(b, Matrix::product(wJT, e));
			}
		}
		
		Matrix::matrix* deltaEuler = Matrix::newMatrix(6, 1);
		Matrix::solver(H, b, deltaEuler);
		
		Matrix::matrix* deltaR = Matrix::identity(3); Matrix::matrix* deltaT = Matrix::newMatrix(3, 1);

		euler2rot(Matrix::scalar_prod(deltaEuler, -1), deltaR, deltaT);
		*update.rotation = *product(deltaR, update.rotation);
		*update.translation = *sum(product(deltaR, update.translation), deltaT);
		
		delete H, b, J, deltaEuler, deltaR, deltaT;
		
		//get inliers and rmse
		for (int i = 0; i < correspondences.size(); i++)
		{
			double distance = std::get<2>(correspondences.at(i));
			if (distance <= P.dist) 
			{
				error2 += pow(distance, 2); 
				inlier_count+=1;
				_static_inliers.push_back(std::get<1>(correspondences.at(i)));
			}
		}

		update.inlier_count = inlier_count;
		update.inlier_rms = sqrt(error2 / (double)inlier_count); //rms error
		update.fitness = std::set<int>(_static_inliers.begin(), _static_inliers.end()).size()/(double)_static->cols;   //no of unique model inliers to no of model points
		
        //next
		iter++; inlier_count = 0; error2 = 0; _static_inliers.clear();
	}

	//update dynamic cloud
	*_dynamic = *Matrix::product(update.rotation, _dynamic);
	Matrix::matrix* temp = Matrix::newMatrix(3, 1);
	for (int col = 1; col <= _dynamic->cols; col++)
	{
		*temp = *sum(Matrix::getColumn(_dynamic, col), update.translation);
		Matrix::setColumn(_dynamic, temp, col);
	}
	print(update.rotation); print(update.translation);
	delete temp;
}

void Registration::icp::point2plane()
{

}