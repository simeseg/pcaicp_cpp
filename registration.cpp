#include "registration.h"

typedef Matrix::matrix* TMat;
/*
#pragma omp declare reduction(matsum : TMat: \
omp_out = Matrix::sum(omp_out, omp_in)\
initializer(omp_priv=new TMat(omp_in-rows, omp_in->cols))
*/

Matrix::matrix* Registration::principal_axis(Matrix::matrix* data)
{   /*
	Matrix::matrix* mean = new Matrix::matrix(3,1);
	//#pragma omp parallel for
	for (int idx = 1; idx <= data->cols; idx++)
	{
		*mean = *sum(mean, Matrix::getColumn(data, idx));
	}
	mean = Matrix::scalar_prod(mean, (double) 1/data->cols);
	print(mean);
	Matrix::matrix* shifted = new Matrix::matrix(data->rows, data->cols);
	//#pragma omp parallel for
	for (int idx = 1; idx <= data->cols; idx++)
	{
		Matrix::setColumn(shifted, diff(Matrix::getColumn(data, idx), mean), idx);
	}
	*/
	Matrix::matrix* covariance = Matrix::scalar_prod(Matrix::product(data, Matrix::transpose(data)),1/(double)(data->cols - 1));
	
	//get largetst eigenpair using power method
	Matrix::matrix* v = new Matrix::matrix(3, 1); set(v, 3, 1, 1);
	Matrix::matrix* p = Matrix::product(covariance, covariance); print(covariance);
	*v = *Matrix::product(p, v); //10 iterations
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
	Matrix::matrix* out = new Matrix::matrix(cloud->rows, cloud->cols);
	Matrix::matrix* t = new Matrix::matrix(3, 1); set(t, 3, 1, vertical_shift);
	Matrix::matrix* prod = Matrix::product(R, cloud);
#pragma omp parallel for
	for (int col = 1; col <= cloud->cols; col++) Matrix::setColumn(out, Matrix::sum(Matrix::getColumn(prod, col), Matrix::product(R, t)), col);
	delete prod, t;
	return out;
}

int Registration::Align(utils::PointCloud* _model, utils::PointCloud* _scene, utils::PointCloud* scene_out)
{
	Registration::params Params = Registration::params(2, "plane", 20, -5, "l1"); Params.number_of_correspondences = 1000;

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

	//filter scene with statistical outlier filter
	utils::PointCloud* scene_filtered = new utils::PointCloud;
	utils::statisticalOutlierRemoval(_scene, scene_filtered, 30, 0.5); *_scene = *scene_filtered;

	//convert to matrix form
	Matrix::matrix* model = new Matrix::matrix(3, _model->points.size());
	Matrix::matrix* model_normals = new Matrix::matrix(3, _model->points.size());
	Matrix::matrix* scene = new Matrix::matrix(3, _scene->points.size());
	Matrix::getPosition(_model, model); 
	Matrix::getNormals(_model, model_normals); 
	Matrix::getPosition(_scene, scene);



	//coarse alignment
	Matrix::matrix* R = Matrix::identity(3); Matrix::matrix* R_eta = Matrix::identity(3);
	getRotations(model, scene, R, R_eta);
	print(R); print(R_eta);

	Matrix::matrix* aligned= alignment(scene, R, Params.vertical_shift);
	Matrix::matrix* aligned_eta = alignment(scene, R_eta, Params.vertical_shift);

	auto writepcd = [&](std::string filename, Matrix::matrix* cloud)
	{
		std::ofstream out(filename);
		for (int i = 1; i < cloud->cols; i++)
		{
			out << Matrix::get(cloud, 1, i) << "," << Matrix::get(cloud, 2, i) << "," << Matrix::get(cloud, 3, i) << "\n";
		}
	};
	writepcd("image/aligned.txt", aligned);
	writepcd("image/aligned_eta.txt", aligned_eta);

	//do icp
	Registration::transform transform;
	Registration::icp icp(model, model_normals, &modelAdaptor, aligned, Params.dist, Params.mode, Params.iterations, Params.loss, &transform);
	Registration::transform transform_eta;
	Registration::icp icp_eta(model, model_normals, &modelAdaptor, aligned_eta, Params.dist, Params.mode, Params.iterations, Params.loss, &transform_eta);

	writepcd("image/icp.txt", aligned);
	writepcd("image/icp_eta.txt", aligned_eta);

	Registration::transform transform_out = (((transform.fitness) > (transform_eta.fitness)) ? (transform) : (transform_eta));

	Matrix::matrix* h_o2s = Matrix::identity(4); 
	Matrix::matrix* dynamicMeanMat = new Matrix::matrix(3, 1); double arr1[] = { dynamicMean.x, dynamicMean.y, dynamicMean.z };
	dynamicMeanMat->data.assign(arr1, arr1 + 3); Matrix::setColumn(h_o2s, dynamicMeanMat, 4); delete dynamicMeanMat, arr1;

	Matrix::matrix* h_m2o = Matrix::identity(4);
	Matrix::matrix* staticMeanMat = new Matrix::matrix(3, 1); double arr2[] = { -staticMean.x, -staticMean.y, -staticMean.z };
	staticMeanMat->data.assign(arr2, arr2 + 3); Matrix::setColumn(h_m2o, staticMeanMat, 4); delete staticMeanMat, arr2;

	Matrix::matrix* h_eig = Matrix::identity(4);
	Matrix::setRow(h_eig, Matrix::transpose(Matrix::getColumn(R, 1)), 1);
	Matrix::setRow(h_eig, Matrix::transpose(Matrix::getColumn(R, 2)), 2);
	Matrix::setRow(h_eig, Matrix::transpose(Matrix::getColumn(R, 3)), 3);
	Matrix::matrix* shift = new Matrix::matrix(3, 1); 
	shift->data[0] = 0; shift->data[0] = 0; shift->data[0] = -Params.vertical_shift; 
	Matrix::setColumn(h_eig, Matrix::product(transform_out.rotation, shift), 4); shift;

	Matrix::matrix* h_final = Matrix::product( transform_out.transformation(), h_m2o);
	*h_final = *Matrix::product(h_eig, h_final);
	*h_final = *Matrix::product(h_o2s, h_final);

	print(h_final);
	std::cout << transform_out.fitness;
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
//#pragma omp parallel for
	for (auto& idx : dynamic_samples)
	{
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
		e = new Matrix::matrix(3, 1); Matrix::set(e, i, 1, 1);
		Matrix::setColumn(J, Matrix::product(Matrix::skewSymmetric3D(rot_p), e), i);
		Matrix::setColumn(J, e, i+3);
		delete e;
	}
}

void Registration::icp::jacobian2(Matrix::matrix* rot_p, Matrix::matrix* normal, Matrix::matrix* J)
{
	Matrix::matrix* pxn = Matrix::product(Matrix::skewSymmetric3D(rot_p), normal);
	for (int i = 1; i <= 3; i++)
	{
		Matrix::set(J, 1, i, Matrix::get(pxn, i, 1));
		Matrix::set(J, 1, i+3, Matrix::get(normal,i,1)); 
	}
	delete pxn;
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

		{
			Matrix::matrix* H = new Matrix::matrix(6, 6);
			Matrix::matrix* b = new Matrix::matrix(6, 1);
			int i;
			for (i = 0; i < correspondences.size(); i++)
			{
				//std::cout << i << "\n";
				Matrix::matrix* p = Matrix::getColumn(_dynamic, std::get<0>(correspondences.at(i)));
				Matrix::matrix* x = Matrix::getColumn(_static, std::get<1>(correspondences.at(i)));
				double distance = std::get<2>(correspondences.at(i));
				if (distance <= P.dist)
				{
					Matrix::matrix* e = diff(p, x);
					double w = kernel(Matrix::norm(e), P.loss);
					Matrix::matrix* J = new Matrix::matrix(3, 6);
					jacobian1(p, J); 
					Matrix::matrix* wJT = Matrix::scalar_prod(Matrix::transpose(J), w);
					*H = *sum(H, Matrix::product(wJT, J));
					*b = *sum(b, Matrix::product(wJT, e));
					delete J, wJT; 
				}
			}

			Matrix::matrix* deltaEuler = new Matrix::matrix(6, 1);
			Matrix::solver(H, b, deltaEuler);
			Matrix::matrix* deltaR = Matrix::identity(3); Matrix::matrix* deltaT = new Matrix::matrix(3, 1);

			euler2rot(Matrix::scalar_prod(deltaEuler, -1), deltaR, deltaT);
			*update->rotation = *product(deltaR, update->rotation);
			*update->translation = *sum(product(deltaR, update->translation), deltaT);

			updateDynamic(deltaR, deltaT);
			delete H, b, deltaEuler, deltaR, deltaT;
		}

		//get inliers and rmse
		for (int i = 0; i < correspondences.size(); i++)
		{
			double distance = std::get<2>(correspondences.at(i));
			if (distance <= P.dist)
			{
				error2 += pow(distance, 2);
				inlier_count += 1;
				_static_inliers.push_back(std::get<1>(correspondences.at(i)));
			}
		}

		update->inlier_count = inlier_count;
		update->inlier_rms = sqrt(error2 / (double)inlier_count); //rms error
		update->fitness = std::set<int>(_static_inliers.begin(), _static_inliers.end()).size() / (double)_static->cols;   //no of unique model inliers to no of model points
		std::cout << "iter: " << iter;
		std::cout << "; inlier count: " << update->inlier_count;
		std::cout << "; fitness: " << update->fitness;
		std::cout << "; inlier rms: " << update->inlier_rms << "\n";
		//next
		iter++; inlier_count = 0; error2 = 0; _static_inliers.clear();
	}
	print(update->rotation); print(update->translation);
}

void Registration::icp::point2plane()
{

	//////Gauss-Newton Method (point to plane)//////
	int iter = 0;
	double error2 = 0;
	int inlier_count = 0;
	std::vector<int> _static_inliers;
	while (iter < P.iterations)
	{
		{
			getCorrespondences();

			Matrix::matrix* H = new Matrix::matrix(6, 6);
			Matrix::matrix* b = new Matrix::matrix(6, 1);
			int i;

//#pragma omp for private(i) 
			for (i = 0; i < correspondences.size(); i++)
			{
				Matrix::matrix* p = Matrix::getColumn(_dynamic, std::get<0>(correspondences.at(i)));
				Matrix::matrix* x = Matrix::getColumn(_static, std::get<1>(correspondences.at(i)));
				Matrix::matrix* n = Matrix::getColumn(_normals, std::get<1>(correspondences.at(i)));
				double distance = std::get<2>(correspondences.at(i));
				if (distance <= P.dist)
				{
					double e = Matrix::dotProduct(diff(p , x), n);
					double w = kernel(e, P.loss);
					Matrix::matrix* J = new Matrix::matrix(1, 6);
					jacobian2(p, n, J);
					Matrix::matrix* wJT = Matrix::scalar_prod(Matrix::transpose(J), w);
					*H = *sum(H, Matrix::product(wJT, J));
					*b = *diff(b, Matrix::scalar_prod(wJT, e));
					delete J, wJT;
				}
			}

			Matrix::matrix* deltaEuler = new Matrix::matrix(6, 1);
			Matrix::solver(H, b, deltaEuler);
			Matrix::matrix* deltaR = Matrix::identity(3); Matrix::matrix* deltaT = new Matrix::matrix(3, 1);
			euler2rot(deltaEuler, deltaR, deltaT);

			*update->rotation = *product(deltaR, update->rotation);
			*update->translation = *sum(product(deltaR, update->translation), deltaT);

			updateDynamic(deltaR, deltaT);

			delete H, b, deltaEuler, deltaR, deltaT;
		}
		//get inliers and rmse
		for (int i = 0; i < correspondences.size(); i++)
		{
			double distance = std::get<2>(correspondences.at(i));
			if (distance <= P.dist)
			{
				error2 += pow(distance, 2);
				inlier_count += 1;
				_static_inliers.push_back(std::get<1>(correspondences.at(i)));
			}
		}

		update->inlier_count = inlier_count;
		update->inlier_rms = sqrt(error2 / (double)inlier_count); //rms error
		update->fitness = std::set<int>(_static_inliers.begin(), _static_inliers.end()).size() / (double)_static->cols;   //no of unique model inliers to no of model points
		std::cout << "iter: " << iter;
		std::cout << "; inlier count: " << update->inlier_count;
		std::cout << "; fitness: " << update->fitness;
		std::cout << "; inlier rms: " << update->inlier_rms << "\n";
		//next
		iter++; inlier_count = 0; error2 = 0; _static_inliers.clear();
	}
	print(update->rotation); print(update->translation);
}

void Registration::icp::updateDynamic(Matrix::matrix* deltaR, Matrix::matrix* deltaT)
{
	//update dynamic cloud
	//print(deltaR), print(deltaT);
	*_dynamic = *Matrix::product(deltaR, _dynamic);
	int col; Matrix::matrix* temp = new Matrix::matrix(3, 1);
//#pragma omp parallel for private(col)
	for (col = 1; col <= _dynamic->cols; col++)
	{
		*temp = *sum(Matrix::getColumn(_dynamic, col), deltaT);
		Matrix::setColumn(_dynamic, temp, col);
	}
	delete temp;
}