#pragma once
#ifndef REG
#define REG

#ifndef UTILS
#include "include/utils.h"
#endif

#include <random>
#include <string>
#include <iterator>
#include <algorithm>

namespace Registration
{

	struct params {
		double dist;
		std::string mode;
		int iterations;
		double vertical_shift;
		std::string loss;
		double k;
		size_t number_of_correspondences;

		params():dist(1.0), mode("point"), iterations(100), vertical_shift(-5.0), loss("l1"), k(1.0), number_of_correspondences(5000) {}
		params(double _dist, std::string _mode, int _iterations, double _vertical_shift, std::string _loss) :
			dist(_dist), mode(_mode), iterations(_iterations), vertical_shift(_vertical_shift), loss(_loss){}

		~params(){}
	};


	struct transform {
		Matrix::matrix* rotation = Matrix::identity(3);
		Matrix::matrix* translation = new Matrix::matrix(3, 1); 
		Matrix::matrix* H = Matrix::identity(4);
		double fitness = 0;
		double inlier_rms = 0;
		int inlier_count = 0;
		transform(){}
		transform(Matrix::matrix* rotation, Matrix::matrix* translation): rotation(rotation), translation(translation){}
		Matrix::matrix* transformation()
		{
			for(int col = 1; col <= 3; col++) Matrix::setColumn(H, Matrix::getColumn(rotation, col), col);
			Matrix::setColumn(H, Matrix::getColumn(translation, 1), 4);
			return H;
		}
		~transform(){}
	};


	struct icp {
		params* P;
		transform* update;
		Matrix::matrix* _dynamic = new Matrix::matrix(1, 1);
		Matrix::matrix* _static = new Matrix::matrix(1, 1);
		Matrix::matrix* _normals = new Matrix::matrix(1, 1);
		std::vector<std::tuple<int, int, double>> correspondences;  //dynamic id , static id, distance
		std::vector<int> _static_indexes, _dynamic_indexes;
		utils::PointCloudAdaptor* tree;                             //tree for the model

		icp(Matrix::matrix* model, utils::PointCloudAdaptor* modeltree, Matrix::matrix* scene, Registration::params* params, transform* transform)
			:_static(model), _dynamic(scene), update(transform), tree(modeltree), P(params)
		{
			setIndexes();
			point2point_svd();
			*scene = *_dynamic;
		}

		icp(Matrix::matrix* model, Matrix::matrix* normals, utils::PointCloudAdaptor* modeltree, Matrix::matrix* scene, Registration::params* params, transform* transform)
			:_static(model), _normals(normals), _dynamic(scene), update(transform), tree(modeltree), P(params)
		{
			setIndexes();
			if (P->mode == "point") point2point_svd();
			if (P->mode == "plane") point2plane();
			*scene = *_dynamic;
		}

		double kernel(const double& residual, const std::string& loss)
		{
			if (loss == "l1") return 1 / residual;
			if (loss == "huber") return P->k / MAX(P->k, residual);
			if (loss == "cauchy") return 1 / (1 + pow(residual / P->k, 2));
			if (loss == "gm") return P->k / pow(P->k + pow(residual, 2), 2);
			if (loss == "tukey") return pow(1.0 - pow(MIN(1.0, abs(residual) / P->k), 2), 2);
		}

		void setIndexes()
		{
			//static cloud
			_static_indexes.resize(_static->cols);
			std::iota(std::begin(_static_indexes), std::end(_static_indexes), 1);

			//dynamic cloud
			_dynamic_indexes.resize(_dynamic->cols);
			std::iota(std::begin(_dynamic_indexes), std::end(_dynamic_indexes), 1);
		}

		void getCorrespondences();
		void umeyama(Matrix::matrix* A, Matrix::matrix* B, Matrix::matrix* R, Matrix::matrix* t);
		void jacobian1(Matrix::matrix* rot_p, Matrix::matrix* J);
		void point2point();
		void point2point_svd();
		void jacobian2(Matrix::matrix* rot_p, Matrix::matrix* normals, Matrix::matrix* J);
		void point2plane();
		void updateDynamic(Matrix::matrix* deltaR, Matrix::matrix* deltaT);

		~icp() {}
	};

	Matrix::matrix* principal_axis(Matrix::matrix* data);
	int getRotations(Matrix::matrix* model, Matrix::matrix* scene, Matrix::matrix* R, Matrix::matrix* R_eta);
	Matrix::matrix* alignment(Matrix::matrix* cloud, Matrix::matrix* R, double vertical_shift);
	int Align(utils::PointCloud* model, utils::PointCloud* scene, utils::PointCloud* scene_out);
}

#endif 