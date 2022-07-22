#pragma once
#include "utils.h"
#include <random>
#include <string>
#include <iterator>
#include <algorithm>

namespace Registration
{

	struct params {
		double dist;
		bool mode;
		int iterations;
		double vertical_shift;
		std::string loss;
		double k;
		size_t number_of_correspondences;

		params():dist(1.0), mode(1), iterations(100), vertical_shift(-5.0), loss("l1"), k(1.0), number_of_correspondences(500) {}
		params(double _dist, bool _mode, int _iterations, double _vertical_shift) :
			dist(_dist), mode(_mode), iterations(_iterations), vertical_shift(_vertical_shift) {}

		~params(){}
	};


	struct transform {
		Matrix::matrix* rotation = Matrix::identity(3);
		Matrix::matrix* translation = Matrix::newMatrix(3, 1); 
		double fitness = 0;
		double inlier_rms = 0;
		int inlier_count = 0;
		transform(){}
		transform(Matrix::matrix* rotation, Matrix::matrix* translation): rotation(rotation), translation(translation){}
		Matrix::matrix* transformation()
		{
			Matrix::matrix* H = Matrix::identity(4);
			for(int col = 1; col <= 3; col++) Matrix::setColumn(H, Matrix::getColumn(rotation, col), col);
			Matrix::setColumn(H, Matrix::getColumn(translation, 1), 4);
			return H;
		}
		~transform(){}
	};


	struct icp {
		params P;
		transform update;
		Matrix::matrix* _dynamic = Matrix::newMatrix(1, 1);
		Matrix::matrix* _static = Matrix::newMatrix(1,1);
		std::vector<std::tuple<int, int, double>> correspondences;  //dynamic id , static id, distance
		std::vector<int> _static_indexes, _dynamic_indexes;
		utils::PointCloudAdaptor* tree;                             //tree for the model

		icp(Matrix::matrix* model, utils::PointCloudAdaptor* modeltree, Matrix::matrix* scene, const double& dist, bool mode, int iterations, const std::string& loss, transform* transform)
			:_static(model), _dynamic(scene), update(*transform), tree(modeltree)
		{
			P.dist = dist; P.mode = mode; P.iterations = iterations; P.loss = loss;
			setIndexes();

		}

		double kernel(const double& residual,const std::string& loss)
		{
			if (loss == "l1") return 1 / residual;
			if (loss == "huber") return P.k / MAX(P.k, residual); 
			if (loss == "cauchy") return 1 / (1 + pow(residual / P.k, 2));
			if (loss == "gm") return P.k / pow(P.k + pow(residual, 2), 2);
			if (loss == "tukey") return pow(1.0 - pow(MIN(1.0, abs(residual) / P.k), 2), 2);
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
		void jacobian1(Matrix::matrix* rot_p, Matrix::matrix* J);
		void point2point();
		void point2plane();

		~icp(){}
	};

	Matrix::matrix* principal_axis(Matrix::matrix* data);
	int getRotations(Matrix::matrix* model, Matrix::matrix* scene, Matrix::matrix* R, Matrix::matrix* R_eta);
	Matrix::matrix* alignment(Matrix::matrix* cloud, Matrix::matrix* R, double vertical_shift);
	int Align(utils::PointCloud* model, utils::PointCloud* scene, utils::PointCloud* scene_out);
}