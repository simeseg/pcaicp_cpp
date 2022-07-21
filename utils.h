#pragma once
#define _USE_MATH_DEFINES
#include <omp.h>
#include <vector>
#include <assert.h>
#include <cmath>
#include <iostream>
#include <numeric>
#include "include/nanoflann.hpp"

#define SGN(x) (((x)<(0))?(-1):(1))
#define MAX(a,b) (((a)>(b))?(a):(b))
#define MIN(a,b) (((a)<(b))?(a):(b))


namespace utils
{
	struct point
	{
		double x = 0, y = 0, z = 0;
		double nx = 0, ny = 0, nz = 0;
	};

	struct PointCloud
	{
		std::vector<point> points;
		int size = points.size();
	};

	utils::point operator+(const utils::point& p1, const utils::point& p2);
	utils::point operator-(const utils::point& p1, const utils::point& p2);
	utils::point operator/=(utils::point& p1, const float& f);

	void PointCloud_mean(const utils::PointCloud& pointcloud, utils::point& mean);
	void shiftByMean(utils::PointCloud* cloud, utils::point& mean);
	void computeNormals(utils::PointCloud* cloud, const size_t& n);
	void orientNormals(utils::PointCloud* cloud);

	//tree struct
	struct kdtree {
		utils::PointCloud* _cloud;
		int _n = 1;
		utils::PointCloudAdaptor pointcloudAdaptor(3, utils::PointCloud* _cloud, _n);
		kdtree(utils::PointCloud* cloud, const int& n) :_cloud(cloud), _n(n) {
			//make a kdtree
			utils::PointCloudAdaptor pointcloudAdaptor(3, *cloud, n);
			pointcloudAdaptor.index->buildIndex();
		}
	};

	//knn adaptor 
	struct PointCloudAdaptor
	{
		typedef nanoflann::metric_L2::traits<double, PointCloudAdaptor>::distance_t metric_t;
		typedef nanoflann::KDTreeSingleIndexAdaptor<metric_t, PointCloudAdaptor, -1> index_t;
		index_t* index;
		const utils::PointCloud& pointcloud;

		PointCloudAdaptor(const int dimensions, const utils::PointCloud& pointcloud, const int leaf_size_max = 10) : pointcloud(pointcloud)
		{
			assert(pointcloud.points.size() != 0);
			const size_t dims = 3;
			index = new index_t(dims, *this, nanoflann::KDTreeSingleIndexAdaptorParams(leaf_size_max));
			index->buildIndex();
		}
		~PointCloudAdaptor()
		{
			delete index;
		}

		inline PointCloudAdaptor& derived() 
		{
			return *this;
		}

		inline size_t kdtree_get_point_count() const
		{
			return pointcloud.points.size();
		}

		inline double kdtree_get_pt(const size_t idx, int dim) const
		{
			assert(dim < 3);
			if (dim == 0) return pointcloud.points.at(idx).x;
			if (dim == 1) return pointcloud.points.at(idx).y;
			if (dim == 2) return pointcloud.points.at(idx).z;
		}

		template <class BBOX>
		bool kdtree_get_bbox(BBOX& bbox) const
		{
			return true;
		}
	};

	// from Open3D
	struct DisjointSet
	{
		std::vector<size_t> parent_;
		std::vector<size_t> size_;
		DisjointSet(size_t size) : parent_(size), size_(size) 
		{
			for (size_t idx = 0; idx < size; idx++)
			{
				parent_[idx] = idx; 
				size_[idx] = 0;
			}
		}

		inline size_t Find(size_t x) {
			if (x != parent_[x]) {
				parent_[x] = Find(parent_[x]);
			}
			return parent_[x];
		}

		void Union(size_t x, size_t y)
		{
			x = Find(x); y = Find(y);
			if (x != y) {
				if (size_[x] < size_[y]) { size_[y] += size_[x]; parent_[x] = y; }
				else { size_[x] += size_[y]; parent_[y] = x;}
			}
		}
	};

	struct WeightedEdge
	{
		size_t v0;
		size_t v1;
		double weight;
		WeightedEdge(size_t v0, size_t v1, double weight)
			:v0(v0), v1(v1), weight(weight){}
	};
}
 
namespace Matrix
{
    

	//matrix definition from http://theory.stanford.edu/~arbrad/pfe/06/matrix.c

	struct matrix
	{
		int rows;
		int cols;
		double* data;
	};

	matrix* newMatrix(int rows, int cols);

	int set(matrix* m, int row, int col, double val);

	double get(matrix* m, int row, int col);

	int min(matrix* m, double* out, const bool& diag);

	int max(matrix* m, double* out, const bool& diag);

	matrix* identity(int size);

	int setColumn(matrix* m, matrix* val, int col);
	
	int setRow(matrix* m, matrix* val, int row);

	matrix* getColumn(matrix* m, int col);

	matrix* getRow(matrix* m, int row);

	matrix* getSub(matrix* m, int d);

	int setSub(matrix* m, matrix* in);

	matrix* sgn(matrix* m);

	int print(matrix* m);

	matrix* transpose(matrix* in);

	matrix* sum(matrix* m1, matrix* m2);

	matrix* diff(matrix* m1, matrix* m2);

	matrix* scalar_prod(matrix* in, double val);

    matrix* product(matrix* m1, matrix* m2);

	matrix* power(matrix* A, int k);

	double dotProduct(matrix* m1, matrix* m2);

	double norm(matrix* m);

	double trace(matrix* m);

	matrix* inverse(matrix* A);

	int eigendecomposition(matrix* A, matrix* Q, matrix* R);
	
	int gram_schmidt(matrix* A, matrix* Q, matrix* R);

	int modified_gram_schmidt(matrix* A, matrix* Q, matrix* R);

	int householder(matrix* A, matrix* Q, matrix* R);

	matrix* diagonal(matrix* m);

	matrix* diagonal_inverse(matrix* m);

	int solver(matrix* A, matrix* b, matrix* x);

	int svd(matrix* m, matrix* U, matrix* S, matrix* VT);
	
	matrix* skewSymmetric3D(matrix* a);

	void pcd2mat(utils::PointCloud* cloud, Matrix::matrix* mat);
	
}

