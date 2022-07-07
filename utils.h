#pragma once
#include <vector>
#include <assert.h>
#include <math.h>
#include <iostream>

#define sgn(x) (x<0?-1:1);

namespace utils
{
	struct point
	{
		double x, y, z;
	};

	struct PointCloud
	{
		std::vector<point> points;
		size_t size = 0;
	};

	utils::point operator+(const utils::point& p1, const utils::point& p2);
	utils::point operator-(const utils::point& p1, const utils::point& p2);
	utils::point operator/=(utils::point& p1, const float& f);

	void PointCloud_mean(const utils::PointCloud& pointcloud, utils::point& mean);
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

	matrix* identity(int size);

	int setColumn(matrix* m, matrix* val, int col);
	
	int setRow(matrix* m, matrix* val, int row);

	matrix* getColumn(matrix* m, int col);

	matrix* getRow(matrix* m, int row);

	int print(matrix* m);

	matrix* transpose(matrix* in);

	matrix* sum(matrix* m1, matrix* m2);

	matrix* diff(matrix* m1, matrix* m2);

	matrix* scalar_prod(matrix* in, double val);

    matrix* product(matrix* m1, matrix* m2);

	matrix* power(matrix* A, int k);

	double dotProduct(matrix* m1, matrix* m2);

	double norm(matrix* m);
	
	int gram_schmidt(matrix* A, matrix* Q, matrix* R);

	int modified_gram_schmidt(matrix* A, matrix* Q, matrix* R);

	int householder(matrix* A, matrix* Q, matrix* R);

	matrix* diagonal(matrix* m);

	matrix* diagonal_inverse(matrix* m);

	int solver(matrix* A, matrix* b, matrix* x);

	int svd(matrix* m, matrix* U, matrix* S, matrix* VT);
}