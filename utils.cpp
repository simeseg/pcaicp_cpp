#include "utils.h"

using namespace utils;

utils::point utils::operator+(const utils::point& p1, const utils::point& p2)
{
	utils::point pt;
	pt.x = p1.x + p2.x;
	pt.y = p1.y + p2.y;
	pt.z = p1.z + p2.z;
	return pt;
}

utils::point utils::operator-(const utils::point& p1, const utils::point& p2)
{
	utils::point pt;
	pt.x = p1.x - p2.x;
	pt.y = p1.y - p2.y;
	pt.z = p1.z - p2.z;
	return pt;
}

utils::point utils::operator/=(utils::point& p1, const float& f)
{
	p1.x /= f;
	p1.y /= f;
	p1.z /= f;
	return p1;
}

void utils::PointCloud_mean(const utils::PointCloud& pointcloud, utils::point& mean)
{
	for (auto& pt : pointcloud.points) { mean = mean + pt; }
	mean /= (float) pointcloud.points.size();
}

Matrix::matrix* Matrix::newMatrix(int rows, int cols)
{
	if (rows <= 0 || cols <= 0) return NULL;
	matrix* m = (matrix*)malloc(sizeof(matrix));

	m->rows = rows;
	m->cols = cols;

	m->data = (double*)malloc(rows * cols * sizeof(double));
	for (int i = 0; i < rows * cols; i++) { m->data[i] = 0.0; }

	return m;
}

int Matrix::set(matrix* m, int row, int col, double val)
{
	if (!m) return -1;
	assert(m->data);
	if (row <= 0 || col <= 0 || row > m->rows || col > m->cols) { return -2; }
	m->data[(row - 1) * m->cols + (col - 1)] = val;
	return 0;
}

double Matrix::get(matrix* m, int row, int col)
{
	if (!m) return -1;
	assert(m->data);
	if (row <= 0 || col <= 0 || row > m->rows || col > m->cols) { return -2; }
	return m->data[(row - 1) * m->cols + (col - 1)];
}

Matrix::matrix* Matrix::identity(int size)
{
	Matrix::matrix* I = Matrix::newMatrix(size, size);
	for (int i = 1; i <= size; i++)
	{
		Matrix::set(I, i, i, 1);
	}
	return I;
}

int Matrix::setColumn(matrix* m, matrix* val, int col)
{
	for (int row = 1; row <= m->rows; row++) 
	{
		Matrix::set(m, row, col, Matrix::get(val, row, 1));
	}
	return 0;
}

int Matrix::setRow(matrix* m, matrix* val, int row)
{
	for (int col = 1; col <= m->cols; row++)
	{
		Matrix::set(m, row, col, Matrix::get(val, 1, col));
	}
	return 0;
}

Matrix::matrix* Matrix::getColumn(matrix* m, int col)
{
	if (!m) std::cerr << "matrix not initialized properly";
	Matrix::matrix* out = Matrix::newMatrix(m->rows, 1);
	for (int row = 1; row <= m->rows; row++)
	{
		Matrix::set(out, row, 1, Matrix::get(m, row, col));
	}
	return out;
}

Matrix::matrix* Matrix::getRow(matrix* m, int row)
{
	if (!m) std::cerr << "matrix not initialized properly";
	Matrix::matrix* out = Matrix::newMatrix(1, m->cols);
	for (int col = 1; col <= m->cols; col++)
	{
		Matrix::set(out, 1, col, Matrix::get(m, row, col));
	}
	return out;
}


int Matrix::print(matrix* m)
{
	if (!m) return -1;
	printf("\n\n");
	for (int row = 1; row <= m->rows; row++)
	{
		for (int col = 1; col <= m->cols; col++)
		{
			printf("%6.3f ", m->data[(row - 1) * m->cols + (col - 1)]);
		}
		printf("\n\n");
	}
	return 0;
}

Matrix::matrix* Matrix::transpose(matrix* in)
{
	if (!in) std::cerr << "matrix not initialized properly";
	Matrix::matrix* out = Matrix::newMatrix(in->cols, in->rows);

	for (int row = 1; row <= in->rows; row++)
	{
		for (int col = 1; col <= in->cols; col++)
		{
			out->data[(col - 1) * out->rows + (row - 1)] = in->data[(row - 1) * in->cols + (col - 1)];
		}
	}
	return out;
}

Matrix::matrix* Matrix::sum(matrix* m1 , matrix* m2)
{
	if (!m1 || !m2) std::cerr<<"matrix not initialized properly";
	if (m1->rows == m2->rows && m1->cols == m2->cols) std::cerr << "sum: matrix size mismatch.";;

	Matrix::matrix* sum = Matrix::newMatrix(m1->rows, m1->cols);
	for (int row = 1; row <= m1->rows; row++)
	{
		for (int col = 1; col <= m1->cols; col++)
		{
			sum->data[(row - 1) * m2->cols + (col - 1)] = m1->data[(row - 1) * m1->cols + (col - 1)] + m2->data[(row - 1) * m2->cols + (col - 1)];
		}
	}
	return sum;
}

Matrix::matrix* Matrix::diff(matrix* m1, matrix* m2)
{
	if (!m1 || !m2) std::cerr << "matrix not initialized properly";
	if (m1->rows != m2->rows || m1->cols != m2->cols) std::cerr << "Diff:: matrix size mismatch.";;

	Matrix::matrix* diff = Matrix::newMatrix(m1->rows, m1->cols);
	for (int row = 1; row <= m1->rows; row++)
	{
		for (int col = 1; col <= m1->cols; col++)
		{
			diff->data[(row - 1) * m2->cols + (col - 1)] = m1->data[(row - 1) * m1->cols + (col - 1)] - m2->data[(row - 1) * m2->cols + (col - 1)];
		}
	}
	return diff;
}


Matrix::matrix* Matrix::scalar_prod(matrix* in, double val)
{
	if (!in) std::cerr << "matrix not initialized properly";

	Matrix::matrix* out = Matrix::newMatrix(in->rows, in->cols);
	for (int row = 1; row <= in->rows; row++)
	{
		for (int col = 1; col <= in->cols; col++)
		{
			out->data[(row - 1) * out->cols + (col - 1)] = val * (in->data[(row - 1) * in->cols + (col - 1)]);
		}
	}
	return out;
}

Matrix::matrix* Matrix::power(matrix* A, int k)
{
	if (!A)std::cerr << "Power: matrix not properly initialized";
	if (k < 0) std::cerr << "negative exponent"; 
	if (k == 0) return Matrix::identity(A->rows);

	Matrix::matrix* P = Matrix::newMatrix(A->rows, A->cols);
	*P = *A;

	for(int i = 1; i < k; i++)
	{
		P = Matrix::product(A, P);
	}
	return P;
}

Matrix::matrix* Matrix::product(matrix* m1, matrix* m2)
{
	if (!m1||!m2) std::cerr << "matrix not initialized properly";
	if (m1->cols != m2->rows) std::cerr << "product: matrix size mismatch";

	Matrix::matrix* prod = Matrix::newMatrix(m1->rows, m2->cols);

	for (int row = 1; row <= m1->rows; row++)
	{
		for (int col = 1; col <= m2->cols; col++)
		{
			double val = 0.0;
			for (int k = 1; k <= m2->rows; k++)
			{
				val += m1->data[(row - 1) * m1->cols + (k - 1)] * m2->data[(k - 1) * m2->cols + (col - 1)];
			}
			prod->data[(row - 1) * prod->cols + (col - 1)] = val;
		}
	}
	return prod;
}

double Matrix::dotProduct(matrix* m1, matrix* m2)
{
	if (!m1 || !m2) std::cerr << "matrix not initialized properly";
	if (m1->rows != m2->rows || m1->cols != m2->cols) std::cerr<<"dot product: matrix size mismatch";

	double prod=0;
	for (int row = 1; row <= m1->rows; row++)
	{
		for (int col = 1; col <= m1->cols; col++)
		{
			prod += m1->data[(row - 1) * m1->cols + (col - 1)] * m2->data[(row - 1) * m2->cols + (col - 1)];
		}
	}
	return prod;
}

double Matrix::norm(matrix* m)
{
	if (!m) std::cerr << "matrix not initialized properly";
	if (m->rows != 1 && m->cols != 1) std::cerr << "Cannot compute for non-row or non-column matrix";

	double val=0;
	for (int row = 1; row <= m->rows; row++)
	{
		for (int col = 1; col <= m->cols; col++)
		{
			val += pow(m->data[(row - 1) * m->cols + (col - 1)], 2);
		}
	}
	if (val <= 0) { printf("Trivial norm"); return NULL; }
	return sqrt(val);
}

int Matrix::gram_schmidt(matrix* A, matrix* Q, matrix* R)
{
	if (!A || !Q || !R)return -1;
	if (Q->cols != R->rows || Q->rows != A->rows || Q->cols != A->cols)return -2;

	//Q matrix
	for (int col = 1; col <= A->cols; col++)
	{
		matrix* a = Matrix::getColumn(A, col);
		matrix* u = a;
		for (int k = 1; k < col; k++)
		{
			matrix* e = Matrix::getColumn(Q, k);
			matrix* p_k = Matrix::scalar_prod(e, Matrix::dotProduct(a, e));
			u = Matrix::diff(u, p_k);
		}
		matrix* e_new = Matrix::scalar_prod(u, (1/Matrix::norm(u)));
		Matrix::setColumn(Q, e_new, col);
	}

	//R matrix
	*R = *Matrix::product(Matrix::transpose(Q), A);
	return 0;
}

int Matrix::modified_gram_schmidt(matrix* A, matrix* Q, matrix* R)
{
	if (!A || !Q || !R)return -1;
	if (Q->cols != R->rows || Q->rows != A->rows || Q->cols != A->cols)return -2;

	matrix* U = newMatrix(A->rows, A->cols);
	for (int i = 1; i<=A->cols; i++)
	{
		setColumn(U, getColumn(A, i), i);
	}

	//Q matrix
	for (int col = 1; col <= A->cols; col++)
	{
		matrix* u = Matrix::getColumn(U, col);
		matrix* e_new = Matrix::scalar_prod(u, (1 / Matrix::norm(u)));
		Matrix::setColumn(Q, e_new, col);

		for (int k = col + 1; k <= A->cols; k++)
		{
			matrix* u = Matrix::getColumn(U, k);
			matrix* q = getColumn(Q, col);
			u = diff(u, scalar_prod(q, dotProduct(q, u)));
			setColumn(U, u, k);
		}
	}
	
	//R matrix
	*R = *Matrix::product(Matrix::transpose(Q), A);
	return 0;
}

int Matrix::householder(matrix* A, matrix* Q, matrix* R)
{
	matrix * I = identity(A->rows);

	int col = 1;
	matrix* a = getColumn(A, col);
	int sn = signbit(get(A, 1, col)); 
	matrix* v = diff(a, scalar_prod(getColumn(I,col), sn*norm(a)));

	if (norm(v) != 0)
	matrix* Qv = newMatrix(Q->rows, Q->cols);
    matrix* Qv = diff(identity(A->rows), scalar_prod(scalar_prod(product(v, transpose(v)), (1 / norm(v))), 2));

	*Q = *product(Qv, A);

	return 0;
}

Matrix::matrix* Matrix::diagonal(matrix* m)
{
	matrix* out = Matrix::identity(m->rows);
	for (int i = 1; i <= m->rows; i++)
	{
		Matrix::set(out, i, i, Matrix::get(m, i, i));
	}
	return out;
}

Matrix::matrix* Matrix::diagonal_inverse(matrix* m)
{
	matrix* out = Matrix::identity(m->rows);
	for (int i = 1; i <= m->rows; i++)
	{
		double val = Matrix::get(m, i, i);
		if (val != 0) Matrix::set(out, i, i, (1 / val));
		else continue;
	}
	return out;
}

int Matrix::solver(matrix* A, matrix* b, matrix* x)
{
	if (!A || !b || !x) return -1;
	if (A->cols != x->rows || A->rows != b->rows) return -2;
	
	//QR decomp of A
	matrix* Q = newMatrix(A->rows, A->cols);
	matrix* R = newMatrix(A->cols, A->cols);
	modified_gram_schmidt(A, Q, R);
	matrix* D = diagonal(R);

	print(product(transpose(Q), Q));

	//loop solve Rx = Q_T*b
	b = product(transpose(Q), b);
	for (int i = D->rows; i >= 1; i--)
	{
		int j; double sum = 0;
		for (int j = i+1; j <= D->cols; j++)
		{
			sum += get(R, i, j) * get(x, j, 1);
		}
		set(x, i, 1, (get(b, i, 1) - sum) / get(D, i, i));
	}
	return 0;
}

int Matrix::svd(matrix* m, matrix* U, matrix* S, matrix* VT)
{
	
}