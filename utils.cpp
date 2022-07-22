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

void utils::shiftByMean(utils::PointCloud* cloud, utils::point& mean)
{
	utils::PointCloud_mean(*cloud, mean);
	for (auto& pt : cloud->points) pt = pt - mean;
}

void utils::computeNormals(utils::PointCloud* cloud, const utils::PointCloudAdaptor& pcadaptor, const size_t& num_neighbors)
{

#pragma omp parallel for
	for (int i = 0; i < cloud->points.size(); i++) //cloud->points.size(); i++)
	{
		//do a knn search
		utils::point pt = cloud->points.at(i);
		double query_point[] = { pt.x, pt.y, pt.z };
		std::vector<size_t> out_indices(num_neighbors);     //change to vertex later
		std::vector<double> out_distances(num_neighbors);   //change to vertex later
		
		nanoflann::KNNResultSet<double> resultSet(num_neighbors);
		resultSet.init(&out_indices[0], &out_distances[0]);
		pcadaptor.index->findNeighbors(resultSet, &query_point[0], int(num_neighbors));

		Matrix::matrix* N = Matrix::newMatrix(3, num_neighbors);

		utils::point mean;
		#pragma omp parallel for
		for (int idx = 1; idx <= num_neighbors; idx++)
		{
			mean = mean + cloud->points.at(out_indices[idx - 1]);
		}
		mean /= (double) num_neighbors;

		#pragma omp parallel for 
		for (int idx = 1; idx <= num_neighbors; idx++)
		{
			Matrix::set(N, 1, idx, cloud->points.at(out_indices[idx-1]).x - mean.x);
			Matrix::set(N, 2, idx, cloud->points.at(out_indices[idx-1]).y - mean.y);
			Matrix::set(N, 3, idx, cloud->points.at(out_indices[idx-1]).z - mean.z);
			//cloud->points.at(out_indices[idx - 1]).nz = 1.0;
		}
	print(N);
		Matrix::matrix* Cov = Matrix::scalar_prod(Matrix::product(N, Matrix::transpose(N)), 1/(double)num_neighbors);
		 std::cout << "get cov \n"; 
		Matrix::matrix* Q = Matrix::newMatrix(3, 3); 
		Matrix::matrix* R = Matrix::newMatrix(3, 3); 
		Matrix::matrix* E = Matrix::newMatrix(3, 1); 
		
		Matrix::eigendecomposition(Cov, Q, R);
		double minEval[3]; Matrix::min(Cov, minEval, true);
		Matrix::matrix* norm = getColumn(Q, minEval[2]);
		
		pt.nx = Matrix::get(norm, 1, 1);
		pt.ny = Matrix::get(norm, 2, 1);
		pt.nz = Matrix::get(norm, 3, 1);
		cloud->points.at(i) = pt;
	}
}

// Make Minimum Spanning Tree(Kruskal's algorithm) to create a simple connected graph for the 
// set of points. (with no cycles and minimum total edge weight) But since MST is not 
// sufficiently dense in edges, so we add the remaining edges to the graph to make a Reimannian graph
// Flipping order based on edge weights (wij = 1 - |ni.nj|) to avoid problem at high curvature eg sharp edges
// Traverse the MST and flip all points in the node neighbourhood based on the edge weights

std::vector<WeightedEdge> Kruskal(std::vector<WeightedEdge>& edges, size_t n_vertices) 
{
	std::sort(edges.begin(), edges.end(), [](WeightedEdge& e0, WeightedEdge& e1){return e0.weight < e1.weight;});

	DisjointSet disjoint_set(n_vertices);

	std::vector<WeightedEdge> mst;

	for (size_t eidx = 0; eidx < edges.size(); ++eidx) 
	{
		size_t set0 = disjoint_set.Find(edges[eidx].v0);
		size_t set1 = disjoint_set.Find(edges[eidx].v1);
		if (set0 != set1) 
		{
			mst.push_back(edges.at(eidx));
			disjoint_set.Union(set0, set1);
		}
	}
	return mst;
}

void utils::orientNormals(utils::PointCloud* cloud)
{
	//depth first search 
	//make edges
	//make Euclidean MST with distance weights
	//add remaining edges to the graph

	std::vector<WeightedEdge> mst;
}

void Matrix::pcd2mat(utils::PointCloud* cloud, Matrix::matrix* mat)
{
#pragma omp parallel for 
	for (int i = 1; i <= cloud->points.size(); i++) 
	{
		Matrix::set(mat, 1, i, cloud->points.at(i-1).x);
		Matrix::set(mat, 2, i, cloud->points.at(i-1).y);
		Matrix::set(mat, 3, i, cloud->points.at(i-1).z);
	}
}


Matrix::matrix* Matrix::newMatrix(int rows, int cols)
{
	if (rows <= 0 || cols <= 0) return NULL;
	matrix* m = (matrix*)malloc(sizeof(matrix));

	m->rows = rows; m->cols = cols;

	m->data = (double*)malloc(rows * cols * sizeof(double));

#pragma omp parallel for
	for (int i = 0; i < rows * cols; i++) { m->data[i] = 0.0; }

	return m;
}

int Matrix::set(matrix* m, int row, int col, double val)
{
	if (!m) std::cerr<<"error set function"; return -1;
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

int Matrix::min(matrix* m, double* out, const bool& diag)
{
	if (!m) std::cerr << "minimum element: bad initialization \n"; return NULL;
	out[0] = get(m, 1, 1); out[1] = 1; out[2] = 1;

	if (diag == false)
	{
		#pragma omp parallel for collapse(2)
		for (int row = 2; row <= m->rows; row++)
		{
			for (int col = 2; col <= m->cols; col++)
			{
				double val = get(m, row, col);
				if (val < out[0])
				{
					out[0] = val; out[1] = row; out[2] = col;
				}
			}
		}
	}

	if (diag == true)
	{
	#pragma omp parallel for 
		for (int idx = 2; idx <= m->rows; idx++)
		{
			double val = get(m, idx, idx);
			if (val < out[0])
			{
				out[0] = val; out[1] = idx; out[2] = idx;
			}
		}
	}
	return 0;
}

int Matrix::max(matrix* m, double* out, const bool& diag)
{
	out[0] = get(m, 1, 1); out[1] = 1; out[2] = 1;

	if (diag == false)
	{
	#pragma omp parallel for collapse(2)
		for (int row = 2; row <= m->rows; row++)
		{
			for (int col = 2; col <= m->cols; col++)
			{
				double val = get(m, row, col);
				if (val > out[0])
				{
					out[0] = val; out[1] = row; out[2] = col;
				}
			}
		}
	}

	if (diag == true)
	{
	#pragma omp parallel for 
		for (int idx = 2; idx <= m->rows; idx++)
		{
			double val = get(m, idx, idx);
			if (val > out[0])
			{
				out[0] = val; out[1] = idx; out[2] = idx;
			}
		}
	}
	return 0;
}

Matrix::matrix* Matrix::identity(int size)
{
	Matrix::matrix* I = Matrix::newMatrix(size, size);
#pragma omp parallel for 
	for (int i = 1; i <= size; i++)
	{
		Matrix::set(I, i, i, 1);
	}
	return I;
}

int Matrix::setColumn(matrix* m, matrix* val, int col)
{
#pragma omp parallel for 
	for (int row = 1; row <= m->rows; row++) 
	{
		Matrix::set(m, row, col, Matrix::get(val, row, 1));
	}
	return 0;
}

int Matrix::setRow(matrix* m, matrix* val, int row)
{
#pragma omp parallel for 
	for (int col = 1; col <= m->cols; col++)
	{
		Matrix::set(m, row, col, Matrix::get(val, 1, col));
	}
	return 0;
}

Matrix::matrix* Matrix::getColumn(matrix* m, int col)
{
	if (!m) std::cerr << "getColumn: matrix not initialized properly \n"; return NULL;
	Matrix::matrix* out = Matrix::newMatrix(m->rows, 1);

#pragma omp parallel for 
	for (int row = 1; row <= m->rows; row++)
	{
		Matrix::set(out, row, 1, Matrix::get(m, row, col));
	}
	return out;
}

Matrix::matrix* Matrix::getRow(matrix* m, int row)
{
	if (!m) std::cerr << "getRow: matrix not initialized properly \n"; return NULL;
	Matrix::matrix* out = Matrix::newMatrix(1, m->cols);

#pragma omp parallel for 
	for (int col = 1; col <= m->cols; col++)
	{
		Matrix::set(out, 1, col, Matrix::get(m, row, col));
	}
	return out;
}

Matrix::matrix* Matrix::getSub(matrix* m, int d)
{
	if (!m) std::cerr << "getsub: matrix not initialized properly \n"; return NULL;
	matrix* out = newMatrix(m->rows - d + 1, m->cols - d + 1);

#pragma omp parallel for collapse(2)
	for (int row = d; row <= m->rows; row++)
	{
		for (int col = d; col <= m->cols; col++)
		{
			set(out, row - d + 1 , col - d + 1, get(m, row, col));
		}
	}
	return out;
}

int Matrix::setSub(matrix* m, matrix* in)
{
	if (!m) std::cerr << "setsub: matrix m not initialized properly \n"; return NULL;
	if (!m) std::cerr << "setsub: matrix in not initialized properly \n"; return NULL;
	double d = m->rows - in->rows + 1;

#pragma omp parallel for collapse(2)
	for (int row = d; row <= m->rows; row++)
	{
		for (int col = d; col <= m->cols; col++)
		{
			set(m, row, col, get(in, row - d + 1, col - d + 1));
		}
	}
	return 0;
}


int Matrix::print(matrix* m)
{
	if (!m) std::cerr << "print error";  return -1;
	printf("\n\n");

	for (int row = 1; row <= m->rows; row++)
	{
		for (int col = 1; col <= m->cols; col++)
		{
			printf("%6.4f ", get(m, row, col));
		}
		printf("\n\n");
	}
	return 0;
}

Matrix::matrix* Matrix::transpose(matrix* in)
{
	if (!in) std::cerr << "transpose: matrix not initialized properly \n"; //return NULL;
	Matrix::matrix* out = Matrix::newMatrix(in->cols, in->rows);

#pragma omp parallel for collapse(2)
	for (int row = 1; row <= in->rows; row++)
	{
		for (int col = 1; col <= in->cols; col++)
		{
			set(out, col, row, get(in, row, col));
		}
	}
	return out;
}

Matrix::matrix* Matrix::sum(matrix* m1 , matrix* m2)
{
	if (!m1 || !m2) std::cerr<<"sum: matrix not initialized properly \n";  return NULL;
	if (m1->rows != m2->rows || m1->cols != m2->cols) std::cerr << "sum: matrix size mismatch \n"; return NULL;

	Matrix::matrix* sum = Matrix::newMatrix(m1->rows, m1->cols);

#pragma omp parallel for collapse(2)
	for (int row = 1; row <= m1->rows; row++)
	{
		for (int col = 1; col <= m1->cols; col++)
		{
			set(sum, row, col, get(m1, row, col) + get(m2, row, col));
		}
	}
	return sum;
}

Matrix::matrix* Matrix::diff(matrix* m1, matrix* m2)
{
	if (!m1 || !m2) std::cerr << "Diff: matrix not initialized properly \n";  return NULL;
	if (m1->rows != m2->rows || m1->cols != m2->cols) std::cerr << "Diff: matrix size mismatch \n";  return NULL;

	Matrix::matrix* diff = Matrix::newMatrix(m1->rows, m1->cols);

#pragma omp parallel for collapse(2)
	for (int row = 1; row <= m1->rows; row++)
	{
		for (int col = 1; col <= m1->cols; col++)
		{
			set(diff, row, col, get(m1, row, col) - get(m2, row, col));
		}
	}
	return diff;
}


Matrix::matrix* Matrix::scalar_prod(matrix* in, double val)
{
	if (!in) std::cerr << "scalar product: matrix not initialized properly \n"; return NULL;

	Matrix::matrix* out = Matrix::newMatrix(in->rows, in->cols);

#pragma omp parallel for collapse(2)
	for (int row = 1; row <= in->rows; row++)
	{
		for (int col = 1; col <= in->cols; col++)
		{
			 set(out, row, col, val * get(in, row, col));
		}
	}
	return out;
}

Matrix::matrix* Matrix::power(matrix* A, int k)
{
	if (!A)std::cerr << "Power: matrix not properly initialized \n"; // return NULL;
	if (k < 0) std::cerr << "negative exponent";  return NULL;
	if (k == 0) return Matrix::identity(A->rows);

	Matrix::matrix* P = Matrix::newMatrix(A->rows, A->cols);
	*P = *A;
#pragma omp parallel 
	for(int i = 1; i < k; i++)
	{
		P = Matrix::product(A, P);
	}
	return P;
}

Matrix::matrix* Matrix::product(matrix* m1, matrix* m2)
{
	if (!m1||!m2) std::cerr << "product: matrix not initialized properly \n";
	if (m1->cols != m2->rows) std::cerr << "product: matrix size mismatch \n";  

	Matrix::matrix* prod = Matrix::newMatrix(m1->rows, m2->cols);

#pragma omp parallel for collapse(2) 
	for (int row = 1; row <= m1->rows; row++)
	{
		for (int col = 1; col <= m2->cols; col++)
		{
			double val = 0.0;
#pragma omp parallel for reduction(+:val)
			for (int k = 1; k <= m2->rows; k++)
			{
				val += get(m1, row, k) * get(m2, k, col);
			}
			set(prod, row, col, val);
		}
	}
	return prod;
}

double Matrix::dotProduct(matrix* m1, matrix* m2)
{
	if (!m1 || !m2) std::cerr << "dotProduct: matrix not initialized properly \n";  return NULL;
	if (m1->rows != m2->rows || m1->cols != m2->cols) std::cerr<<"dot product: matrix size mismatch \n";  return NULL;

	double prod=0;
#pragma omp parallel for collapse(2) reduction(+:prod)
	for (int row = 1; row <= m1->rows; row++)
	{
		for (int col = 1; col <= m1->cols; col++)
		{
			prod += get(m1, row, col) * get(m2, row, col);
		}
	}
	return prod;
}

double Matrix::norm(matrix* m)
{
	if (!m) std::cerr << "Norm: matrix not initialized properly \n";  return NULL;
	if (m->rows != 1 && m->cols != 1) std::cerr << "Norm: cannot compute for non-row or non-column matrix \n";  return NULL;

	double val=0;
#pragma omp parallel for collapse(2) reduction(+:val)
	for (int row = 1; row <= m->rows; row++)
	{
		for (int col = 1; col <= m->cols; col++)
		{
			val += pow(get(m, row, col), 2);
		}
	}
	if (val <= 0) return 0;
	return sqrt(val);
}

double Matrix::trace(matrix* m)
{
	if (m->rows != m->cols) std::cerr << "Trace: Not square matrix \n";  return NULL;

	double val = 0;
#pragma omp parallel for reduction(+:val)

	for (int row = 1; row <= m->rows; row++)
	{
			val += get(m, row, row);
	}
	return val;
}

Matrix::matrix* Matrix::inverse(matrix* A)
{
	matrix* out = newMatrix(A->rows, A->cols);
	return out;
}

int Matrix::eigendecomposition(matrix* A, matrix* Q, matrix* R)
{
	if (!A || !Q || !R) std::cerr << "eigendecomposition: bad initialization \n"; return NULL;
	//QR
	*Q = *Matrix::identity(Q->rows); matrix* Q_f = Matrix::identity(Q->rows);
	for(int i = 0; i < 30; i++)
	{
		Matrix::householder(A, Q, R); *A = *product(R, Q); *Q_f = *product(Q_f, Q);
	}
	*Q = *Q_f;
	return 0;
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

Matrix::matrix* Matrix::sgn(matrix* m)
{
	matrix* sign = newMatrix(m->rows, m->cols);
	for (int row = 1; row <= m->rows; row++)
	{
		for (int col = 1; col <= m->cols; col++)
		{
			double val = get(m, row, col);
			if (val == 0) set(sign, row, col, 0);
			else set(sign, row, col, val / abs(val));
		}
	}
	return sign;
}

int Matrix::householder(matrix* A, matrix* Q, matrix* R)
{
	if (!A || !Q || !R)std::cerr << "householder: bad initialization \n"; return -1;
	if (Q->cols != R->rows || Q->rows != A->rows || Q->cols != A->cols)return -2;

	*R = *A;  *Q = *identity(A->rows);

	for (int col = 1; col < A->cols; col++) {

		matrix* H = identity(A->rows);
		matrix* a = getColumn(getSub(R, col), 1);
		matrix* e = newMatrix(a->rows, 1); set(e, 1, 1, 1);
		double a1 = get(a, 1, 1);
		matrix* u = sum(a, scalar_prod(e, norm(a) * SGN(a1))); 
		double u1 = get(u, 1, 1); matrix* v;
        print(R);
		if (u1 == 0) v = scalar_prod(u, 0);
		else v = scalar_prod(u, (1 / u1));
		double vn = norm(v);
		double beta = (2 / pow(vn, 2));
		if (vn == 0) { beta = 0;}
		setSub(H, diff(identity(a->rows), scalar_prod(product(v, transpose(v)), beta)));
		
		//update Q and R
		//get Q
		*Q = *product(Q, H);
		//get R
		*R = *product(H, R);
	}
	return 0;
}

Matrix::matrix* Matrix::diagonal(matrix* m)
{
	matrix* out = Matrix::identity(m->rows);
#pragma omp parallel for 
	for (int i = 1; i <= m->rows; i++)
	{
		if(m->cols ==1) Matrix::set(out, i, i, Matrix::get(m, i, 1));
		else Matrix::set(out, i, i, Matrix::get(m, i, i));
	}
	return out;
}

Matrix::matrix* Matrix::diagonal_inverse(matrix* m)
{
	matrix* out = Matrix::identity(m->rows);
#pragma omp parallel for
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
	householder(A, Q, R);
	matrix* D = diagonal(R);

	//loop solve Rx = Q_T*b
	b = product(transpose(Q), b);
	for (int i = D->rows; i >= 1; i--)
	{
		int j; double sum = 0;
		#pragma omp parallel for reduction(+:sum)
		for (int j = i+1; j <= D->cols; j++)
		{
			sum += get(R, i, j) * get(x, j, 1);
		}
		set(x, i, 1, (get(b, i, 1) - sum) / get(D, i, i));
	}
	return 0;
}

Matrix::matrix* Matrix::skewSymmetric3D(matrix* a)
{
	matrix* out = newMatrix(3, 3);
	double x = 0, y = 0, z = 0; x = get(a, 1, 1); y = get(a, 2, 1); z = get(a, 3, 1);
	set(out, 1, 1, 0); set(out, 1, 2, -z); set(out, 1, 3, y);
	set(out, 2, 1, z); set(out, 2, 2, 0); set(out, 2, 3, -x);
	set(out, 3, 1, -y); set(out, 3, 2, x); set(out, 3, 3, 0);
	return out;
}

void Matrix::euler2rot(Matrix::matrix* euler, Matrix::matrix* rot, Matrix::matrix* t)
{
	double alpha = get(euler, 1, 1);
	matrix* Rx = identity(3); double rx[] = { cos(alpha), sin(alpha), 0, -sin(alpha), cos(alpha), 0, 0, 0, 1 }; Rx->data = rx;

	*rot = *product(rot, Rx);
	delete Rx, rx;

	double beta = get(euler, 2, 1);
	matrix* Ry = identity(3); double ry[] = { cos(beta), 0, -sin(beta), 0, 1, 0, sin(beta), 0, cos(beta) }; Ry->data = ry;
	*rot = *product(rot, Ry);
	delete Ry, ry;

	double gamma = get(euler, 3, 1);
	matrix* Rz = identity(3); double rz[] = { cos(gamma), sin(gamma), 0, -sin(gamma), cos(gamma), 0, 0, 0, 1 }; Rz->data = rz;
	*rot = *product(rot, Rz);
	delete Rz, rz;

	set(t, 1, 1, get(euler, 4, 1));
	set(t, 2, 1, get(euler, 5, 1));
	set(t, 3, 1, get(euler, 6, 1));
}

void Matrix::updateFromEuler(Matrix::matrix* deltaEuler, Matrix::matrix* rot, Matrix::matrix* t)
{
	matrix* deltaR = newMatrix(3, 3); matrix* deltaT = newMatrix(3, 1);
	euler2rot(deltaEuler, deltaR, deltaT);
	*rot = *product(deltaR, rot);
	*t = *sum(product(deltaR, t), deltaT);
	delete deltaR, deltaT;
}

int Matrix::svd(matrix* m, matrix* U, matrix* S, matrix* VT)
{
	return 0;
}
