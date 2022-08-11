#include "utils.h"
#include <valarray>
#include <forward_list>

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

void utils::writepcd(const std::string& filename, const utils::PointCloud& cloud)
{
	std::ofstream out(filename);
	for (int i = 0; i < cloud.points.size(); i++)
	{
		utils::point pt = cloud.points.at(i);
		out << pt.x << "," << pt.y << "," << pt.z << "," << pt.nx << "," << pt.ny << "," << pt.nz << "\n";
	}
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

		utils::point mean;
		#pragma omp parallel for
		for (int idx = 1; idx <= num_neighbors; idx++)
		{
			mean = mean + cloud->points.at(out_indices[idx - 1]);
		}
		mean /= (double) num_neighbors;

		Matrix::matrix* N = new Matrix::matrix(3, num_neighbors);

		#pragma omp parallel for 
		for (int idx = 1; idx <= num_neighbors; idx++)
		{
			Matrix::set(N, 1, idx, cloud->points.at(out_indices[idx-1]).x - mean.x);
			Matrix::set(N, 2, idx, cloud->points.at(out_indices[idx-1]).y - mean.y);
			Matrix::set(N, 3, idx, cloud->points.at(out_indices[idx-1]).z - mean.z);
			//cloud->points.at(out_indices[idx - 1]).nz = 1.0;
		}
		Matrix::matrix* Cov = Matrix::scalar_prod(Matrix::product(N, Matrix::transpose(N)), 1/(double)(num_neighbors-1));  
		Matrix::matrix* Q = new Matrix::matrix(3, 3);
		Matrix::matrix* R = new Matrix::matrix(3, 3);
		Matrix::matrix* E = new Matrix::matrix(3, 1);
		
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

void utils::statisticalOutlierRemoval(utils::PointCloud* cloud, utils::PointCloud* cloudout, size_t nb_neighbors, double std_ratio)
{
	//make a kdtree
	utils::PointCloudAdaptor Adaptor(3, *cloud, 10);
	Adaptor.index->buildIndex();

	std::vector<double> average_distances; average_distances.resize(cloud->points.size());
	std::vector<int> indices;

	int valid_distances = 0;

#pragma omp parallel for reduction(+: valid_distances) schedule(static)
	for (int i = 0; i < cloud->points.size(); i++)
	{
		
		utils::point pt; pt = cloud->points.at(i);
		double query_point[] = { pt.x, pt.y, pt.z };
		std::vector<size_t> temp_indices(nb_neighbors);
		std::vector<double> dist(nb_neighbors);
		nanoflann::KNNResultSet<double> resultSet(nb_neighbors);
		resultSet.init(&temp_indices[0], &dist[0]);
		Adaptor.index->findNeighbors(resultSet, &query_point[0], nb_neighbors); 
		
		double mean = 0.0;
		valid_distances++;
		std::for_each(dist.begin(), dist.end(), [](double &d) {d = sqrt(d); });
		mean = std::accumulate(dist.begin(), dist.end(), 0.0) / (double)dist.size();
		average_distances[i] = mean;
	}

	//calculate mean of averages
	double cloud_average = std::accumulate(average_distances.begin(), average_distances.end(), 0.0) / (double)average_distances.size();
	//calculate std  
	double stdev = 0;  for (auto& avg : average_distances) stdev += pow(avg - cloud_average, 2); stdev = sqrt(stdev / (double)(valid_distances - 1));

	for (int i = 0 ; i< cloud->points.size(); i++) 
	{
		if (average_distances[i] > 0 && average_distances[i] < cloud_average + std_ratio * stdev) cloudout->points.push_back(cloud->points.at(i));
	}

}

int utils::normalinversion(utils::PointCloud* cloud)
{

	using value_type = double;

	std::list<std::forward_list<double>> points_;
	for(auto& pt:cloud->points)
	{
		std::forward_list<double> in{ pt.x, pt.y, pt.z, (pow(pt.x, 2) + pow(pt.y, 2) + pow(pt.z, 2)) };
		points_.push_back(in);
	}

	//using value_type = double;
	//using point = std::forward_list< value_type >;
	//using points = std::forward_list< point >;

	//value_type const eps = std::numeric_limits< value_type >::epsilon();

	// define and setup QH class instance
	//using quick_hull_type = quick_hull< typename points::const_iterator >;
	//quick_hull_type quick_hull_(4, eps); // (1)
	//quick_hull_.add_points(std::cbegin(points_), std::cend(points_)); // (2)



	int i;
	for (i = 0; i < cloud->points.size(); i++)
	{
		double d = sqrt(pow(cloud->points.at(i).x, 2) + pow(cloud->points.at(i).y, 2));
		utils::point* temp = new utils::point; 
		temp->x = cloud->points.at(i).x - d * cloud->points.at(i).nx; 
		temp->y = cloud->points.at(i).y - d * cloud->points.at(i).ny;
		double d_temp = sqrt(pow(temp->x, 2) + pow(temp->y, 2));
		if (d_temp > d) 
		{
			cloud->points.at(i).nx = -1 * cloud->points.at(i).nx; 
			cloud->points.at(i).ny = -1 * cloud->points.at(i).ny; 
			cloud->points.at(i).nz = -1 * cloud->points.at(i).nz;
		}
	}
	return 0;
}


void Matrix::getPosition(utils::PointCloud* cloud, Matrix::matrix* mat)
{
#pragma omp parallel for 
	for (int i = 1; i <= cloud->points.size(); i++) 
	{
		Matrix::set(mat, 1, i, cloud->points.at(i - 1).x);
		Matrix::set(mat, 2, i, cloud->points.at(i - 1).y);
		Matrix::set(mat, 3, i, cloud->points.at(i - 1).z);
	}
}

void Matrix::getNormals(utils::PointCloud* cloud, Matrix::matrix* mat)
{
#pragma omp parallel for 
	for (int i = 1; i <= cloud->points.size(); i++)
	{
		Matrix::set(mat, 1, i, cloud->points.at(i - 1).nx);
		Matrix::set(mat, 2, i, cloud->points.at(i - 1).ny);
		Matrix::set(mat, 3, i, cloud->points.at(i - 1).nz);
	}
}


int Matrix::set(matrix* m, int row, int col, double val)
{
	if (!m) { std::cerr << "error set function \n"; return -1; }
	//assert(m->data);
	if (row <= 0 || col <= 0 || row > m->rows || col > m->cols) { return -2; }
	m->data.at((row - 1) * m->cols + (col - 1)) = val;
	return 0;
}

double Matrix::get(matrix* m, int row, int col)
{
	if (!m) { return -1; }
	//assert(m->data);
	if (row <= 0 || col <= 0 || row > m->rows || col > m->cols) { return -2; }
	return m->data.at((row - 1) * m->cols + (col - 1));
}

int Matrix::min(matrix* m, double* out, const bool& diag)
{
	if (!m) { std::cerr << "minimum element: bad initialization \n"; return -1;}
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
	Matrix::matrix* I = new Matrix::matrix(size, size);
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
	for (int row = 1; row <= val->rows; row++) 
	{
		Matrix::set(m, row, col, Matrix::get(val, row, 1));
	}
	return 0;
}

int Matrix::setRow(matrix* m, matrix* val, int row)
{
#pragma omp parallel for 
	for (int col = 1; col <= val->cols; col++)
	{
		Matrix::set(m, row, col, Matrix::get(val, 1, col));
	}
	return 0;
}

Matrix::matrix* Matrix::getColumn(matrix* m, int col)
{
	if (!m) { std::cerr << "getColumn: matrix not initialized properly \n"; return NULL; }
	Matrix::matrix* out = new Matrix::matrix(m->rows, 1);

#pragma omp parallel for 
	for (int row = 1; row <= m->rows; row++)
	{
		Matrix::set(out, row, 1, Matrix::get(m, row, col));
	}
	return out;
}

Matrix::matrix* Matrix::getRow(matrix* m, int row)
{
	if (!m) { std::cerr << "getRow: matrix not initialized properly \n"; return NULL; }
	Matrix::matrix* out = new Matrix::matrix(1, m->cols);

#pragma omp parallel for 
	for (int col = 1; col <= m->cols; col++)
	{
		Matrix::set(out, 1, col, Matrix::get(m, row, col));
	}
	return out;
}

Matrix::matrix* Matrix::getSub(matrix* m, int r, int c)
{
	if (!m) { std::cerr << "getsub: matrix not initialized properly \n"; return NULL; }
	matrix* out = new matrix(m->rows - r + 1, m->cols - c + 1);

#pragma omp parallel for collapse(2)
	for (int row = r; row <= m->rows; row++)
	{
		for (int col = c; col <= m->cols; col++)
		{
			set(out, row - r + 1, col - c + 1, get(m, row, col));
		}
	}
	return out;
}

Matrix::matrix* Matrix::getSub(matrix* m, int r, int c, int R, int C)
{
	if (!m) { std::cerr << "getsub: matrix not initialized properly \n"; return NULL; }
	matrix* out = new matrix(R - r + 1, C - c + 1);

#pragma omp parallel for collapse(2)
	for (int row = r; row <= R; row++)
	{
		for (int col = c; col <= C; col++)
		{
			set(out, row - r + 1 , col - c + 1, get(m, row, col));
		}
	}
	return out;
}

int Matrix::setSub(matrix* m, matrix* in, int r, int c)
{
	if (!m) { std::cerr << "setsub: matrix m not initialized properly \n"; return -1; }
	if (in->rows + r -  1> m->rows || in->cols + c - 1> m->cols) { std::cerr <<r<<" "<<c<<" "<< m->rows << " " << m->cols << " " << in->rows << " " << in->cols << " setsub: matrix cannot be applied \n"; return -1; }

#pragma omp parallel for collapse(2)
	for (int row = r; row <= in->rows + r - 1; row++)
	{
		for (int col = c; col <= in->cols + c - 1; col++)
		{
			set(m, row, col, get(in, row - r + 1, col - c + 1));
		}
	}
	return 0;
}


int Matrix::print(matrix* m)
{
	
	if (!m) { std::cerr << "print error \n"; return -1; }
	printf("\n\n");

	for (int row = 1; row <= m->rows; row++)
	{
		for (int col = 1; col <= m->cols; col++)
		{
			printf("%6.7f ", get(m, row, col));
		}
		printf("\n\n");
	}
	return 0;
}

Matrix::matrix* Matrix::transpose(matrix* in)
{
	if (!in) {std::cerr << "transpose: matrix not initialized properly \n"; return NULL;}
	Matrix::matrix* out = new Matrix::matrix(in->cols, in->rows);

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
	if (!m1 || !m2) { std::cerr << "sum: matrix not initialized properly \n"; return NULL; }
	if (m1->rows != m2->rows || m1->cols != m2->cols) { std::cerr << "sum: matrix size mismatch \n"; return NULL; } 
	Matrix::matrix* sum = new Matrix::matrix(m1->rows, m1->cols);

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
	if (!m1 || !m2) { std::cerr << "Diff: matrix not initialized properly \n";  return NULL; }
	if (m1->rows != m2->rows || m1->cols != m2->cols) { std::cerr << "Diff: matrix size mismatch \n";  return NULL; }

	Matrix::matrix* diff = new Matrix::matrix(m1->rows, m1->cols);

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
	if (!in) { std::cerr << "scalar product: matrix not initialized properly \n"; return NULL; } 

	Matrix::matrix* out = new Matrix::matrix(in->rows, in->cols);

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


Matrix::matrix* Matrix::product(matrix* m1, matrix* m2)
{
	if (!m1 || !m2) { std::cerr << "product: matrix not initialized properly \n"; return NULL; }
	if (m1->cols != m2->rows) { std::cerr << "product: matrix size mismatch  \n"; return NULL; }

	Matrix::matrix* prod = new Matrix::matrix(m1->rows, m2->cols);

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


Matrix::matrix* Matrix::power(matrix* A, int k)
{
	if (!A) { std::cerr << "Power: matrix not properly initialized \n"; return NULL; }
	if (A->rows != A->cols) { std::cerr << "Power: not square matrix \n"; return NULL; }
	if (k < 0) { std::cerr << "negative exponent";  return NULL; }
	if (k == 0) return Matrix::identity(A->rows);

	/*
	* //iterative
	Matrix::matrix* P = new Matrix::matrix(A->rows, A->cols);
	*P = *A;
	for(int i = 2; i < k; i++)
		*P = *Matrix::product(A, P);
	*/
	//recursive
	Matrix::matrix* P = Matrix::power(A, k / 2);
	if (k & 1) { return Matrix::product(A, Matrix::product(P, P)); }

	return Matrix::product(A, A);
}

double Matrix::dotProduct(matrix* m1, matrix* m2)
{
	if (!m1 || !m2) { std::cerr << "dotProduct: matrix not initialized properly \n";  return NULL; }
	if (m1->rows != m2->rows || m1->cols != m2->cols) { std::cerr << "dot product: matrix size mismatch \n";  return NULL; }

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
	if (!m) { std::cerr << "Norm: matrix not initialized properly \n";  return NULL; }
	if (m->rows != 1 && m->cols != 1) { std::cerr << "Norm: cannot compute for non-row or non-column matrix \n";  return NULL; }

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
	if (m->rows != m->cols) { std::cerr << "Trace: Not square matrix \n";  return NULL; }

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
	matrix* out = new matrix(A->rows, A->cols);
	return out;
}

int Matrix::eigendecomposition(matrix* A, matrix* Q, matrix* R)
{
	if (!A || !Q || !R) { std::cerr << "eigendecomposition: bad initialization \n"; return NULL; }

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
	if (!A || !Q || !R) { return -1; }
	if (Q->cols != R->rows || Q->rows != A->rows || Q->cols != A->cols) { return -2; }

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
	if (!A || !Q || !R) { return -1; }
	if (Q->cols != R->rows || Q->rows != A->rows || Q->cols != A->cols) { return -2; }

	matrix* U = new matrix(A->rows, A->cols);
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
	matrix* sign = new matrix(m->rows, m->cols);
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
	if (!A || !Q || !R) { std::cerr << "householder: bad initialization \n"; return -1; }
	if (Q->cols != R->rows || Q->rows != A->rows || Q->cols != A->cols) { return -2; }

	*R = *A;  *Q = *identity(A->rows);

	for (int col = 1; col < A->cols; col++) {

		matrix* H = identity(A->rows);
		matrix* a = getColumn(getSub(R, col, col), 1);
		matrix* e = new matrix(a->rows, 1); set(e, 1, 1, 1);
		double a1 = get(a, 1, 1);
		matrix* u = sum(a, scalar_prod(e, norm(a) * SGN(a1))); 
		double u1 = get(u, 1, 1); matrix* v;
		if (u1 == 0) v = scalar_prod(u, 0);
		else v = scalar_prod(u, (1 / u1));
		double vn = norm(v);
		double beta = (2 / pow(vn, 2));
		if (vn == 0) { beta = 0;}
		setSub(H, diff(identity(a->rows), scalar_prod(product(v, transpose(v)), beta)), A->rows - a->rows + 1, A->rows - a->rows + 1);
		
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
	if (!m) { std::cerr << "diagonal: matrix not initialized properly \n";  return NULL; }

	matrix* out = Matrix::identity(m->rows);
#pragma omp parallel for 
	for (int i = 1; i <= MIN(m->rows, m->cols); i++)
	{
		if(m->cols ==1) Matrix::set(out, i, i, Matrix::get(m, i, 1));
		else Matrix::set(out, i, i, Matrix::get(m, i, i));
	}
	return out;
}

Matrix::matrix* Matrix::diagonal_inverse(matrix* m)
{
	if (!m) { std::cerr << "diagnal inverse: matrix not initialized properly \n";  return NULL; }
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

bool Matrix::issquare(matrix* m)
{
	return m->rows == m->cols;
}

double Matrix::determinant(matrix* m)
{
	if (!m) { std::cerr << "determinat: matrix not initialized properly \n";  return NULL; }
	assert(issquare(m));

	if (m->cols == 1) return get(m, 1, 1);

	else if (m->cols == 2)
	{
		return get(m, 1, 1) * get(m, 2, 2) - get(m, 2, 1) * get(m, 1, 2);
	}

	else if (m->cols == 3)
	{
		return get(m, 1, 1) * get(m, 2, 2) * get(m, 3, 3) +
			   get(m, 1, 2) * get(m, 2, 3) * get(m, 3, 1) +
			   get(m, 1, 3) * get(m, 2, 1) * get(m, 3, 2) -
			   get(m, 1, 1) * get(m, 2, 3) * get(m, 3, 2) -
			   get(m, 1, 2) * get(m, 2, 1) * get(m, 3, 3) - 
			   get(m, 1, 3) * get(m, 2, 2) * get(m, 3, 1) ;
	}

	else
	{
		matrix* Q = new matrix(m->rows, m->cols);
		matrix* R = new matrix(m->cols, m->cols);
		Matrix::householder(m, Q, R);
		double val = get(R, 1, 1); int i;
		for (i = 2; i <= R->cols; i++) val *= get(R, i, i);
		return val;
	}
}

int Matrix::solver(matrix* A, matrix* b, matrix* x)
{
	if (!A || !b || !x) { return -1; }
	if (A->cols != x->rows || A->rows != b->rows) { return -2; }
	
	//QR decomp of A
	matrix* Q = new matrix(A->rows, A->cols);
	matrix* R = new matrix(A->cols, A->cols);
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
	if (!a) { std::cerr << "skewSymmetric: matrix not initialized properly \n";  return NULL; }
	matrix* out = new matrix(3, 3);
	double x = 0, y = 0, z = 0; x = get(a, 1, 1); y = get(a, 2, 1); z = get(a, 3, 1);
	set(out, 1, 1, 0); set(out, 1, 2, -z); set(out, 1, 3, y);
	set(out, 2, 1, z); set(out, 2, 2, 0); set(out, 2, 3, -x);
	set(out, 3, 1, -y); set(out, 3, 2, x); set(out, 3, 3, 0);
	return out;
}

void Matrix::euler2rot(Matrix::matrix* euler, Matrix::matrix* rot, Matrix::matrix* t)
{
	double alpha = get(euler, 1, 1);
	matrix* Rx = identity(3); double rx[] = { cos(alpha), sin(alpha), 0, -sin(alpha), cos(alpha), 0, 0, 0, 1 }; Rx->data.assign(rx, rx + 9);

	*rot = *product(rot, Rx);
	delete Rx, rx;

	double beta = get(euler, 2, 1);
	matrix* Ry = identity(3); double ry[] = { cos(beta), 0, -sin(beta), 0, 1, 0, sin(beta), 0, cos(beta) }; Ry->data.assign(rx, rx + 9);
	*rot = *product(rot, Ry);
	delete Ry, ry;

	double gamma = get(euler, 3, 1);
	matrix* Rz = identity(3); double rz[] = { cos(gamma), sin(gamma), 0, -sin(gamma), cos(gamma), 0, 0, 0, 1 }; Rz->data.assign(rx, rx + 9);
	*rot = *product(rot, Rz);
	delete Rz, rz;

	set(t, 1, 1, get(euler, 4, 1));
	set(t, 2, 1, get(euler, 5, 1));
	set(t, 3, 1, get(euler, 6, 1));
}

void Matrix::updateFromEuler(Matrix::matrix* deltaEuler, Matrix::matrix* rot, Matrix::matrix* t)
{
	matrix* deltaR = new matrix(3, 3); matrix* deltaT = new matrix(3, 1);
	euler2rot(deltaEuler, deltaR, deltaT);
	*rot = *product(deltaR, rot);
	*t = *sum(product(deltaR, t), deltaT);
	delete deltaR, deltaT;
}

Matrix::matrix* Matrix::givens(matrix* m, int r, int c)
{
//returns the matrix that annihialtes the (r,c) element
	return m;
}

int Matrix::svd(matrix* A, matrix* U, matrix* VT)
{
	//Golub-Kahan SVD

	//kth householder transformation matrix
	auto hholder = [](matrix* A, matrix* a) {
		matrix* H = identity(A->rows);
		//matrix* a = getColumn(getSub(A, k, k), 1);
		matrix* e = new matrix(a->rows, 1); set(e, 1, 1, 1);
		double a1 = get(a, 1, 1);
		matrix* u = sum(a, scalar_prod(e, norm(a) * SGN(a1)));
		double u1 = get(u, 1, 1); matrix* v;
		if (u1 == 0) v = scalar_prod(u, 0);
		else v = scalar_prod(u, (1 / u1));
		double vn = norm(v);
		double beta = (2 / pow(vn, 2));
		if (vn == 0) { beta = 0; }
		setSub(H, diff(identity(a->rows), scalar_prod(product(v, transpose(v)), beta)), A->rows - a->rows + 1, A->rows - a->rows + 1);
		return H;
	};

	//calculate bidiagonal form of A
	auto bidiagonal = [&](matrix* A, matrix* B, matrix* U, matrix* VT)
	{
		for (int k = 1; k <= MIN(A->rows, A->cols); k++)
		{
			//left hand
			matrix* b = getColumn(getSub(B, k, k), 1);
			matrix* Q = hholder(B, b);
			*U = *product(U, Q);
			*B = *product(Q, B);

			//right hand
			if (k <= A->cols - 2)
			{
				matrix* BT = transpose(B);
				matrix* b = getColumn(getSub(BT, k + 1, k), 1);
				matrix* P = transpose(hholder(BT, b));
				*VT = *product(P, VT);
				*B = *product(B, P);
				delete BT;
			}
		}
	};

	auto givens = [&](int n, int p1, int p2, double c, double s)
	{
		matrix* G = identity(n);
		set(G, p1, p1, c); set(G, p1, p2, s); set(G, p2, p1, -s); set(G, p2, p2, c);
		return G;
	};

	auto diagZero = [&](matrix* B, int k)
	{//zero superdiagonal element where diagonal is zero at (k,k)
		matrix* G = identity(B->cols);
		for (int i = 1; i <= B->cols - k; i++) {
			double alpha = get(B, k, k + i), beta = get(B, k + i, k + i), r = hypot(alpha, beta), c = beta / r, s = -alpha / r;
			matrix* G_i = givens(B->cols, k, k + i, c, s);
			*B = *product(G_i, B);
			*G = *product(G_i, G);
		}
		return G;
	};

	auto lastZero = [&](matrix* B)
	{//when last diagonal element is zero
		matrix* G = identity(B->cols);
		for (int i = B->cols - 1; i >= 1; i--) {
			double alpha = get(B, i, i), beta = get(B, i, B->cols), r = hypot(alpha, beta), c = alpha / r, s = -beta / r;
			matrix* G_i = givens(B->cols, i, B->cols, c, s);
			*B = *product(B, G_i);
			*G = *product(G, G_i);
		}
		return G;
	};

	auto golubKahan = [&](matrix* B, matrix* Q, matrix* PT)
	{
		//B is square, no zeros on diagonal or superdiagonal
		int n = B->cols;
		//trailing 2x2 matrix of BT*B
		double a = pow(get(B, n - 1, n - 1), 2) + pow(get(B, n - 2, n - 1), 2),
			b = get(B, n - 1, n - 1) * get(B, n - 1, n),
			c = b,
			d = pow(get(B, n, n), 2);
		//eigenvalues of trailing 2x2 matrix
		double lambda1 = ((a + d) + sqrt(pow(a, 2) - 2 * a * d + pow(d, 2) + 4 * b * c)) / 2.0;
		double lambda2 = ((a + d) - sqrt(pow(a, 2) - 2 * a * d + pow(d, 2) + 4 * b * c)) / 2.0;
		//shift
		double mu = ((abs(lambda1 - a) < abs(lambda2 - a)) ? lambda1 : lambda2);
		//chasing the bulge (implicit QR)
		double alpha = pow(get(B, 1, 1), 2), beta = get(B, 1, 1) * get(B, 1, 2);
		for (int k = 1; k <= n - 1 ; k++)
		{
			//right
			double h = hypot(alpha, beta);
			double c = (double)alpha / h, s = -(double)beta / h;
			matrix* G = givens(n, k, k+1, c, s);
			*B = *product(B, G);
			*PT = *product(PT, G);

			//left
			alpha = get(B, k, k); beta = get(B, k + 1, k);
			h = hypot(alpha, beta);
			c = alpha / h, s = - beta / h; ;
			G = givens(n, k, k+1, c, -s);
			*B = *product(G, B);
			*Q = *product(G, Q);
			if (k < n - 1) { alpha = get(B, k, k + 1); beta = get(B, k, k + 2); }
		}
	};


	auto svd_iQR = [&](matrix* B, matrix* Q, matrix* PT)
	{
		//Implicit QR step

		double constexpr epsilon = 1e-40; // std::numeric_limits< double >::epsilon(); //change to machine epsilon

		int q = 1, p = 0, iter = 1;

		while (q < B->cols)
		{
			//std::cout << "iter: " << iter << "\n"; 
			iter++; if (iter > 500)break;

			for (int i = 1; i <= B->cols - 1; i++)
			{
				if (abs(get(B, i, i + 1)) <= epsilon * (abs(get(B, i, i)) + abs(get(B, i + 1, i + 1)))) { set(B, i, i + 1, 0); }
			}

			// check diagonal elements
			for (int i = 2; i <= B->cols; i++)
			{
				//std::cout << i << " " << abs(get(B, B->cols - i + 1, B->cols - i + 2)) << " ";
				if (abs(get(B, B->cols - i + 1, B->cols - i + 2)) <= epsilon) q = i;
				else  break;
				//std::cout << "\n";
			}
			for (int i = 1; i < B->cols - q; i++)
			{
			    // check superdiagonal elements for zero
				//std::cout << i <<" "<< abs(get(B, i, i + 1)) << " ";
				if (abs(get(B, i, i + 1)) <= epsilon) { p = i; }
				else {break;}
				//std::cout << "\n";
			}

			//std::cout << "p " << p << " q " << q << "\n\n";
			matrix* B11 = identity(1); matrix* B22 = identity(1); matrix* B33 = identity(1);

			if (p >= 1)                 B11 = getSub(B, 1, 1, p, p);
			if (p < B->cols - q)        B22 = getSub(B, p + 1, p + 1, B->cols - q + 1, B->cols - q + 1);
			if (q >= 1)                 B33 = getSub(B, B->cols - q + 1, B->cols - q + 1, B->cols, B->cols);

			if (q < B->cols)
			{
				//check and zero out superdiagonal elements in B22 if the diagonal is also zero
				bool diag = false;
				for (int i = 1; i <= B22->cols; i++) {
					if (get(B, i, i) == 0)
					{
						diag = true;
						if (i < B22->cols) { matrix* G = diagZero(B, i); *Q = *product(G, Q); }
						if (i == B22->cols) { matrix* G = lastZero(B); *Q = *product(Q, G); }
					}
				}

				if (diag == false)
				{
					matrix* Q22 = getSub(Q, p + 1, p + 1, B->cols - q + 1, B->cols - q + 1);
					matrix* P22 = getSub(PT, p + 1, p + 1, B->cols - q + 1, B->cols - q + 1);
					golubKahan(B22, Q22, P22);
					setSub(B, B22, p + 1, p + 1);
					setSub(Q, Q22, p + 1, p + 1);
					setSub(PT, P22, p + 1, p + 1);
					delete Q22, P22;
				}
			}
		}
	};

	//implementation  A = Q_f * B * PT_f
	assert(A->rows >= A->cols);

	matrix* B = new matrix(A->rows, A->cols); *B = *A;
	matrix* Q = identity(A->rows); matrix* PT = identity(A->cols);

	//get bidiagonal form of A: A = Q * B * PT
	bidiagonal(A, B, Q, PT);

	//remove last (row - col) zero rows
	matrix* B_  = getSub(B, 1, 1, A->cols, A->cols);
	matrix* Q_  = identity(A->cols); 
	matrix* PT_ = identity(A->cols); 

	//do francis algorithm : B = Q_ * B * PT_
	svd_iQR(B_, Q_, PT_);

	//insert back
	setSub(B, B_, 1, 1);
	matrix* Q_t = identity(A->rows);
	setSub(Q_t, Q_, 1, 1);
	matrix* Q_f = product(Q, transpose(Q_t));
	matrix* PT_f = product(transpose(PT_), PT);

	//check
	//print(A);
	//print(Q_f);
	//print(B);
	//print(PT_f); 
	//print((product(product(Q_f, B), PT_f))); 

	return 0;
}


