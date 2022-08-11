#include "geometry.h"


std::vector<std::tuple<int, int, int>*> mesh::TriangleMesh::delaunayTriangulation(const std::vector<vertex*>& cloud)
{
	vertices->reserve(cloud.size());
	faces->reserve(8 + (cloud.size() - 6) * 2);

	//project vertices on unit sphere to triangulate
    for (auto& itDots : cloud)
    {
        vertex* projectedDot = new vertex((itDots), VECTOR_LENGTH);
        vertices->push_back(projectedDot);
    }
    
    // prepare initial convex hull with 6 vertices and 8 triangle faces
    buildInitialHull(vertices);



    for (auto& itDots:*vertices)
    {
        vertex* dot = itDots;
        if (!dot->isVisited)
        {
            insertVertex(dot);
        }
    }

    std::cout << "vertices: " << vertices->at(1)->index << "\n";
    std::cout << "faces: " << faces->size() << "\n";

    // remove trianges connected with auxiliary dots
    removeBadTriangles();

    // generate output
    std::vector<std::tuple<int, int, int>*> mesh = std::vector<std::tuple<int, int, int>*>();
    for (auto& itMesh:*faces)
    {
        face* face = itMesh;
        mesh.push_back(new std::tuple<int, int, int>( face->Vertex[0]->index, face->Vertex[1]->index, face->Vertex[2]->index ));
    }

    return mesh;
}


void mesh::TriangleMesh::buildInitialHull(std::vector<vertex*>* cloud)
{
    vertex* initialVertices[INIT_VERTICES_COUNT];
    face* initialHullFaces[INIT_FACES_COUNT];

    for (int i = 0; i < INIT_VERTICES_COUNT; i++)
    {
        initialVertices[i] = _auxiliaryVertices[i];
    }

    // if close enough, use input dots to replace auxiliary dots so won't be removed in the end
    double minDistance[INIT_VERTICES_COUNT] = { 0, 0, 0, 0, 0, 0 };
    for (auto& it : *cloud)
    {
        double distance[INIT_VERTICES_COUNT];
        for (int i = 0; i < INIT_VERTICES_COUNT; i++)
        {
            distance[i] = mesh::TriangleMesh::distance(_auxiliaryVertices[i], it);
            if (minDistance[i] == 0 || distance[i] < minDistance[i])
            {
                minDistance[i] = distance[i];
            }
        }

        for (int i = 0; i < INIT_VERTICES_COUNT; i++)
        {
            if (minDistance[i] == distance[i] && isMinimumValueInArray(distance, INIT_VERTICES_COUNT, i))
            {
                initialVertices[i] = it;
            }
        }
    }

    int vertex0Index[] = { 0, 0, 0, 0, 1, 1, 1, 1 };
    int vertex1Index[] = { 4, 3, 5, 2, 2, 4, 3, 5 };
    int vertex2Index[] = { 2, 4, 3, 5, 4, 3, 5, 2 };

    for (int i = 0; i < INIT_FACES_COUNT; i++)
    {
        vertex* v0 = initialVertices[vertex0Index[i]];
        vertex* v1 = initialVertices[vertex1Index[i]];
        vertex* v2 = initialVertices[vertex2Index[i]];

        face* face = new mesh::face(v0, v1, v2);
        initialHullFaces[i] = face;
        faces->push_back(face);
    }

    int neighbor0Index[] = { 1, 2, 3, 0, 7, 4, 5, 6 };
    int neighbor1Index[] = { 4, 5, 6, 7, 0, 1, 2, 3 };
    int neighbor2Index[] = { 3, 0, 1, 2, 5, 6, 7, 4 };

    for (int i = 0; i < INIT_FACES_COUNT; i++)
    {
        face* n0 = initialHullFaces[neighbor0Index[i]];
        face* n1 = initialHullFaces[neighbor1Index[i]];
        face* n2 = initialHullFaces[neighbor2Index[i]];
        initialHullFaces[i]->assignNeighbors(n0, n1, n2);
    }

    // dot already in the mesh, avoid being visited by insertVertex() again
    for (int i = 0; i < INIT_VERTICES_COUNT; i++)
    {
        initialVertices[i]->isVisited = true;
    }
}

void mesh::TriangleMesh::insertVertex(vertex* v)
{
    double det[] = { 0, 0, 0 };

    face* face = faces->at(0);
    int counter = 0;
    for(auto& it : *faces)
    {
        

        det[0] = determinant(face->Vertex[0], face->Vertex[1], v);
        det[1] = determinant(face->Vertex[1], face->Vertex[2], v);
        det[2] = determinant(face->Vertex[2], face->Vertex[0], v);
        std::cout << counter << " " << it->Vertex[0]->isVisited << "\n";
        // if this dot projected into an existing triangle, split the existing triangle to 3 new ones
        if (det[0] >= 0 && det[1] >= 0 && det[2] >= 0)
        {
            if (!face->hasVertexCoincidentWith(v))
            {
                splitTriangle(face, v);
            }

            return;
        }

        // on one side, search neighbors
        else if (det[1] >= 0 && det[2] >= 0)
            face = face->neighbor[0];
        else if (det[0] >= 0 && det[2] >= 0)
            face = face->neighbor[1];
        else if (det[0] >= 0 && det[1] >= 0)
            face = face->neighbor[2];

        // cannot determine effectively 
        else if (det[0] >= 0)
            face = face->neighbor[1];
        else if (det[1] >= 0)
            face = face->neighbor[2];
        else if (det[2] >= 0)
            face = face->neighbor[0];
        else
            face = faces->at(counter);

        counter += 1;
    }
}

void mesh::TriangleMesh::removeBadTriangles()
{
    std::vector<mesh::face*>::iterator it;
    for (it = faces->begin(); it != faces->end();)
    {
        face* face = *it;
        bool isExtraTriangle = false;
        for (int i = 0; i < 3; i++)
        {
            if (face->Vertex[i]->isAuxiliary)
            {
                isExtraTriangle = true;
                break;
            }
        }

        if (isExtraTriangle)
        {
            delete* it;
            it = faces->erase(it);
        }
        else
        {
            it++;
        }
    }
}

void mesh::TriangleMesh::splitTriangle(face* face, vertex* v)
{
    mesh::face* newface1 = new mesh::face(v, face->Vertex[1], face->Vertex[2]);
    mesh::face* newface2 = new mesh::face(v, face->Vertex[2], face->Vertex[0]);

    face->Vertex[2] = face->Vertex[1];
    face->Vertex[1] = face->Vertex[0];
    face->Vertex[0] = v;

    newface1->assignNeighbors(face, face->neighbor[1], newface2);
    newface2->assignNeighbors(newface1, face->neighbor[2], face);
    face->assignNeighbors(newface2, face->neighbor[0], newface1);

    fixNeighborhood(newface1->neighbor[1], face, newface1);
    fixNeighborhood(newface2->neighbor[1], face, newface2);

    faces->push_back(newface1);
    faces->push_back(newface2);

    // optimize triangles according to delaunay triangulation definition
    doLocalOptimization(face, face->neighbor[1]);
    doLocalOptimization(newface1, newface1->neighbor[1]);
    doLocalOptimization(newface2, newface2->neighbor[1]);
}

void mesh::TriangleMesh::fixNeighborhood(face* target, face* oldNeighbor, face* newNeighbor)
{
    for (int i = 0; i < 3; i++)
    {
        if (target->neighbor[i] == oldNeighbor)
        {
            target->neighbor[i] = newNeighbor;
            break;
        }
    }
}

void mesh::TriangleMesh::doLocalOptimization(face* f0, face* f1)
{
    for (int i = 0; i < 3; i++)
    {
        if (f1->Vertex[i] == f0->Vertex[0] || f1->Vertex[i] == f0->Vertex[1] || f1->Vertex[i] == f0->Vertex[2])
        {
            continue;
        }

        double matrix[] = {
            f1->Vertex[i]->x - f0->Vertex[0]->x,
            f1->Vertex[i]->y - f0->Vertex[0]->y,
            f1->Vertex[i]->z - f0->Vertex[0]->z,

            f1->Vertex[i]->x - f0->Vertex[1]->x,
            f1->Vertex[i]->y - f0->Vertex[1]->y,
            f1->Vertex[i]->z - f0->Vertex[1]->z,

            f1->Vertex[i]->x - f0->Vertex[2]->x,
            f1->Vertex[i]->y - f0->Vertex[2]->y,
            f1->Vertex[i]->z - f0->Vertex[2]->z
        };

        if (determinant(matrix) <= 0)
        {
            // terminate after optimized
            break;
        }

        if (trySwapDiagonal(f0, f1))
        {
            return;
        }
    }
}

bool mesh::TriangleMesh::trySwapDiagonal(face* f0, face* f1)
{
    for (int j = 0; j < 3; j++)
    {
        for (int k = 0; k < 3; k++)
        {
            if (f0->Vertex[j] != f1->Vertex[0] &&
                f0->Vertex[j] != f1->Vertex[1] &&
                f0->Vertex[j] != f1->Vertex[2] &&
                f1->Vertex[k] != f0->Vertex[0] &&
                f1->Vertex[k] != f0->Vertex[1] &&
                f1->Vertex[k] != f0->Vertex[2])
            {
                f0->Vertex[(j + 2) % 3] = f1->Vertex[k];
                f1->Vertex[(k + 2) % 3] = f0->Vertex[j];

                f0->neighbor[(j + 1) % 3] = f1->neighbor[(k + 2) % 3];
                f1->neighbor[(k + 1) % 3] = f0->neighbor[(j + 2) % 3];
                f0->neighbor[(j + 2) % 3] = f1;
                f1->neighbor[(k + 2) % 3] = f0;

                fixNeighborhood(f0->neighbor[(j + 1) % 3], f1, f0);
                fixNeighborhood(f1->neighbor[(k + 1) % 3], f0, f1);

                doLocalOptimization(f0, f0->neighbor[j]);
                doLocalOptimization(f0, f0->neighbor[(j + 1) % 3]);
                doLocalOptimization(f1, f1->neighbor[k]);
                doLocalOptimization(f1, f1->neighbor[(k + 1) % 3]);

                return true;
            }
        }
    }

    return false;
}

bool mesh::TriangleMesh::isMinimumValueInArray(double arr[], int length, int index)
{
    for (int i = 0; i < length; i++)
    {
        if (arr[i] < arr[index])
        {
            return false;
        }
    }

    return true;
}

double mesh::TriangleMesh::distance(vertex * v0, vertex * v1)
{
    return sqrt(pow((v0->x - v1->x), 2) + pow((v0->y - v1->y), 2) + pow((v0->z - v1->z), 2));
}

double mesh::TriangleMesh::determinant(vertex* v0, vertex* v1, vertex* v2)
{
    double matrix[] = {
        v0->x, v0->y, v0->z,
        v1->x, v1->y, v1->z,
        v2->x, v2->y, v2->z
    };

    return determinant(matrix);
}

double mesh::TriangleMesh::determinant(double matrix[])
{
    // inversed for left handed coordinate system
    double determinant = matrix[2] * matrix[4] * matrix[6]
        + matrix[0] * matrix[5] * matrix[7]
        + matrix[1] * matrix[3] * matrix[8]
        - matrix[0] * matrix[4] * matrix[8]
        - matrix[1] * matrix[5] * matrix[6]
        - matrix[2] * matrix[3] * matrix[7];

    // adjust result based on float number accuracy, otherwise causing deadloop
    return abs(determinant) <= DBL_EPSILON ? 0 : determinant;
}

