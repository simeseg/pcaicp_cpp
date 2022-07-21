#pragma once
#include <vector>
#include <algorithm>
#include "utils.h"

#define INIT_VERTICES_COUNT 6
#define INIT_FACES_COUNT 8
#define VECTOR_LENGTH 1

namespace mesh {

	struct vertex:utils::point {
		size_t index =0;
		double x, y, z;
		bool isVisited = false;
		bool isAuxiliary = false;

		vertex(const double& x, const double& y, const double& z) : index(generateRunningId()), x(x), y(y), z(z){}
		vertex(utils::point pt): index(generateRunningId()), x(pt.x), y(pt.y), z(pt.z){}
		vertex(const double& x, const double& y, const double& z, bool isAuxiliary) : index(generateRunningId()), x(x), y(y), z(z), isAuxiliary(isAuxiliary){}
		vertex(vertex* v, const double& lengthAfterProjection): index(v->index), isVisited(v->isVisited), isAuxiliary(v->isAuxiliary)
		{
			double length = sqrt(pow(v->x, 2) + pow(v->y, 2) + pow(v->z, 2));
			double scale = lengthAfterProjection / length;
			x = scale * v->x;
			y = scale * v->y;
			z = scale * v->z;
		}

		~vertex() {}
		bool isCoincidentWith(vertex* v) { return (x == v->x && y == v->y && z == v->z); }
		int generateRunningId() { static int id; return id++;}
	};


	struct face {
		size_t index =0;
		vertex* Vertex[3];
		face* neighbor[3];

		face(vertex* v0, vertex* v1, vertex* v2) { index = generateRunningId(); Vertex[0] = (v0); Vertex[1] = (v1); Vertex[2] = (v2); }
		~face() { }

		bool hasVertexCoincidentWith(vertex* v) { return (Vertex[0]->isCoincidentWith(v) || Vertex[1]->isCoincidentWith(v) || Vertex[2]->isCoincidentWith(v)); }
		void assignNeighbors(face* n0, face* n1, face* n2) { neighbor[0] = n0; neighbor[1] = n1; neighbor[2] = n2;}
		int generateRunningId() { static int id = 0; return id++;}
	};


	struct tetra {
		size_t v0, v1, v2, v3;
	};


	struct TriangleMesh {
		vertex* _auxiliaryVertices[INIT_VERTICES_COUNT];
		std::vector<vertex*>* vertices;
		std::vector<face*>* faces;
		TriangleMesh()
		{
			for (int i = 0; i < INIT_VERTICES_COUNT; i++) {
				_auxiliaryVertices[i] = new vertex(
					(i % 2 == 0 ? 1 : -1) * (i / 2 == 0 ? VECTOR_LENGTH : 0),
					(i % 2 == 0 ? 1 : -1) * (i / 2 == 1 ? VECTOR_LENGTH : 0),
					(i % 2 == 0 ? 1 : -1) * (i / 2 == 2 ? VECTOR_LENGTH : 0),
					true);
			}
			
			vertices = new std::vector<vertex*>();
			faces = new std::vector<face*>();
		}

		~TriangleMesh() { }
			//for (auto& v : _auxiliaryVertices) { delete v; }
			//for (auto& v : *vertices) { delete v; };
			//for (auto& f : *faces) { delete f; }
			//delete this; }

		void buildInitialHull(std::vector<vertex*>* cloud);
		void insertVertex(vertex* v);
		void removeBadTriangles();
		void splitTriangle(face* face, vertex* v);
		void fixNeighborhood(face* target, face* oldNeighbor, face* newNeighbor);
		void doLocalOptimization(face* f0, face* f1);
		bool trySwapDiagonal(face* f0, face* f1);
		bool isMinimumValueInArray(double arr[], int length, int index);
		double distance(vertex* v0, vertex* v1);
		double determinant(vertex* v0, vertex* v1, vertex* v2);
		double determinant(double matrix[]);

		std::vector<std::tuple<int, int, int>*> delaunayTriangulation(const std::vector<vertex*>& cloud);

	};
}

