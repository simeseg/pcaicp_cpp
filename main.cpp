#define NOMINMAX
#include "segment_pcd.h"
#include "geometry.h"
#include "registration.h"

#pragma warning(disable:4996)

int main()
{
	//read rgb image
	segmentpcd::Image o_image;
	segmentpcd::_LoadRGBImage("image/0_mask_out.jpg", o_image);
	

	//read ply file
	//calibration parameters
	segmentpcd::CalibData calibdata;
	calibdata.ExtrinsicPara.CameraToWorld_R[0][0] = 0.9938779; calibdata.ExtrinsicPara.CameraToWorld_R[0][1] = -0.00574401; calibdata.ExtrinsicPara.CameraToWorld_R[0][2] = -0.11033379; calibdata.ExtrinsicPara.CameraToWorld_T[0][0] = -6.73626975;
	calibdata.ExtrinsicPara.CameraToWorld_R[1][0] = 0.0035367; calibdata.ExtrinsicPara.CameraToWorld_R[1][1] =  0.99978988; calibdata.ExtrinsicPara.CameraToWorld_R[1][2] = -0.02019092; calibdata.ExtrinsicPara.CameraToWorld_T[1][0] = -0.31545435;
	calibdata.ExtrinsicPara.CameraToWorld_R[2][0] = 0.1104265; calibdata.ExtrinsicPara.CameraToWorld_R[2][1] =  0.01967709; calibdata.ExtrinsicPara.CameraToWorld_R[2][2] =  0.99368947; calibdata.ExtrinsicPara.CameraToWorld_T[2][0] =  1044.87198;

    //intrinsic
	calibdata.IntrinsicPara.u0 = 1251.27033466;
	calibdata.IntrinsicPara.v0 = 1018.533226;
	calibdata.IntrinsicPara.fx = 4696.67157029;
	calibdata.IntrinsicPara.fy = 4687.85893501;
	calibdata.IntrinsicPara.skew = 0.0000;
	calibdata.IntrinsicPara.lambda = 1.0000;
	calibdata.IntrinsicPara.k1 = -0.0686;
	calibdata.IntrinsicPara.k2 = 0.1679;
	calibdata.IntrinsicPara.k3 = -0.7314;
	calibdata.IntrinsicPara.p1 = -0.00051;
	calibdata.IntrinsicPara.p2 = 0.00125;
	calibdata.nImageHeight = o_image.height; 
	calibdata.nImageWidth = o_image.width;

	//input and output coordinate type
	utils::PointCloud pointcloud;
	segmentpcd::_LoadPly("image/PointCloud0.ply", pointcloud);

	segmentpcd::_project(pointcloud, calibdata, o_image);
	//std::cout << o_image.pixels.at(200).r << "  "<< o_image.pixels.at(200).Xc;

	segmentpcd::Bboxes bboxes; 
	segmentpcd::_readBboxes("image/0_mask_out.txt", bboxes);
	//std::cout << bboxes.bboxes.size() << " " << bboxes.num;

	segmentpcd::_masks masks;
	int scene_id = 0;
	segmentpcd::_get_masks(masks, bboxes.num, scene_id);
	std::cout << masks.masks.at(0).pixels.at(0).b;

	segmentpcd::_outputClouds o_pointclouds;
	segmentpcd::_clip_bolt_pcd(bboxes, o_image, masks, o_pointclouds);
	std::cout << o_pointclouds.output_clouds.at(0).points.size()<<"\n";
	
	/*
	//test QR solver
	Matrix::matrix* A = Matrix::newMatrix(3, 3);
	Matrix::matrix* Q = Matrix::newMatrix(3, 3);
	Matrix::matrix* R = Matrix::newMatrix(3, 3);
	double val[] = { 2.92, 0.86, -1.15, 0.86, 6.51, 3.32, -1.15, 3.32, 4.57};
	A->data = val;
	
	Matrix::matrix* b = Matrix::newMatrix(3,1); double bval[] = { 2,1,4}; b->data = bval;
	Matrix::matrix* x = Matrix::newMatrix(3, 1);
	Matrix::matrix* E = Matrix::newMatrix(3, 1);
	
	Matrix::eigendecomposition(A, Q, R);
	print(A); print(Q); print(R);
	*/

	utils::PointCloud modelcloud;
	segmentpcd::_LoadPly("image/SurfaceSampledModel.ply", modelcloud);

	/*
	//test triangulation
	std::ofstream out("image/model.txt");
	std::vector<mesh::vertex*> cloud; 
	for (int i = 0; i<modelcloud.points.size() ; i++)
	{
		utils::point pt = modelcloud.points.at(i);
		out << pt.x << "," << pt.y << "," << pt.z << "," << pt.nx << "," << pt.ny << "," << pt.nz << "\n";
		mesh::vertex v(pt); cloud.push_back(&v);
		//std::cout << i << "\n";
	}
	mesh::TriangleMesh mesh = mesh::TriangleMesh::TriangleMesh();
	std::vector<std::tuple<int, int, int>*> triangulation = mesh.delaunayTriangulation(cloud);
	std::cout << "mesh size: " << triangulation.size();
	*/

	//test alignment
	utils::PointCloud scene_out; scene_out.points.reserve(o_pointclouds.output_clouds.at(0).points.size());
	Registration::coarseAlign(&modelcloud, &o_pointclouds.output_clouds.at(0), &scene_out);

	return 1;
}