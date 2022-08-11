#define NOMINMAX
#include "segment_pcd.h"
#include "geometry.h"
#include "registration.h"

#pragma warning(disable:4996)


int main()
{
	int scene_id = 61;
	
	//read rgb image
	segmentpcd::Image o_image;
	char rgb_buffer[250];
	sprintf_s(rgb_buffer, "D:/PointCloudSegmentation/Dataset/New_data/Wolfson_lab/1.Success_1st_Mask/%d_mask_out.jpg", scene_id);
	segmentpcd::_LoadRGBImage(rgb_buffer, o_image);
	

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
	char pcd_buffer[250];
	sprintf_s(pcd_buffer, "D:/PointCloudSegmentation/Dataset/New_data/Wolfson_lab/220613_3D/PointCloud%d.ply", scene_id);
	segmentpcd::_LoadPly(pcd_buffer, pointcloud);

	segmentpcd::_project(pointcloud, calibdata, o_image);
	//std::cout << o_image.pixels.at(200).r << "  "<< o_image.pixels.at(200).Xc;

	segmentpcd::Bboxes bboxes; 
	char bbox_buffer[250];
	sprintf_s(bbox_buffer, "D:/PointCloudSegmentation/Dataset/New_data/Wolfson_lab/1.Success_1st_Mask/%d_mask_out.txt", scene_id);
	segmentpcd::_readBboxes(bbox_buffer, bboxes);
	//std::cout << bboxes.bboxes.size() << " " << bboxes.num;

	segmentpcd::_masks masks;
	
	segmentpcd::_get_masks(masks, bboxes.num, scene_id);
	std::cout << masks.masks.at(0).pixels.at(0).b;

	segmentpcd::_outputClouds o_pointclouds;
	segmentpcd::_clip_bolt_pcd(bboxes, o_image, masks, o_pointclouds);
	std::cout << o_pointclouds.output_clouds.at(0).points.size()<<"\n";

	utils::PointCloud modelcloud;
	segmentpcd::_LoadPly("D:/PointCloudSegmentation/BoltData/SurfaceSampledModel.ply", modelcloud);

	//test alignment
	utils::PointCloud scene_out; scene_out.points.reserve(o_pointclouds.output_clouds.at(0).points.size());
	Registration::Align(&modelcloud, &o_pointclouds.output_clouds.at(0), &scene_out);


	/*
	//test QR solver
	Matrix::matrix* A = new Matrix::matrix(3, 3);
	Matrix::matrix* Q = new Matrix::matrix(3, 3);
	Matrix::matrix* R = new Matrix::matrix(3, 3);
	double val[] = { 2.92, 0.86, -1.15, 0.86, 6.51, 3.32, -1.15, 3.32, 4.57};
	A->data = val;
	
	Matrix::matrix* b = new Matrix::matrix(3,1); double bval[] = { 2,1,4}; b->data = bval;
	Matrix::matrix* x = new Matrix::matrix(3, 1);
	Matrix::matrix* E = new Matrix::matrix(3, 1);
	
	Matrix::eigendecomposition(A, Q, R);
	print(A); print(Q); print(R);
	
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

	Matrix::matrix* A = new Matrix::matrix(6, 6); 
	//Matrix::matrix* A = new Matrix::matrix(3, 3);
	
	//for (int i = 0; i < A->data.size(); i++) { A->data[i] = i + 1; }; 
	double v[] =   { 2, 5, 4, 6, 0, 8,
					 7, 4, 3, 3, 1, 8,
					 0, 2, 7, 0, 3, 1,
					 1, 4, 7, 1, 2, 0,
					 3, 9, 4, 4, 0, 4,
					 3, 7, 3, 5, 4, 2 };

	double v2[] = {  2, 5, 4, 
					 7, 4, 3, 
					 0, 2, 7,  
					 1, 4, 7, 
					 3, 9, 4,
					 3, 7, 3 };

	double v3[] = { 1, 1, 0, 0, 1e-20, 1e-20, 0,0, 1e-40 };

	A->data.assign(v, v+36);

	Matrix::matrix* U = new Matrix::matrix(A->rows, A->rows);
	Matrix::matrix* S = new Matrix::matrix(A->rows, A->cols);
	Matrix::matrix* V = new Matrix::matrix(A->cols, A->cols);
	Matrix::svd(A, U, V);
	std::cout<<Matrix::determinant(A);
	//Matrix::householder(A, U, V); print(A); print(U); print(V); print(Matrix::product(U, V));
	*/


	return 1;
}

