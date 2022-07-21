#pragma warning(disable:26451)
#include "utils.h"
#include "segment_pcd.h"
#include "include/happly.h"
#define STB_IMAGE_IMPLEMENTATION 
#include "include/stb_image.h"

using namespace segmentpcd;

BOOL segmentpcd::_SavePointCloudToPly(const utils::PointCloud& i_PointCloud, const CString& SavePath, BOOL bBinary)
{
	if (bBinary)
		//binary
	{
		if (i_PointCloud.size <= 0)
		{
			return FALSE;
		}

		std::ofstream outfile;
		std::ios::openmode mode = std::ios::out | std::ios::trunc | std::ios::binary;
		CStringA strA(SavePath);
		outfile.open(strA, mode);
		if (!outfile.is_open())
		{
			return FALSE;
		}

		int total = i_PointCloud.size;

		const char* format_header = "binary_little_endian 1.0"; // binary일 시 "binary_little_endian 1.0"
		outfile << "ply" << std::endl
			<< "format " << format_header << std::endl
			<< "comment intekplus point cloud generated" << std::endl
			<< "element vertex " << total << std::endl
			<< "property float x" << std::endl
			<< "property float y" << std::endl
			<< "property float z" << std::endl;

		outfile << "element face 0" << std::endl
			<< "property list uchar int vertex_indices" << std::endl
			<< "end_header" << std::endl;

		double fpoint[3];
		for (int i = 0; i < total - 1; i++)
		{
			fpoint[0] = i_PointCloud.points.at(i).x;
			fpoint[1] = i_PointCloud.points.at(i).y;
			fpoint[2] = i_PointCloud.points.at(i).z;

			outfile.write(reinterpret_cast<const char*>(&fpoint[0]), sizeof(double));
			outfile.write(reinterpret_cast<const char*>(&fpoint[1]), sizeof(double));
			outfile.write(reinterpret_cast<const char*>(&fpoint[2]), sizeof(double));
		}

		outfile.close();
		std::cerr << "[write_ply] Saved " << total << " points (" << SavePath << ")" << std::endl;
		return TRUE;
	}
	else				//ascii
	{
		if (i_PointCloud.size <= 0)
		{
			return FALSE;
		}

		std::ofstream outfile;
		std::ios::openmode mode = std::ios::out | std::ios::trunc;
		CStringA strA(SavePath);
		outfile.open(strA, mode);
		if (!outfile.is_open())
		{
			return FALSE;
		}

		int total = i_PointCloud.size;

		const char* format_header = "ascii 1.0"; // binary일 시 "binary_little_endian 1.0"
		outfile << "ply" << std::endl
			<< "format " << format_header << std::endl
			<< "comment intekplus point cloud generated" << std::endl
			<< "element vertex " << total << std::endl
			<< "property float x" << std::endl
			<< "property float y" << std::endl
			<< "property float z" << std::endl;

		outfile << "element face 0" << std::endl
			<< "property list uchar int vertex_indices" << std::endl
			<< "end_header" << std::endl;

		double fpoint[3];
		CStringA str;
		str.GetBuffer(total * ((10) * 3 + 2 + 1) + 1000);
		CStringA tempstr;
		for (int i = 0; i < total - 1; i++)
		{
			fpoint[0] = i_PointCloud.points.at(i).x;
			fpoint[1] = i_PointCloud.points.at(i).y;
			fpoint[2] = i_PointCloud.points.at(i).z;

			tempstr.Format("%f %f %f\n", fpoint[0], fpoint[1], fpoint[2]);
			str += tempstr;
		}
		outfile << str << std::endl;
		str.ReleaseBuffer();

		outfile.close();
		std::cerr << "[write_ply] Saved " << total << " points (" << SavePath << ")" << std::endl;
		return TRUE;
	}
}

BOOL segmentpcd::_LoadPly(char const* i_strPlyPath, utils::PointCloud& o_PointCloud)
{
	//std::string str = std::string(CT2A(i_strPlyPath.GetString()));
	happly::PLYData plyIn(i_strPlyPath);

	if (!plyIn.hasElement("vertex"))
	{
		return FALSE;
	}

	std::vector<std::array<double, 3>> vecPos = plyIn.getVertexPositions();
	if (vecPos.size() > 2592 * 2048)	//버퍼 최대 사이즈는 고정.
		return FALSE;

	o_PointCloud.points = std::vector<utils::point>(vecPos.size());
	o_PointCloud.size = o_PointCloud.points.size();


	for (int i = 0; i < o_PointCloud.points.size(); i++)
	{
		o_PointCloud.points.at(i).x = vecPos.at(i)[0];
		o_PointCloud.points.at(i).y = vecPos.at(i)[1];
		o_PointCloud.points.at(i).z = vecPos.at(i)[2];
	}

	return TRUE;
}

BOOL segmentpcd::_LoadRGBImage(char const* i_strImagePath, segmentpcd::Image& o_image)
{
	//std::string str = std::string(CT2A(i_strImagePath.GetString()));
	int _channel = 3;
	unsigned char* rgb = stbi_load(i_strImagePath, &o_image.width, &o_image.height, &_channel, _channel);

	if (rgb == NULL)
	{
		printf("Error loading RGB image \n");
		exit(1);
	}

	long nMaxPixelSize = o_image.width * o_image.height;

	o_image.pixels = std::vector<segmentpcd::Pixel>(nMaxPixelSize);

	unsigned byteperpixel = _channel;

	for (int n = 0; n < nMaxPixelSize - 1; n++)
	{
		unsigned char* pixelOffset = rgb + (n)*byteperpixel;
		o_image.pixels.at(n).r = (int)pixelOffset[0];
		o_image.pixels.at(n).g = (int)pixelOffset[1];
		o_image.pixels.at(n).b = (int)pixelOffset[2];
	}

	free(rgb);

	return TRUE;
}

void segmentpcd::_project(const utils::PointCloud& i_PointCloud,
	const segmentpcd::CalibData& i_CalibData,
	segmentpcd::Image& o_Image)
{
	double fx = i_CalibData.IntrinsicPara.fx;
	double fy = i_CalibData.IntrinsicPara.fy;
	double cx = i_CalibData.IntrinsicPara.u0;
	double cy = i_CalibData.IntrinsicPara.v0;
	double skew_c = i_CalibData.IntrinsicPara.skew;

	double k1 = i_CalibData.IntrinsicPara.k1;
	double k2 = i_CalibData.IntrinsicPara.k2;
	double k3 = i_CalibData.IntrinsicPara.k3;
	double p1 = i_CalibData.IntrinsicPara.p1;
	double p2 = i_CalibData.IntrinsicPara.p2;

	double r_square, radial_d, tangential_d_x, tangential_d_y;

	double x_nu, y_nu;
	double x_nd, y_nd;
	long x_p, y_p;

	long nMaxPixelSize = i_CalibData.nImageWidth * i_CalibData.nImageHeight;

	long nIndex;
	double pt_w[3];
	double pt_c[3];

			for (int n = 0; n < i_PointCloud.size - 1; n++)
			{
				pt_w[0] = i_PointCloud.points.at(n).x;
				pt_w[1] = i_PointCloud.points.at(n).y;
				pt_w[2] = i_PointCloud.points.at(n).z;

				//World Coord.-> Camera Coord.
				pt_c[0] = i_CalibData.ExtrinsicPara.CameraToWorld_R[0][0] * pt_w[0] + i_CalibData.ExtrinsicPara.CameraToWorld_R[1][0] * pt_w[1] + i_CalibData.ExtrinsicPara.CameraToWorld_R[2][0] * pt_w[2] + i_CalibData.ExtrinsicPara.CameraToWorld_T[0][0];
				pt_c[1] = i_CalibData.ExtrinsicPara.CameraToWorld_R[0][1] * pt_w[0] + i_CalibData.ExtrinsicPara.CameraToWorld_R[1][1] * pt_w[1] + i_CalibData.ExtrinsicPara.CameraToWorld_R[2][1] * pt_w[2] + i_CalibData.ExtrinsicPara.CameraToWorld_T[1][0];
				pt_c[2] = i_CalibData.ExtrinsicPara.CameraToWorld_R[0][2] * pt_w[0] + i_CalibData.ExtrinsicPara.CameraToWorld_R[1][2] * pt_w[1] + i_CalibData.ExtrinsicPara.CameraToWorld_R[2][2] * pt_w[2] + i_CalibData.ExtrinsicPara.CameraToWorld_T[2][0];

				//Camera Coord.-> Normalized Undistorted Image Coord.
				x_nu = pt_c[0] / pt_c[2];
				y_nu = pt_c[1] / pt_c[2];

				//Normalized Undistorted Image Coord.-> Normalized Distorted Image Coord.
				r_square = x_nu * x_nu + y_nu * y_nu;
				radial_d = 1 + k1 * r_square + k2 * r_square * r_square + k3 * r_square * r_square * r_square;
				tangential_d_x = 2 * p1 * x_nu * y_nu + p2 * (r_square + 2 * x_nu * x_nu);
				tangential_d_y = p1 * (r_square + 2 * y_nu * y_nu) + 2 * p2 * x_nu * y_nu;

				x_nd = x_nu * radial_d + tangential_d_x;
				y_nd = y_nu * radial_d + tangential_d_y;

				//Normalized Distorted Image Coord.-> Pixel Coord.
				x_p = (long)(fx * x_nd + skew_c * fx * y_nd + cx);
				y_p = (long)(fy * y_nd + cy);

				nIndex = y_p * i_CalibData.nImageWidth + x_p;

				//camera
 				o_Image.pixels.at(nIndex).Xc = x_nd * pt_c[2];
				o_Image.pixels.at(nIndex).Yc = y_nd * pt_c[2];
				o_Image.pixels.at(nIndex).Zc = pt_c[2];

				//world
				o_Image.pixels.at(nIndex).Xw = pt_w[0];
				o_Image.pixels.at(nIndex).Yw = pt_w[1];
				o_Image.pixels.at(nIndex).Zw = pt_w[2];

				//color
				//o_Image.pixels.at(nIndex).r = pt_w[0];
				//o_Image.pixels.at(nIndex).g = pt_w[1];
				//o_Image.pixels.at(nIndex).b = pt_w[2];
			}
}

void segmentpcd::_readBboxes(char const* i_strBboxPath, segmentpcd::Bboxes& bboxes)
{
	std::ifstream reader(i_strBboxPath);

	if (reader.is_open())
	{
		std::string line;
		while (std::getline(reader, line))
		{
			std::stringstream ss(line);
			std::string token;
			bboxes.bboxes.push_back(segmentpcd::Bbox());

			std::string index, left, right, size_x, size_y;
			if (!(ss >> index >> left >> right >> size_x >> size_y)) { break;}
			bboxes.bboxes.back().index = std::stoi(index);
			bboxes.bboxes.back().x_left = std::stoi(left);
			bboxes.bboxes.back().y_left = std::stoi(right); 
			bboxes.bboxes.back().size_x = std::stoi(size_x);
			bboxes.bboxes.back().size_y = std::stoi(size_y);
		}
		bboxes.num = bboxes.bboxes.size();
		reader.close();
	}
	else
	{
		printf("Could not open file \n");
	}
}


void segmentpcd::_read_mask(char const* i_strMaskPath, segmentpcd::Image& mask)
{
	//bmp reader

	std::ifstream stream(i_strMaskPath, std::ios::binary);
	if (!stream) { throw std::ios::failure("can't open file");}
	{
		BITMAPFILEHEADER bfh;
		BITMAPINFO bih;

		stream.read(reinterpret_cast<char*>(&bfh), sizeof(bfh));
		stream.read(reinterpret_cast<char*>(&bih), sizeof(bih));

		mask.width = (int)(WORD)bih.bmiHeader.biWidth;
		mask.height = (int)(WORD)bih.bmiHeader.biHeight;
		std::cout << mask.width << " " << mask.height;

		stream.seekg(bfh.bfOffBits, std::ios::beg);
		int padded_row_size = ((mask.width * bih.bmiHeader.biBitCount + 31) / 32) * 4;
		int imagesize = mask.height*padded_row_size;

		std::vector<uint8_t> data(imagesize);
		stream.read(reinterpret_cast<char*>(data.data()), data.size());

		std::vector<uint8_t> pixels;
		pixels.reserve(mask.width * mask.height * bih.bmiHeader.biBitCount/8);

		//invert
		for (int i = 0; i < mask.height; i++)
		{
			
			auto k = mask.height - 1 - i;
			auto ptr = reinterpret_cast<uint8_t*>(data.data()) + k * padded_row_size;
			pixels.insert(pixels.end(), ptr, ptr + (mask.width*bih.bmiHeader.biBitCount)/8);
		}

		//create mask
		//const char* sub = ".txt";
		//std::ofstream out((strcat((char*)i_strMaskPath, sub)));
		for (int i = 0; i < pixels.size(); i += 3)
		{
			int b = (int)pixels.at(i);
			int g = (int)pixels.at(i + 1);
			int r = (int)pixels.at(i + 2);
			segmentpcd::Pixel pixel;
			mask.pixels.push_back(pixel);
			mask.pixels.back().b = b;
			mask.pixels.back().g = g;
			mask.pixels.back().r = r;
			mask.pixels.back().grey = (int)(0.3 * r + 0.59 * g + 0.11 * b);

			//if (b > 200) { out << "1,"; }
			//else {out << "0,"; }
		}
	}
}


void segmentpcd::_get_masks(segmentpcd::_masks& masks, size_t& num_bboxes, int& scene_id) 
{
	for (int id = 0; id < num_bboxes; id++)
	{
		segmentpcd::Image mask;
		char buffer[50];
		int n = sprintf_s(buffer, "image/%d_%d_mask.bmp", scene_id, id);
		std::cout << buffer<<"\n";
		segmentpcd::_read_mask(buffer, mask);
		masks.masks.push_back(mask);
	}
}


void segmentpcd::_clip_bolt_pcd(segmentpcd::Bboxes& bboxes, segmentpcd::Image& o_image, segmentpcd::_masks& masks, segmentpcd::_outputClouds& o_pointclouds)
{
	for (int id = 0; id < bboxes.num; id++)
	{
		segmentpcd::Bbox bbox = bboxes.bboxes.at(id);

		//output pointcloud
		utils::PointCloud o_pointcloud;
		o_pointclouds.output_clouds.push_back(o_pointcloud);
		//std::cout << o_pointclouds.output_clouds.size();

		//char buffer[50];
		//int n = sprintf_s(buffer, "image/0_%d_pcd.txt", id);
		//std::ofstream out(buffer);

		for (int y_p = bbox.y_left; y_p < bbox.y_left + masks.masks.at(id).height; y_p++)
		{
			for (int x_p = bbox.x_left; x_p < bbox.x_left + masks.masks.at(id).width; x_p++)
			{
				//full image indices
				int nIndex = y_p * o_image.width + x_p;

				//mask indices
				int x_m = x_p - bbox.x_left, y_m = y_p - bbox.y_left;
				int nIndex_m = y_m * masks.masks.at(id).width + x_m;
				
				//at non-zero depth values only
				if (o_image.pixels.at(nIndex).Zw != 0)
				{
					if (masks.masks.at(id).pixels.at(nIndex_m).b > masks.thresh)
					{
						utils::point pt;
						o_pointclouds.output_clouds.back().points.push_back(pt);
						o_pointclouds.output_clouds.back().points.back().x = o_image.pixels.at(nIndex).Xw;
						o_pointclouds.output_clouds.back().points.back().y = o_image.pixels.at(nIndex).Yw;
						o_pointclouds.output_clouds.back().points.back().z = o_image.pixels.at(nIndex).Zw;
						//out<< o_image.pixels.at(nIndex).Xw << "," << o_image.pixels.at(nIndex).Yw << "," << o_image.pixels.at(nIndex).Zw << "\n";
					}	
				}	
			}
		}
	}
}

