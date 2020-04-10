#include "windows.h"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#ifdef _DEBUG
#pragma comment(lib,"opencv_world320d.lib")
#else
#pragma comment(lib,"opencv_world320.lib")
#endif

#include "polygon_demo.hpp"
#include "math.h"

using namespace std;
using namespace cv;


int main(int argc, char* argv[])
{

	Point3d p1(-0.5, 0, 2.121320);
	Point3d p2(0.5, 0, 2.121320);
	//Point3d p3(1, 1, 1);
	Point3d p3(0.5, -0.707107, 2.828427);
	//Point3d p3(0.5, 0.707107, 2.828427);
	//Point3d p1(0.5, -0.707107, 2.828427);
	//Point3d p2(0.5, 0.707107, 2.828427);
	//Point3d p3(-0.5,0,2.121320);



	Point3d p11(1.363005, -0.427130, 2.339082);
	Point3d p12(1.748084, 0.437983, 2.017688);
	//Point3d p13(0.680802,0.000003,0.955763);
	Point3d p13(2.636461, 0.184843, 2.400710);
	//Point3d p13(1.4981, 0.8710, 2.8837);
	//Point3d p11(2.636461, 0.184843, 2.400710);
	//Point3d p12(1.4981, 0.8710, 2.8837);
	//Point3d p13(1.363005, -0.427130, 2.339082);



	std::vector<cv::Point3d> pts;
	pts.push_back(p1);
	pts.push_back(p2);
	pts.push_back(p3);
	//print(pts);
	//printf("\n");

	std::vector<cv::Point3d> pts2;
	pts2.push_back(p11);
	pts2.push_back(p12);
	pts2.push_back(p13);
	//print(pts2);

	for (int i = 0; i < pts.size(); i++){
		pts[i].x = pts[i].x + 0.5;
		pts[i].z = pts[i].z - 2.121320;
		//pts[i].x = pts[i].x - 0.5;
		//pts[i].y = pts[i].y + 0.707107;
		//pts[i].z = pts[i].z - 2.828427;
	}
	//print(pts);

	//Vec p1p2
	Point3d u1(pts[1].x - pts[0].x, pts[1].y - pts[0].y, pts[1].z - pts[0].z);
	//Vec p1'p2'
	Point3d u11(pts2[1].x - pts2[0].x, pts2[1].y - pts2[0].y, pts2[1].z - pts2[0].z);

	//printf("p1'p2' : %lf %lf %lf\n",u11.x, u11.y, u11.z);
	//Vec p1p2, p1'p2'
	std::vector<cv::Point3d> pts3;
	pts3.push_back(u1);
	pts3.push_back(u11);
	//print(pts3);

	//printf("%lf\n", sqrt(pow(pts3[0].x, 2) + pow(pts3[0].y, 2) + pow(pts3[0].z, 2)));
	//printf("%lf\n", sqrt(pow(pts3[1].x, 2) + pow(pts3[1].y, 2) + pow(pts3[1].z, 2)));

	//xx' + yy' + zz'
	double D = (pts3[0].x*pts3[1].x) + (pts3[0].y*pts3[1].y) + (pts3[0].z*pts3[1].z);
	//||u|| * ||u'||
	double T = sqrt(pow(pts3[0].x, 2) + pow(pts3[0].y, 2) + pow(pts3[0].z, 2)) * sqrt(pow(pts3[1].x, 2) + pow(pts3[1].y, 2) + pow(pts3[1].z, 2));

	double cos1 = D / T;
	double sin1 = sqrt(1 - pow(cos1, 2));

	//uxu'
	Point3d cross_u(pts3[0].y*pts3[1].z - pts3[0].z*pts3[1].y, pts3[0].z*pts3[1].x - pts3[0].x*pts3[1].z, pts3[0].x*pts3[1].y - pts3[0].y*pts3[1].x);
	//||uxu'||
	double size_u = sqrt((pow(cross_u.x, 2) + pow(cross_u.y, 2) + pow(cross_u.z, 2)));
	//v
	Point3d v(cross_u.x / size_u, cross_u.y / size_u, cross_u.z / size_u);

	//printf("%lf\n", sqrt(pow(v.x, 2) + pow(v.y, 2) + pow(v.z, 2)));


	//Rotation Matrix(R1)
	Point3d R0(cos1 + (pow(v.x, 2)*(1 - cos1)), (v.x*v.y*(1 - cos1)) - (v.z*sin1), (v.x*v.z*(1 - cos1)) + v.y*sin1);
	Point3d R1(v.y*v.x*(1 - cos1) + (v.z*sin1), cos1 + (pow(v.y, 2)*(1 - cos1)), v.y*v.z*(1 - cos1) - v.x*sin1);
	Point3d R2(v.z*v.x*(1 - cos1) - v.y*sin1, v.z*v.y*(1 - cos1) + v.x*sin1, cos1 + (pow(v.z, 2)*(1 - cos1)));

	std::vector<cv::Point3d> R1pts;
	R1pts.push_back(R0);
	R1pts.push_back(R1);
	R1pts.push_back(R2);

	//print(R1pts);
	
	//Vec p1p3
	Point3d u2(pts[2].x - pts[0].x, pts[2].y - pts[0].y, pts[2].z - pts[0].z);
	//Vec p1'p3'
	Point3d u12(pts2[2].x - pts2[0].x, pts2[2].y - pts2[0].y, pts2[2].z - pts2[0].z);
	//Vec p1p2
	//Point3d u1(pts[1].x - pts[0].x, pts[1].y - pts[0].y, pts[1].z - pts[0].z);
	//Vec p1'p2'
	//Point3d u11(pts2[1].x - pts2[0].x, pts2[1].y - pts2[0].y, pts2[1].z - pts2[0].z);

	//R1p1p2
	Point3d R1p1p2(0, 0, 0);
	R1p1p2.x = R1pts[0].x*u1.x + R1pts[0].y*u1.y + R1pts[0].z*u1.z;
	R1p1p2.y = R1pts[1].x*u1.x + R1pts[1].y*u1.y + R1pts[1].z*u1.z;
	R1p1p2.z = R1pts[2].x*u1.x + R1pts[2].y*u1.y + R1pts[2].z*u1.z;


	printf("R1p1p2 : %lf %lf %lf\n", R1p1p2.x, R1p1p2.y, R1p1p2.z);
	printf("p1'p2' : %lf %lf %lf\n", u11.x, u11.y, u11.z);

	//R1p1p3
	Point3d R1p1p3(0, 0, 0);
	R1p1p3.x = R1pts[0].x*u2.x + R1pts[0].y*u2.y + R1pts[0].z*u2.z;
	R1p1p3.y = R1pts[1].x*u2.x + R1pts[1].y*u2.y + R1pts[1].z*u2.z;
	R1p1p3.z = R1pts[2].x*u2.x + R1pts[2].y*u2.y + R1pts[2].z*u2.z;

	printf("R1p1p3 : %lf %lf %lf\n", R1p1p3.x, R1p1p3.y, R1p1p3.z);
	printf("p1'p3' : %lf %lf %lf\n", u12.x, u12.y, u12.z);

	
	//R1p1p3, p1'p3'  ver1
	/*
	std::vector<cv::Point3d> pts4;
	pts4.push_back(R1p1p3);
	pts4.push_back(u12);
	*/

	//(R1*p1p2) x (R1*p1p3)
	Point3d R2_u(R1p1p2.y*R1p1p3.z - R1p1p2.z*R1p1p3.y, R1p1p2.z*R1p1p3.x - R1p1p2.x*R1p1p3.z, R1p1p2.x*R1p1p3.y - R1p1p2.y*R1p1p3.x);
	//(R1*p1p2) x (p1'p3')
	Point3d R22_u(R1p1p2.y*u12.z - R1p1p2.z*u12.y, R1p1p2.z*u12.x - R1p1p2.x*u12.z, R1p1p2.x*u12.y - R1p1p2.y*u12.x);
	printf("(R1*p1p2) x (R1*p1p3) , size = %lf\n", sqrt(pow(R2_u.x, 2) + pow(R2_u.y, 2) + pow(R2_u.z, 2)));
	printf("(R1*p1p2) x (p1'p3') , size = %lf\n", sqrt(pow(R22_u.x, 2) + pow(R22_u.y, 2) + pow(R22_u.z, 2)));
	//ver2
	std::vector<cv::Point3d> pts4;
	pts4.push_back(R2_u);
	pts4.push_back(R22_u);


	/*
	double DD = (pts4[0].x*pts4[1].x) + (pts4[0].y*pts4[1].y) + (pts4[0].z*pts4[1].z);
	double TT = sqrt(pow(pts4[0].x, 2) + pow(pts4[0].y, 2) + pow(pts4[0].z, 2)) * sqrt(pow(pts4[1].x, 2) + pow(pts4[1].y, 2) + pow(pts4[1].z, 2));

	double cos11 = DD / TT;
	double sin11 = sqrt(1 - pow(cos11, 2));

	Point3d cross_u2(pts4[0].y*pts4[1].z - pts4[0].z*pts4[1].y, pts4[0].z*pts4[1].x - pts4[0].x*pts4[1].z, pts4[0].x*pts4[1].y - pts4[0].y*pts4[1].x);
	double size_u2 = sqrt((pow(cross_u2.x, 2) + pow(cross_u2.y, 2) + pow(cross_u2.z, 2)));
	Point3d v2(cross_u2.x / size_u2, cross_u2.y / size_u2, cross_u2.z / size_u2);
	//v2는 R1*p1p2일수도 있다.
	//Point3d v2(R1p1p2.x, R1p1p2.y, R1p1p2.z);
	*/
	//printf("%lf", sqrt(pow(v2.x, 2) + pow(v2.y, 2) + pow(v2.z, 2)));

	double DD = (pts4[0].x*pts4[1].x) + (pts4[0].y*pts4[1].y) + (pts4[0].z*pts4[1].z);
	double TT = sqrt(pow(pts4[0].x, 2) + pow(pts4[0].y, 2) + pow(pts4[0].z, 2)) * sqrt(pow(pts4[1].x, 2) + pow(pts4[1].y, 2) + pow(pts4[1].z, 2));

	double cos11 = DD / TT;
	double sin11 = sqrt(1 - pow(cos11, 2));

	Point3d cross_u2(pts4[0].y*pts4[1].z - pts4[0].z*pts4[1].y, pts4[0].z*pts4[1].x - pts4[0].x*pts4[1].z, pts4[0].x*pts4[1].y - pts4[0].y*pts4[1].x);
	double size_u2 = sqrt((pow(cross_u2.x, 2) + pow(cross_u2.y, 2) + pow(cross_u2.z, 2)));
	Point3d v2(cross_u2.x / size_u2, cross_u2.y / size_u2, cross_u2.z / size_u2);
	
	printf("size_v2 : %lf\n", sqrt(pow(v2.x, 2) + pow(v2.y, 2) + pow(v2.z, 2)));
	
	
	//Rotation Matrix
	Point3d R00(cos11 + (pow(v2.x, 2)*(1 - cos11)), (v2.x*v2.y*(1 - cos11)) - (v2.z*sin11), (v2.x*v2.z*(1 - cos11)) + v2.y*sin11);
	Point3d R11(v2.y*v2.x*(1 - cos11) + (v2.z*sin11), cos11 + (pow(v2.y, 2)*(1 - cos11)), v2.y*v2.z*(1 - cos11) - v2.x*sin11);
	Point3d R22(v2.z*v2.x*(1 - cos11) - v2.y*sin11, v2.z*v2.y*(1 - cos11) + v2.x*sin11, cos11 + (pow(v2.z, 2)*(1 - cos11)));

	std::vector<cv::Point3d> R2pts;
	R2pts.push_back(R00);
	R2pts.push_back(R11);
	R2pts.push_back(R22);

	//print(R2pts);
	//R1p1'p3'
	//Point3d R1p1p3_2(0, 0, 0);
	//R1p1p3_2.x = R1pts[0].x*u12.x + R1pts[0].y*u12.y + R1pts[0].z*u12.z;
	//R1p1p3_2.y = R1pts[1].x*u12.x + R1pts[1].y*u12.y + R1pts[1].z*u12.z;
	//R1p1p3_2.z = R1pts[2].x*u12.x + R1pts[2].y*u12.y + R1pts[2].z*u12.z;

	//printf("%lf\n", sqrt(pow(R1p1p3.x, 2) + pow(R1p1p3.y, 2) + pow(R1p1p3.z, 2)));
	//printf("%lf\n", sqrt(pow(R1p1p3_2.x, 2) + pow(R1p1p3_2.y, 2) + pow(R1p1p3_2.z, 2)));
	//printf("p1'p3' %lf\n", sqrt(pow(u12.x, 2) + pow(u12.y, 2) + pow(u12.z, 2)));


	//검증 : R2(R1P1P2 x R1P1P3) = (R1P1P2 x P1'P3')
	Point3d R1p1p32(0, 0, 0);
	R1p1p32.x = R2pts[0].x*R2_u.x + R2pts[0].y*R2_u.y + R2pts[0].z*R2_u.z;
	R1p1p32.y = R2pts[1].x*R2_u.x + R2pts[1].y*R2_u.y + R2pts[1].z*R2_u.z;
	R1p1p32.z = R2pts[2].x*R2_u.x + R2pts[2].y*R2_u.y + R2pts[2].z*R2_u.z;


	std::vector<cv::Point3d> pts5;
	pts5.push_back(R2_u);
	pts5.push_back(R22_u);

	printf("(R2(R1P1P2XR1P1P3) : %lf %lf %lf\n", R1p1p32.x, R1p1p32.y, R1p1p32.z);
	printf("(R1*p1p2) x (p1'p3') : %lf %lf %lf\n ", R22_u.x, R22_u.y, R22_u.z);

	

	//printf("(R1*p1p2) x (R1*p1p3) : %lf %lf %lf\n ", R2_u.x, R2_u.y, R2_u.z);
	//printf("%lf %lf %lf\n", R1p1p32.x + 1.363005, R1p1p32.y - 0.427130, R1p1p32.z + 2.339082);


	//printf("%lf\n", sqrt(pow(pts4[0].x, 2) + pow(pts4[0].y, 2) + pow(pts4[0].z, 2)));
	//printf("%lf\n", sqrt(pow(pts4[1].x, 2) + pow(pts4[1].y, 2) + pow(pts4[1].z, 2)));



	return 0;
}