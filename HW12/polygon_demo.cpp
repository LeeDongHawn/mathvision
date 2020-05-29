#include "polygon_demo.hpp"
#include "opencv2/imgproc.hpp"
#include "math.h"


using namespace std;
using namespace cv;

PolygonDemo::PolygonDemo()
{
	m_data_ready = false;
}

PolygonDemo::~PolygonDemo()
{
}

void PolygonDemo::refreshWindow()
{
	Mat frame = Mat::zeros(480, 640, CV_8UC3);

	if (!m_data_ready)
		putText(frame, "Input data points (double click: finish)", Point(10, 470), FONT_HERSHEY_SIMPLEX, .5, Scalar(0, 148, 0), 1);

	drawPolygon(frame, m_data_pts, m_data_ready);
	if (m_data_ready)
	{
		// polygon area
		if (m_param.compute_area)
		{
			int area = polyArea(m_data_pts);
			char str[100];
			sprintf_s(str, 100, "Area = %d", area);
			putText(frame, str, Point(25, 25), FONT_HERSHEY_SIMPLEX, .8, Scalar(0, 255, 255), 1);
		}

		// pt in polygon
		if (m_param.check_ptInPoly)
		{
			for (int i = 0; i < (int)m_test_pts.size(); i++)
			{
				if (ptInPolygon(m_data_pts, m_test_pts[i]))
				{
					circle(frame, m_test_pts[i], 2, Scalar(0, 255, 0), CV_FILLED);
				}
				else
				{
					circle(frame, m_test_pts[i], 2, Scalar(128, 128, 128), CV_FILLED);
				}
			}
		}

		// homography check
		if (m_param.check_homography && m_data_pts.size() == 4)
		{
			// rect points
			int rect_sz = 100;
			vector<Point> rc_pts;
			rc_pts.push_back(Point(0, 0));
			rc_pts.push_back(Point(0, rect_sz));
			rc_pts.push_back(Point(rect_sz, rect_sz));
			rc_pts.push_back(Point(rect_sz, 0));
			rectangle(frame, Rect(0, 0, rect_sz, rect_sz), Scalar(255, 255, 255), 1);

			// draw mapping
			char* abcd[4] = { "A", "B", "C", "D" };
			for (int i = 0; i < 4; i++)
			{
				line(frame, rc_pts[i], m_data_pts[i], Scalar(255, 0, 0), 1);
				circle(frame, rc_pts[i], 2, Scalar(0, 255, 0), CV_FILLED);
				circle(frame, m_data_pts[i], 2, Scalar(0, 255, 0), CV_FILLED);
				putText(frame, abcd[i], m_data_pts[i], FONT_HERSHEY_SIMPLEX, .8, Scalar(0, 255, 255), 1);
			}

			// check homography
			int homo_type = classifyHomography(rc_pts, m_data_pts);
			char type_str[100];
			switch (homo_type)
			{
			case NORMAL:
				sprintf_s(type_str, 100, "normal");
				break;
			case CONCAVE:
				sprintf_s(type_str, 100, "concave");
				break;
			case TWIST:
				sprintf_s(type_str, 100, "twist");
				break;
			case REFLECTION:
				sprintf_s(type_str, 100, "reflection");
				break;
			case CONCAVE_REFLECTION:
				sprintf_s(type_str, 100, "concave reflection");
				break;
			}

			putText(frame, type_str, Point(15, 125), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 255), 1);
		}

		// fit circle
		if (m_param.fit_circle)
		{
			Point2d center;
			double radius = 0;
			bool ok = fitCircle(m_data_pts, center, radius);
			if (ok)
			{
				circle(frame, center, (int)(radius + 0.5), Scalar(0, 255, 0), 1);
				circle(frame, center, 2, Scalar(0, 255, 0), CV_FILLED);
			}
		}

		//fit ellipse
		if (m_param.fit_ellipse)
		{
			Point m;
			Point v;
			float theta = 0;

			bool ok = fitEllipse(m_data_pts, m, v, theta);
			if (ok)
			{
				//img / center / axes / angle / startAngle / endAngle / color / thickness / lineType 
				ellipse(frame, m, v, theta, 0, 360, Scalar(0, 255, 0), 3, 8);
			}
		}

		//fit line
		if (m_param.fit_green_red_line)
		{
			//fit green line(y=ax+b)
			Point2d point1;
			Point2d point2;
			//fit red line(ax+by+c=0)
			Point2d point3;
			Point2d point4;

			bool ok = fit_green_red_line(m_data_pts, point1, point2, point3, point4);

			if (ok)
			{
				//draw line
				line(frame, point1, point2, Scalar(0, 255, 0), 1, 1);
				line(frame, point3, point4, Scalar(0, 0, 255), 1, 1);
			}


		}

		//fit robust
		if (m_param.fit_robust)
		{
			Point2d point1, point2;
			Point2d point3, point4;
			int iteration = 0;
			std::vector<cv::Point2d> point5, point6;
			bool ok = fit_green_red_line(m_data_pts, point1, point2, point3, point4);
			bool ok_robust = fit_robust(m_data_pts, point5, point6, iteration);
			
			if (ok){
				// ax+b
				line(frame, point1, point2, Scalar(0, 255, 0), 1);
				putText(frame, "y=ax+b", Point(5, 25), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 255, 0), 1);
			}
			
			if (ok_robust){
				for (int n = 0; n < iteration; n++)
				{
					// robust ax+b
					string text_ = "iteration : " + std::to_string(n + 1);
					line(frame, point5[n], point6[n], Scalar(0, 255 / iteration*n, 255), 1);
					putText(frame, text_, Point(5, 45 + n * 20), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 255 / iteration*n, 255), 1);
				}

			}
		}

		if (m_param.fit_ransac)
		{

			Point2d point1, point2;
			Point2d point3, point4;
			Point2d point9, point10;
			std::vector<cv::Point2d> point11, point12, point13, point14;
			int iteration = 0;
			bool ok = fit_green_red_line(m_data_pts, point1, point2, point3, point4);
			//bool ok_ransac = fit_ransac(m_data_pts, point9, point10, iteration);

			bool ok_ransac2 = fit_ransac2(m_data_pts, point11, point12, iteration);

			if (ok){
				// ax+by+c=0
				line(frame, point3, point4, Scalar(0, 255, 0), 1, 1);
				putText(frame, "ax+by+c=0", Point(5, 25), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 255, 0), 1);
			}

			/*
			if (ok_ransac) {
				// ransac_ ax+by+c=0
				line(frame, point9, point10, Scalar(255, 0, 0), 3, 3);
				putText(frame, "ransac_ ax+by+c=0", Point(5, 45), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 0, 0), 1);
			}
			*/


			//ransac_ax+by+c=0
			if (ok_ransac2){
				string text_ = "iteration : " + std::to_string(iteration);
				putText(frame, text_, Point(150, 25), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 0, 0), 1);
				for (int n = 0; n < iteration; n++)
				{
					if (n == iteration-1){
						string text_ = "best_ransac_iteration : " + std::to_string(n + 1);
						line(frame, point11[n], point12[n], Scalar(255, 0, 0), 1);
						putText(frame, text_, Point(5, 65 + n * 20), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 0, 0), 1);
					}
					else {
						string text_ = "iteration : " + std::to_string(n + 1);
						line(frame, point11[n], point12[n], Scalar(0, 255 / iteration*n, 255), 1);
						//line(frame, point13[n], point14[n], Scalar(255 / iteration*n, 0 , 255), 1);
						putText(frame, text_, Point(5, 65 + n * 20), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 255 / iteration*n, 255), 1);
					}
					
				}

			}
			
		}
		

	}

	imshow("PolygonDemo", frame);
}


// return the area of polygon
int PolygonDemo::polyArea(const std::vector<cv::Point>& vtx)
{
	print(vtx);
	printf("\n");
	//printf("x0 : %d, y0 : %d", vtx[0],vtx[1]);

	//printf("%d", vtx.size()); vtx size는 x,y를 한 쌍으로 묶은 것임.

	//두개 더하면 index가 변화함 즉 vtx[0+1] 연산을 수행함
	//printf("%d + %d = %d\n", vtx[0], vtx[1], vtx[0] + vtx[1]);
	//printf("%d + %d = %d\n", vtx.at(0), vtx.at(1), vtx.at(0) + vtx.at(1));


	//printf("%d\n", vtx[0].x);
	//printf("%d\n", vtx[0].y);

	int i = 1;
	int s = 0;

	//공식대로 값을 넣어준다.
	for (i; i < vtx.size() - 1; i++) {

		s = s + (((vtx[i].x - vtx[0].x) * (vtx[i + 1].y - vtx[0].y)) - ((vtx[i + 1].x - vtx[0].x) *  (vtx[i].y - vtx[0].y))) / 2;
		printf("(%d-%d)*(%d-%d) - (%d-%d)*(%d-%d)\n", vtx[i].x, vtx[0].x, vtx[i + 1].y, vtx[0].y, vtx[i + 1].x, vtx[0].x, vtx[i].y, vtx[0].y);
		printf("%d*%d - %d*%d\n", vtx[i].x - vtx[0].x, vtx[i + 1].y - vtx[0].y, vtx[i + 1].x - vtx[0].x, vtx[i].y - vtx[0].y);
		printf("%d - %d = %d\n", (vtx[i].x - vtx[0].x) * (vtx[i + 1].y - vtx[0].y), (vtx[i + 1].x - vtx[0].x) *  (vtx[i].y - vtx[0].y),
			(vtx[i].x - vtx[0].x) * (vtx[i + 1].y - vtx[0].y) - (vtx[i + 1].x - vtx[0].x) *  (vtx[i].y - vtx[0].y));
		printf("result/2=%d\n", (((vtx[i].x - vtx[0].x) * (vtx[i + 1].y - vtx[0].y)) - ((vtx[i + 1].x - vtx[0].x) *  (vtx[i].y - vtx[0].y))) / 2);
	}


	//절대값을 취한다.
	return abs(s);
}

// return true if pt is interior point
bool PolygonDemo::ptInPolygon(const std::vector<cv::Point>& vtx, Point pt)
{
	return false;
}

// return homography type: NORMAL, CONCAVE, TWIST, REFLECTION, CONCAVE_REFLECTION
int PolygonDemo::classifyHomography(const std::vector<cv::Point>& pts1, const std::vector<cv::Point>& pts2)
{
	//point가 4개(x0,y0) (x1,y1) (x2,y2) (x3,y3)가 아니면 -1을 반환한다.
	if (pts1.size() != 4 || pts2.size() != 4) return -1;

	//pts1 : 작은사각형 좌표
	print(pts1);
	//pts2 : 화면 그리는 사각형 좌표
	printf("%d %d %d %d\n", pts2[0], pts2[1], pts2[2], pts2[3]);

	return NORMAL;
}




// estimate a circle that best approximates the input points and return center and radius of the estimate circle
bool PolygonDemo::fitCircle(const std::vector<cv::Point>& pts, cv::Point2d& center, double& radius)
{
	int n = (int)pts.size();
	if (n < 3) return false;

	Mat A = Mat::zeros(n, 3, CV_32FC1);
	Mat x = Mat::zeros(3, 1, CV_32FC1);
	Mat b = Mat::zeros(n, 1, CV_32FC1);

	for (int i = 0; i < pts.size(); i++)
	{
		A.at<float>(i, 0) = -2 * pts[i].x;
		A.at<float>(i, 1) = -2 * pts[i].y;
		A.at<float>(i, 2) = 1;
		b.at<float>(i, 0) = -(pts[i].x*pts[i].x + pts[i].y*pts[i].y);
	}

	Mat Apinv;
	invert(A, Apinv, DECOMP_SVD);
	x = Apinv*b;

	center.x = x.at<float>(0, 0);
	center.y = x.at<float>(1, 0);
	radius = sqrt(x.at<float>(0, 0)*x.at<float>(0, 0) + x.at<float>(1, 0)*x.at<float>(1, 0) - x.at<float>(2, 0));

	Mat test1 = Mat::zeros(n, 1, CV_32FC1);

	test1 = A*x;
	printf("\n");
	printf("A*x\n");
	print(test1);
	printf("\n");
	printf("b\n");
	print(b);
	printf("\n");
}


// estimate a ellipse that best approximates the input points and return center and axes and angle of the estimate ellipse
bool PolygonDemo::fitEllipse(const std::vector<cv::Point>& pts, cv::Point& m, cv::Point& v, float& theta)
{

	int n = (int)pts.size();
	if (n < 5) return false;


	Mat A = Mat::zeros(n, 6, CV_32FC1);
	Mat x = Mat::zeros(6, 1, CV_32FC1);
	Mat b = Mat::zeros(n, 1, CV_32FC1);


	for (int i = 0; i < pts.size(); i++)

	{
		A.at<float>(i, 0) = pts[i].x*pts[i].x;
		A.at<float>(i, 1) = pts[i].x*pts[i].y;
		A.at<float>(i, 2) = pts[i].y*pts[i].y;
		A.at<float>(i, 3) = pts[i].x;
		A.at<float>(i, 4) = pts[i].y;
		A.at<float>(i, 5) = 1;

	}

	Mat w, u, vt;

	SVD::compute(A, w, u, vt, SVD::FULL_UV);

	printf("\n");
	printf("Singular value : \n");
	print(w);


	printf("A mat : \n");
	print(A);
	printf("\n");
	printf("Vt mat : \n");
	print(vt);
	printf("\n");
	Mat Avt = A*vt;
	printf("A*vt mat : \n");
	print(Avt);
	printf("\n");
	print(u);

	printf("\n");
	Mat vav = u*w;
	print(vav);
	printf("\n");


	/*
	printf("A*vt = w*u\n");
	Mat Avt = A*vt;
	Mat wu = w*u;
	print(Avt);
	printf("\n");
	print(wu);
	printf("\n");
	*/

	Mat vt_t;
	vt_t = vt.t();
	//printf("Vt\n");
	//print(vt);
	//printf("\n");
	//printf("Vt_t\n");
	//print(vt_t);
	//printf("\n");

	for (int i = 0; i < 6; i++)
	{
		x.at<float>(i, 0) = vt_t.at<float>(i, 5);
	}

	//printf("x\n");
	//print(x);

	// center / axes / angle
	float a1 = x.at<float>(0, 0);
	float b1 = x.at<float>(1, 0);
	float c1 = x.at<float>(2, 0);
	float d1 = x.at<float>(3, 0);
	float e1 = x.at<float>(4, 0);
	float f1 = x.at<float>(5, 0);

	theta = 0.5*atan(b1 / (a1 - c1));

	float cx = (2 * c1*d1 - b1*e1) / (b1*b1 - 4 * a1*c1);
	float cy = (2 * a1*e1 - b1*d1) / (b1*b1 - 4 * a1*c1);
	float cu = a1*cx*cx + b1*cx*cy + c1*cy*cy - f1;
	float width = sqrt(cu / (a1*cos(theta)*cos(theta) + b1* cos(theta)*sin(theta) + c1*sin(theta)*sin(theta)));
	float height = sqrt(cu / (a1*sin(theta)*sin(theta) - b1* cos(theta)*sin(theta) + c1*cos(theta)*cos(theta)));

	theta *= 180 / 3.14;

	m.x = cx;
	m.y = cy;
	v.x = width;
	v.y = height;

	Mat result_check = A*x;
	printf("A*x\n");
	print(result_check);
	return 1;



}

// estimate a ellipse that best approximates the input points and return center and axes and angle of the estimate y=ax+b, ax+by+c=0
bool PolygonDemo::fit_green_red_line(const std::vector<cv::Point>& pts, cv::Point2d& point1, cv::Point2d& point2, cv::Point2d& point3, cv::Point2d& point4)
{
	//part1_green_line(y=ax+b)
	int n = (int)pts.size();
	if (n < 2) return false;

	Mat A = Mat::zeros(n, 2, CV_32FC1);
	Mat x = Mat::zeros(2, 1, CV_32FC1);
	Mat b = Mat::zeros(n, 1, CV_32FC1);

	for (int i = 0; i < pts.size(); i++)

	{
		A.at<float>(i, 0) = pts[i].x;
		A.at<float>(i, 1) = 1;
		b.at<float>(i, 0) = pts[i].y;
	}

	//print(pts);
	Mat Apinv;
	invert(A, Apinv, DECOMP_SVD);
	x = Apinv*b;
	//print(x);

	Mat ones = Mat::ones(1, 1, CV_32FC1);
	Mat solution = x*ones;

	//print(solution);

	point1.x = 0;
	point2.x = 640;
	point1.y = solution.at<float>(0, 0)*point1.x + solution.at<float>(1, 0);
	point2.y = solution.at<float>(0, 0)*point2.x + solution.at<float>(1, 0);

	//printf("p1.x : %f : ", point1.x);
	//printf("p1.y : %f : ", point1.y);
	//printf("p2.x : %f : ", point2.x);
	//printf("p2.y : %f : ", point2.y);
	//printf("\n");


	//part2_red_line(ax+by+c=0)
	Mat A2 = Mat::zeros(n, 3, CV_32FC1);
	Mat x2 = Mat::zeros(3, 1, CV_32FC1);
	Mat b2 = Mat::zeros(n, 1, CV_32FC1);

	for (int i = 0; i < pts.size(); i++)
	{
		A2.at<float>(i, 0) = pts[i].x;
		A2.at<float>(i, 1) = pts[i].y;
		A2.at<float>(i, 2) = 1;
		b2.at<float>(i, 0) = 0;
	}

	Mat w, u, vt, vt_t;

	SVD::compute(A2, w, u, vt, SVD::FULL_UV);
	/*
	printf("Singular value : \n");
	print(w);
	printf("\n");

	printf("Vt mat : \n");
	print(vt);
	printf("\n");

	
	printf("Vt_t mat : \n");
	print(vt_t);
	printf("\n");
	*/

	transpose(vt, vt_t);

	for (int i = 0; i < 3; i++)
	{
		x2.at<float>(i, 0) = vt_t.at<float>(i, 2);
	}

	//print(x2);


	Mat ones2 = Mat::ones(1, 1, CV_32FC1);
	Mat solution2 = x2*ones2;

	print(solution2);
	printf("\n");

	point3.x = 0;
	point4.x = 640;
	//y = -((a*x/b)+(c/b))
	point3.y = -1 * ((solution2.at<float>(0, 0)*point3.x / solution2.at<float>(1, 0)) + (solution2.at<float>(2, 0) / solution2.at<float>(1, 0)));
	point4.y = -1 * ((solution2.at<float>(0, 0)*point4.x / solution2.at<float>(1, 0)) + (solution2.at<float>(2, 0) / solution2.at<float>(1, 0)));

	//printf("p3.x : %f : ", point3.x);
	printf("p3.y : %f : \n", point3.y);
	//printf("p4.x : %f : ", point4.x);
	printf("p4.y : %f : \n", point4.y);



	return 1;

}


bool PolygonDemo::fit_robust(const std::vector<cv::Point>& pts, std::vector<cv::Point2d>& point5, std::vector<cv::Point2d>& point6, int& iteration)
{

	
	int n = (int)pts.size();
	if (n < 2) return false;
	//print(pts);

	Mat A = Mat::zeros(n, 2, CV_32FC1);
	Mat p = Mat::zeros(2, 1, CV_32FC1);
	Mat y = Mat::zeros(n, 1, CV_32FC1);

	for (int i = 0; i < pts.size(); i++)

	{
		A.at<float>(i, 0) = pts[i].x;
		A.at<float>(i, 1) = 1;
		y.at<float>(i, 0) = pts[i].y;
	}

	//print(pts);
	Mat Apinv;
	invert(A, Apinv, DECOMP_SVD);
	p = Apinv*y;
	//print(p);

	
	Mat r;
	
	Mat w_r, AtWA, AtWA_inv;
	Mat W = Mat::zeros(n, n, CV_32FC1);
	double a, b;

	for (int i = 0; i < iteration; i++){
		printf("iteration : %d\n", i);
		// residual matrix
		r = y - A*p; 
		// cauchu weight matrix
		w_r = 1 / (abs(r) / 1.3998 + 1); 
		
		for (int j = 0; j < n; j++) {
			// Weight matrix
			W.at<float>(j, j) = w_r.at<float>(j, 0); 
		}
		
		printf("\n");
		print(W);
		AtWA = A.t() * W * A;
		invert(AtWA, AtWA_inv, DECOMP_SVD);
		p = AtWA_inv * A.t() * W * y;


		//좌표 반영 시 ones 행렬 곱해주기
		print(p);
		Mat ones = Mat::ones(1, 1, CV_32FC1);
		Mat solution = p*ones;
		printf("\n");
		//print(solution);
		//printf("\n");
		a = solution.at<float>(0, 0);
		b = solution.at<float>(1, 0);
		printf("a    %lf   : \n", a);
		printf("b    %lf   : \n", b);


		point5.push_back(cv::Point2d(0, a * 0 + b));
		point6.push_back(cv::Point2d(640, a * 640 + b));
		

	}

	
	return 1;
}

bool PolygonDemo::fit_ransac(const std::vector<cv::Point>& pts, cv::Point2d& point9, cv::Point2d& point10, int& iteration){

	int n = (int)pts.size();
	if (n < 2) return false;
	int sigma = 100;
	int T = 2 * sigma;
	int max_cnt = 0;

	double alpha = 0.8;
	double p = 0.977;
	int sample = 2;
	//print(pts);
	//printf("%d",pts[0].x);
	//printf("%d", pts[0].y);

	//반복 횟수
	iteration = (int)ceilf(log(1 - p) / log(1 - pow(1 - alpha, sample)));
	iteration = 3;
	printf("%d", iteration);

	// residual과 실제 점 좌표
	Mat r;
	Mat A1 = Mat::zeros(n, 3, CV_32FC1);
	Mat x1 = Mat::zeros(3, 1, CV_32FC1);
	Mat b1 = Mat::zeros(n, 1, CV_32FC1);
	Mat best_model = Mat::zeros(n, 1, CV_32FC1);

	for (int i = 0; i < n; i++)
	{
		A1.at<float>(i, 0) = pts[i].x;
		A1.at<float>(i, 1) = pts[i].y;
		A1.at<float>(i, 2) = 1;
		b1.at<float>(i, 0) = 0;
	}


	for (int i = 0; i < iteration; i++){

		// 일단 뽑을 2개의 점의 index를 결정한다.
		int k1 = rand() % n;
		int k2 = rand() % n;

		// 무작위 두 개의 점을 뽑는다.
		Point2d random_p1, random_p2;
		random_p1.x = pts[k1].x;
		random_p1.y = pts[k1].y;
		random_p2.x = pts[k2].x;
		random_p2.y = pts[k2].y;

		vector<cv::Point> random;
		random.push_back(random_p1);
		random.push_back(random_p2);
		//printf("%d : \n",random[0].x);
		//printf("%d : \n", random[0].y);
		//printf("%d : \n", random[1].x);
		//printf("%d : \n", random[1].y);

		//printf("----------------\n");

		// 모델 추정(위에서 추출한 점을 지나는 직선)
		//part2_red_line(ax+by+c=0)
		Mat A2 = Mat::zeros(n, 3, CV_32FC1);
		Mat x2 = Mat::zeros(3, 1, CV_32FC1);
		Mat b2 = Mat::zeros(n, 1, CV_32FC1);

		for (int i = 0; i < 2; i++)
		{
			A2.at<float>(i, 0) = random[i].x;
			A2.at<float>(i, 1) = random[i].y;
			A2.at<float>(i, 2) = 1;
			b2.at<float>(i, 0) = 0;
		}
		Mat w, u, vt, vt_t;
		SVD::compute(A2, w, u, vt, SVD::FULL_UV);
		transpose(vt, vt_t);
		for (int i = 0; i < 3; i++)
		{
			x2.at<float>(i, 0) = vt_t.at<float>(i, 2);
		}

		
		// residual matrix 
		r = abs(b1 - A1*x2);
		// threshold 보다 작은 데이터의 개수 cnt
		int cnt;
		for (int j = 0; j < n; j++){
			if (r.at<float>(j,0) < T){
				cnt++;
				best_model.at<float>(j, 0) = 1;
			}

		}
		//
		if (cnt > max_cnt) {
			max_cnt = cnt;
			best_model = x2;
		}
	}


	Mat ones = Mat::ones(1, 1, CV_32FC1);
	Mat solution = best_model*ones;
	
	print(solution);
	printf("\n");
	float a = solution.at<float>(0, 0);
	double b = solution.at<float>(1, 0);
	double c = solution.at<float>(2, 0);
	printf("a    %lf   : \n", a);
	printf("b    %lf   : \n", b);
	printf("c    %lf   : \n", c);
	printf("%f\n", solution.at<float>(1, 0));
	//y = -((a*x/b)+(c/b))
	point9.x = 0;
	point10.x = 640;
	point9.y = -1 * ((a * 0 / b) + (c / b));
	point10.y = -1 * ((a * 640 / b) + (c / b));
	
	printf("%lf\n",point9.y);
	printf("%lf\n",point10.y);

	return 1;
}


bool PolygonDemo::fit_ransac2(const std::vector<cv::Point>& pts,  std::vector<cv::Point2d>& point11, std::vector<cv::Point2d>& point12, int& iteration)
{

	int n = (int)pts.size();
	if (n < 2) return false;
	int sigma = 100;
	int T = 2 * sigma;
	int max_cnt = 0;

	double alpha = 0.8;
	double p = 0.977;
	int sample = 2;
	//print(pts);
	//printf("%d",pts[0].x);
	//printf("%d", pts[0].y);

	//반복 횟수
	//iteration = (int)ceilf(log(1 - p) / log(1 - pow(1 - alpha, sample)));
	iteration = 20;
	
	// residual과 실제 점 좌표
	Mat r;
	Mat A1 = Mat::zeros(n, 3, CV_32FC1);
	Mat x1 = Mat::zeros(3, 1, CV_32FC1);
	Mat b1 = Mat::zeros(n, 1, CV_32FC1);
	Mat best_model = Mat::zeros(n, 1, CV_32FC1);

	for (int i = 0; i < n; i++)
	{
		A1.at<float>(i, 0) = pts[i].x;
		A1.at<float>(i, 1) = pts[i].y;
		A1.at<float>(i, 2) = 1;
		b1.at<float>(i, 0) = 0;
	}


	for (int i = 0; i < iteration; i++){

		// 일단 뽑을 2개의 점의 index를 결정한다.
		int k1 = rand() % n;
		int k2 = rand() % n;

		// 무작위 두 개의 점을 뽑는다.
		Point2d random_p1, random_p2;
		random_p1.x = pts[k1].x;
		random_p1.y = pts[k1].y;
		random_p2.x = pts[k2].x;
		random_p2.y = pts[k2].y;

		vector<cv::Point> random;
		random.push_back(random_p1);
		random.push_back(random_p2);
		//printf("%d : \n",random[0].x);
		//printf("%d : \n", random[0].y);
		//printf("%d : \n", random[1].x);
		//printf("%d : \n", random[1].y);

		//printf("----------------\n");

		// 모델 추정(위에서 추출한 점을 지나는 직선)
		//part2_red_line(ax+by+c=0)
		Mat A2 = Mat::zeros(n, 3, CV_32FC1);
		Mat x2 = Mat::zeros(3, 1, CV_32FC1);
		Mat b2 = Mat::zeros(n, 1, CV_32FC1);

		for (int i = 0; i < 2; i++)
		{
			A2.at<float>(i, 0) = random[i].x;
			A2.at<float>(i, 1) = random[i].y;
			A2.at<float>(i, 2) = 1;
			b2.at<float>(i, 0) = 0;
		}
		Mat w, u, vt, vt_t;
		SVD::compute(A2, w, u, vt, SVD::FULL_UV);
		transpose(vt, vt_t);
		for (int i = 0; i < 3; i++)
		{
			x2.at<float>(i, 0) = vt_t.at<float>(i, 2);
		}


		// residual matrix 
		r = abs(b1 - A1*x2);
		// threshold 보다 작은 데이터의 개수 cnt
		int cnt;
		for (int j = 0; j < n; j++){
			if (r.at<float>(j, 0) < T){
				cnt++;
			}

		}

		// 만일 cnt가 max_cnt보다 크다면 현재 f(x)를 저장한다.
		if (cnt > max_cnt) {
			max_cnt = cnt;
			best_model = x2;
		}

		Mat ones = Mat::ones(1, 1, CV_32FC1);
		Mat solution = best_model*ones;

		print(solution);
		printf("\n");
		float a = solution.at<float>(0, 0);
		double b = solution.at<float>(1, 0);
		double c = solution.at<float>(2, 0);
		printf("a    %lf   : \n", a);
		printf("b    %lf   : \n", b);
		printf("c    %lf   : \n", c);
		printf("%f\n", solution.at<float>(1, 0));
		//y = -((a*x/b)+(c/b))
		point11.push_back(cv::Point2d(0, -1 * ((a * 0 / b) + (c / b))));
		point12.push_back(cv::Point2d(640, -1 * ((a * 640 / b) + (c / b))));

	}

	return 1;
}

void PolygonDemo::drawPolygon(Mat& frame, const std::vector<cv::Point>& vtx, bool closed)
{
	int i = 0;
	for (i = 0; i < (int)m_data_pts.size(); i++)
	{
		circle(frame, m_data_pts[i], 2, Scalar(255, 255, 255), CV_FILLED);
	}
	for (i = 0; i < (int)m_data_pts.size() - 1; i++)
	{
		//line(frame, m_data_pts[i], m_data_pts[i + 1], Scalar(255, 255, 255), 1);
	}
	if (closed)
	{
		//line(frame, m_data_pts[i], m_data_pts[0], Scalar(255, 255, 255), 1);
	}
}

void PolygonDemo::handleMouseEvent(int evt, int x, int y, int flags)
{
	if (evt == CV_EVENT_LBUTTONDOWN)
	{
		if (!m_data_ready)
		{
			m_data_pts.push_back(Point(x, y));
		}
		else
		{
			m_test_pts.push_back(Point(x, y));
		}
		refreshWindow();
	}
	else if (evt == CV_EVENT_LBUTTONUP)
	{
	}
	else if (evt == CV_EVENT_LBUTTONDBLCLK)
	{
		m_data_ready = true;
		refreshWindow();
	}
	else if (evt == CV_EVENT_RBUTTONDBLCLK)
	{
	}
	else if (evt == CV_EVENT_MOUSEMOVE)
	{
	}
	else if (evt == CV_EVENT_RBUTTONDOWN)
	{
		m_data_pts.clear();
		m_test_pts.clear();
		m_data_ready = false;
		refreshWindow();
	}
	else if (evt == CV_EVENT_RBUTTONUP)
	{
	}
	else if (evt == CV_EVENT_MBUTTONDOWN)
	{
	}
	else if (evt == CV_EVENT_MBUTTONUP)
	{
	}

	if (flags&CV_EVENT_FLAG_LBUTTON)
	{
	}
	if (flags&CV_EVENT_FLAG_RBUTTON)
	{
	}
	if (flags&CV_EVENT_FLAG_MBUTTON)
	{
	}
	if (flags&CV_EVENT_FLAG_CTRLKEY)
	{
	}
	if (flags&CV_EVENT_FLAG_SHIFTKEY)
	{
	}
	if (flags&CV_EVENT_FLAG_ALTKEY)
	{
	}
}