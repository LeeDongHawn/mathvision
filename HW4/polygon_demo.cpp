#include "polygon_demo.hpp"
#include "opencv2/imgproc.hpp"
#include <opencv2/opencv.hpp>

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
    }

    imshow("PolygonDemo", frame);
}

// return the area of polygon
int PolygonDemo::polyArea(const std::vector<cv::Point>& vtx)
{

	int i = 1;
	int s = 0;

	//cal area
	for (i; i < vtx.size() - 1; i++) {

		s = s + (((vtx[i].x - vtx[0].x) * (vtx[i + 1].y - vtx[0].y)) - ((vtx[i + 1].x - vtx[0].x) *  (vtx[i].y - vtx[0].y))) / 2;
		//printf("(%d-%d)*(%d-%d) - (%d-%d)*(%d-%d)\n", vtx[i].x, vtx[0].x, vtx[i + 1].y, vtx[0].y, vtx[i + 1].x, vtx[0].x, vtx[i].y, vtx[0].y);
		//printf("%d*%d - %d*%d\n", vtx[i].x - vtx[0].x, vtx[i + 1].y - vtx[0].y, vtx[i + 1].x - vtx[0].x, vtx[i].y - vtx[0].y);
		//printf("%d - %d = %d\n", (vtx[i].x - vtx[0].x) * (vtx[i + 1].y - vtx[0].y), (vtx[i + 1].x - vtx[0].x) *  (vtx[i].y - vtx[0].y),
		//	(vtx[i].x - vtx[0].x) * (vtx[i + 1].y - vtx[0].y) - (vtx[i + 1].x - vtx[0].x) *  (vtx[i].y - vtx[0].y));
		//printf("result/2=%d\n", (((vtx[i].x - vtx[0].x) * (vtx[i + 1].y - vtx[0].y)) - ((vtx[i + 1].x - vtx[0].x) *  (vtx[i].y - vtx[0].y))) / 2);
	}


	//make abs
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
    //point size must 4 (x0,y0) (x1,y1) (x2,y2) (x3,y3)
	if (pts1.size() != 4 || pts2.size() != 4) return -1;

	//print(pts1);
	//printf("\n");
	//print(pts2);
	//print("\n");
	Mat H = findHomography(pts2, pts1, RANSAC);
	
	print(H);
	printf("\n");

	/*
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			printf("%lf\n", H.at<double>(i, j));
		}
	}
	*/

	//D = h1h5 - h2h4
	double D = (H.at<double>(0, 0)*H.at<double>(1, 1)) - (H.at<double>(0, 1)*H.at <double>(1, 0));

	//case : normal
	if (D > 0){
		printf("%lf*%lf - %lf*%lf = %lf\n", H.at<double>(0, 0), H.at<double>(1, 1), H.at<double>(0, 1), H.at <double>(1, 0),
			(H.at<double>(0, 0)*H.at<double>(1, 1)) - (H.at<double>(0, 1)*H.at <double>(1, 0)));
		return NORMAL;
	}

	else {
		//check reflection R = root(h7*h7+h8*h8)
		double R;
		R = sqrt(pow(H.at<double>(2, 0),2)+pow(H.at<double>(2,1),2));

		//check concave c1 
		//AB X AD
		double C1 = ((pts2[1].x - pts2[0].x)*(pts2[1].y - pts2[0].y)) - ((pts2[3].x - pts2[0].x)*(pts2[3].y - pts2[0].y));
		//BA X BC
		double C2 = ((pts2[0].x - pts2[1].x*(pts2[0].y - pts2[1].y)) - (pts2[2].x - pts2[1].x)*(pts2[2].y - pts2[1].y));
		//CB X CD
		double C3 = ((pts2[1].x - pts2[2].x*(pts2[1].y - pts2[2].y)) - (pts2[3].x - pts2[2].x)*(pts2[3].y - pts2[2].y));
		//DC X DA
		double C4 = ((pts2[2].x - pts2[3].x*(pts2[2].y - pts2[3].y)) - (pts2[0].x - pts2[3].x)*(pts2[0].y - pts2[3].y));

		if (C1 < 0 | C2 < 0 | C3 < 0 | C4 < 0) {
			printf("D : %lf, C1 : %lf, C2 : %lf, C3 : %lf, C4 : %lf\n", D, C1, C2, C3, C4);
			return CONCAVE;

			if (R == 0) {
				printf("D : %lf, C1 : %lf, C2 : %lf, C3 : %lf, C4 : %lf\n", D, C1, C2, C3, C4);
				return CONCAVE_REFLECTION;
			}
		}

		if (R == 0) {
			printf("D : %lf, R : %lf\n", D, R);
			return REFLECTION;
		}
		else {
			printf("%D : %lf, R : %lf\n", D, R);
			return TWIST;
		}





		return CONCAVE;
	}

}

// estimate a circle that best approximates the input points and return center and radius of the estimate circle
bool PolygonDemo::fitCircle(const std::vector<cv::Point>& pts, cv::Point2d& center, double& radius)
{
    int n = (int)pts.size();
    if (n < 3) return false;

    return false;
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
        line(frame, m_data_pts[i], m_data_pts[i + 1], Scalar(255, 255, 255), 1);
    }
    if (closed)
    {
        line(frame, m_data_pts[i], m_data_pts[0], Scalar(255, 255, 255), 1);
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
