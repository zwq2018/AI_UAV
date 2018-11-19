/* PSINS c++ hearder file PSINS.h */
/*
	By     : Yan Gongmin @ NWPU
	Date   : 2015-02-17, 2017-04-29, 2017-06-04, 2017-07-19
	From   : College of Automation, 
	         Northwestern Polytechnical University, 
			 Xi'an 710072, China
*/

#ifndef _PSINS_H
#define _PSINS_H

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdarg.h>
#include <time.h>

/* type re-define */
#ifndef BOOL
typedef int		BOOL;
#endif

/*constant define*/
#ifndef TRUE
#define TRUE	1
#define FALSE	0
#endif

#ifndef NULL
#define NULL	((void *)0)
#endif

#ifndef PI
#define PI		3.14159265358979
#endif
#define PI_2	(PI/2.0)
#define PI_4	(PI/4.0)
#define _2PI	(2.0*PI)

#ifndef EPS
#define EPS		2.220446049e-16F
#endif
#ifndef INF
#define INF		3.402823466e+30F
#endif

//#define ASSERT_NULL
#ifdef ASSERT_NULL
	#define assert(b)
#else
	BOOL	assert(BOOL b);
#endif

//#define MAT_COUNT_STATISTIC
#define IO_FILE_SURPORT

int		sign(double val, double eps=EPS);
double	range(double val, double minVal, double maxVal);
double	atan2Ex(double y, double x);
double  diffYaw(double yaw, double yaw0);
#define asinEx(x)		asin(range(x, -1.0, 1.0))
#define acosEx(x)		acos(range(x, -1.0, 1.0))
#define max(x,y)        ( (x)>=(y)?(x):(y) )
#define min(x,y)        ( (x)<=(y)?(x):(y) )
#define CC180toC360(yaw)  ( (yaw)>0.0 ? (_2PI-(yaw)) : -(yaw) )   // counter-clockwise +-180deg -> clockwise 0~360deg for yaw
#define C360toCC180(yaw)  ( (yaw)>=PI ? (_2PI-(yaw)) : -(yaw) )   // clockwise 0~360deg -> counter-clockwise +-180deg for yaw

// class define
class CGLV;
class CVect3;	class CMat3;	class CQuat;
class CEarth;	class CIMU;		class CSINS;	class CAligni0;
class CVect;	class CMat;		class CKalman;	class CSINSKF;	class CSINSTDKF;

// Max Matrix Dimension define
#define MMD		15
#define MMD2	(MMD*MMD)


// global variables and functions, can not be changed in any way
extern const CVect3	I31, O31;
extern const CQuat	qI;
extern const CMat3	I33, O33;
extern const CVect  On1;
extern const CGLV	glv;


class CGLV
{
public:
	double Re, f, g0, wie;											// the Earth's parameters
	double e, e2;
	double mg, ug, deg, min, sec, hur, ppm, ppmpsh;					// commonly used units
	double dps, dph, dpsh, dphpsh, ugpsh, ugpsHz, mpsh, mpspsh, secpsh;

	CGLV(double Re=6378137.0, double f=(1.0/298.257), double wie0=7.2921151467e-5, double g0=9.7803267714);
};

class CVect3 
{
public:
	double i, j, k;

	CVect3(void);
	CVect3(double xx, double yy=0.0, double zz=0.0);
	CVect3(const double *pdata);

	CVect3 operator+(const CVect3 &v) const;				// vector addition
	CVect3 operator-(const CVect3 &v) const;				// vector subtraction
	CVect3 operator*(const CVect3 &v) const;				// vector cross multiplication
	CVect3 operator*(double f) const;						// vector multiply scale
	CVect3 operator/(double f) const;						// vector divide scale
	CVect3& operator+=(const CVect3 &v);					// vector addition
	CVect3& operator-=(const CVect3 &v);					// vector subtraction
	CVect3& operator*=(double f);							// vector multiply scale
	CVect3& operator/=(double f);							// vector divide scale
	BOOL IsZero(double eps=EPS) const;						// assert if all elements are zeros
	BOOL IsZeroXY(double eps=EPS) const;					// assert if x&&y-elements are zeros
	BOOL IsNaN() const;										// assert if any element is NaN
	friend CVect3 operator*(double f, const CVect3 &v);		// scale multiply vector
	friend CVect3 operator-(const CVect3 &v);				// minus
	friend double norm(const CVect3 &v);					// vector norm
	friend double normXY(const CVect3 &v);					// vector norm or X & Y components
	friend CVect3 sqrt(const CVect3 &v);					// sqrt
	friend double dot(const CVect3 &v1, const CVect3 &v2);	// vector dot multiplication
	friend CMat3 a2mat(const CVect3 &att);					// Euler angles to DCM 
	friend CQuat a2qua(double pitch, double roll, double yaw);	// Euler angles to quaternion
	friend CQuat a2qua(const CVect3 &att);					// Euler angles to quaternion
	friend CQuat rv2q(const CVect3 &rv);					// rotation vector to quaternion
	friend CMat3 askew(const CVect3 &v);					// askew matrix;
	friend CMat3 pos2Cen(const CVect3 &pos);				// to geographical position matrix
	friend CVect3 pp2vn(const CVect3 &pos1, const CVect3 &pos0, double ts=1.0, CEarth *pEth=NULL);  // position difference to velocity
};

class CQuat
{
public:
	double q0, q1, q2, q3;

	CQuat(void);
	CQuat(double qq0, double qq1=0.0, double qq2=0.0, double qq3=0.0);
	CQuat(const double *pdata);

	CQuat operator+(const CVect3 &phi) const;	// true quaternion add misalign angles
	CQuat operator-(const CVect3 &phi) const;	// calculated quaternion delete misalign angles
	CVect3 operator-(CQuat &quat) const;		// get misalign angles from calculated quaternion & true quaternion
	CQuat operator*(const CQuat &q) const;		// quaternion multiplication
	CVect3 operator*(const CVect3 &v) const;	// quaternion multiply vector
	CQuat& operator*=(const CQuat &q);			// quaternion multiplication
	CQuat& operator-=(const CVect3 &phi);		// calculated quaternion delete misalign angles
	void normlize(CQuat *q);					// quaternion norm
	friend CQuat operator~(const CQuat &q);		// quaternion conjugate
	friend CVect3 q2att(const CQuat &qnb);		// quaternion to Euler angles 
	friend CMat3  q2mat(const CQuat &qnb);		// quaternion to DCM
	friend CVect3 q2rv(const CQuat &q);			// quaternion to rotation vector
};

class CMat3 
{
public:
	double e00, e01, e02, e10, e11, e12, e20, e21, e22;

	CMat3(void);
	CMat3(double xx, double xy, double xz,
		  double yx, double yy, double yz,
		  double zx, double zy, double zz );
	CMat3(const CVect3 &v0, const CVect3 &v1, const CVect3 &v2);  // M = [v0; v1; v2]

	CMat3 operator+(const CMat3 &m) const;					// matirx addition
	CMat3 operator-(const CMat3 &m) const;					// matirx subtraction
	CMat3 operator*(const CMat3 &m) const;					// matirx multiplication
	CMat3 operator*(double f) const;						// matirx multiply scale
	CVect3 operator*(const CVect3 &v) const;				// matirx multiply vector
	friend CMat3 operator-(const CMat3 &m);					// minus
	friend CMat3 operator~(const CMat3 &m);					// matirx transposition
	friend CMat3 operator*(double f, const CMat3 &m);		// scale multiply matirx
	friend double det(const CMat3 &m);						// matirx determination
	friend CMat3 inv(const CMat3 &m);						// matirx inverse
	friend CVect3 diag(const CMat3 &m);						// diagonal of a matrix
	friend CMat3 diag(const CVect3 &v);						// diagonal matrix
	friend CMat3 dv2att(CVect3 &va1, const CVect3 &va2, CVect3 &vb1, const CVect3 &vb2);  // attitude determination using double-vector
	friend CVect3 m2att(const CMat3 &Cnb);					// DCM to Euler angles 
	friend CQuat  m2qua(const CMat3 &Cnb);					// DCM to quaternion
};

class CVect
{
public:
	int row, clm;
	double dd[MMD];

	CVect(void);
	CVect(int row0, int clm0=1);
	CVect(int row0, double f);
	CVect(int row0, const double *pf);
	CVect(int row0, double f, double f1, ...);
	CVect(const CVect3 &v);
	CVect(const CVect3 &v1, const CVect3 v2);

	void Set(double f, ...);
	void Set2(double f, ...);
	CVect operator+(const CVect &v) const;		// vector addition
	CVect operator-(const CVect &v) const;		// vector subtraction
	CVect operator*(double f) const;			// vector multiply scale
	CVect& operator+=(const CVect &v);			// vector addition
	CVect& operator-=(const CVect &v);			// vector subtraction
	CVect& operator*=(double f);				// vector multiply scale
	CVect operator*(const CMat &m) const;		// row-vector multiply matrix
	CMat operator*(const CVect &v) const;		// 1xn vector multiply nx1 vector, or nx1 vector multiply 1xn vector
	double& operator()(int r);					// vector element
	friend CVect operator~(const CVect &v);		// vector transposition
	friend double norm(const CVect &v);			// vector norm
};

class CMat
{
public:
	int row, clm, rc;
	double dd[MMD2];

	CMat(void);
	CMat(int row0, int clm0);
	CMat(int row0, int clm0, double f);
	CMat(int row0, int clm0, const double *pf);

	void SetDiag(double f, ...);
	void SetDiag2(double f, ...);
	CMat operator+(const CMat &m) const;				// matirx addition
	CMat operator-(const CMat &m) const;				// matirx subtraction
	CMat operator*(double f) const;						// matirx multiply scale
	CVect operator*(const CVect &v) const;				// matirx multiply vector
	CMat operator*(const CMat &m) const;				// matirx multiplication
	CMat& operator+=(const CMat &m0);					// matirx addition
	CMat& operator+=(const CVect &v);					// matirx + diag(vector)
	CMat& operator-=(const CMat &m0);					// matirx subtraction
	CMat& operator*=(double f);							// matirx multiply scale
	CMat& operator++();									// 1.0 + diagonal
	double& operator()(int r, int c);					// get element m(r,c)
	void SetRow(int i, const CVect &v);					// set i-row from vector
	void SetClm(int j, const CVect &v);					// set j-column from vector
	CVect GetRow(int i) const;							// get i-row from matrix
	CVect GetClm(int j) const;							// get j-column from matrix
	void SetClmVect3(int i, int j, const CVect3 &v);	// set i...(i+2)-row&j-column from CVect3
	void SetRowVect3(int i, int j, const CVect3 &v);	// set i-row&j...(j+2)-column from CVect3
	void SetMat3(int i, int j, const CMat3 &m);			// set i...(i+2)-row&j...(j+2)-comumn from CMat3
	void ZeroRow(int i);								// set i-row to 0
	void ZeroClm(int j);								// set j-column to 0
	friend CMat array2mat(const double *f, int r, int c);	// convert array to mat
	friend CMat operator~(const CMat &m);				// matirx transposition
	friend void symmetry(CMat &m);						// matirx symmetrization
	friend double norm1(CMat &m);						// 1-norm
	friend CVect diag(const CMat &m);					// diagonal of a matrix
	friend CMat diag(const CVect &v);					// diagonal matrix
	friend void RowMul(CMat &m, const CMat &m0, const CMat &m1, int r); // m(r,:)=m0(r,:)*m1
	friend void RowMulT(CMat &m, const CMat &m0, const CMat &m1, int r); // m(r,:)=m0(r,:)*m1'
#ifdef MAT_COUNT_STATISTIC
	static int iCount, iMax;
	~CMat(void);
#endif
};

class CRAvar
{
public:
	#define RAMAX 10
	int nR0, maxCount, Rmaxflag[RAMAX];
	double ts, R0[RAMAX], Rmax[RAMAX], Rmin[RAMAX], tau[RAMAX], r0[RAMAX];

	CRAvar(void);
	CRAvar(int nR0, int maxCount0=2);
	void set(double r0, double tau, double rmax=0.0, double rmin=0.0, int i=0);
	void set(const CVect3 &r0, const CVect3 &tau, const CVect3 &rmax=O31, const CVect3 &rmin=O31);
	void set(const CVect &r0, const CVect &tau, const CVect &rmax=On1, const CVect &rmin=On1);
	void Update(double r, double ts, int i=0);
	void Update(const CVect3 &r, double ts);
	void Update(const CVect &r, double ts);
	double operator()(int k);			// get element sqrt(R0(k))
};

class CEarth
{
public:
	double a, b;
	double f, e, e2;
	double wie;

	double sl, sl2, sl4, cl, tl, RMh, RNh, clRNh, f_RMh, f_RNh, f_clRNh;
	CVect3 pos, vn, wnie, wnen, wnin, gn, gcc;

	CEarth(double a0=glv.Re, double f0=glv.f, double g0=glv.g0);
	void Update(const CVect3 &pos, const CVect3 &vn=CVect3(0.0));
	CVect3 vn2dpos(const CVect3 &vn, double ts=1.0) const;
};

class CIMU
{
public:
	int nSamples, prefirst;
	CVect3 phim, dvbm, wm_1, vm_1;

	CIMU(void);
	void Update(const CVect3 *wm, const CVect3 *vm, int nSamples);
};

class CSINS
{
public:
	double nts, tk;
	CEarth eth;
	CIMU imu;
	CQuat qnb;
	CMat3 Cnb, Cnb0, Cbn, Kg, Ka;
	CVect3 wib, fb, fn, an, web, wnb, att, vn, vb, pos, eb, db, _tauGyro, _tauAcc;
	CMat3 Maa, Mav, Map, Mva, Mvv, Mvp, Mpv, Mpp;	// for etm
	CVect3 vnL, posL; CMat3 CW, MpvCnb;		// for lever

	CSINS(const CQuat &qnb0=qI, const CVect3 &vn0=O31, const CVect3 &pos0=O31, double tk0=0.0);    // initialization using quat attitude, velocity & position
	void SetTauGA(CVect3 &tauG, CVect3 &tauA);
	void Update(const CVect3 *wm, const CVect3 *vm, int nSamples, double tk0);		// SINS update using Gyro&Acc samples
	void lever(const CVect3 &dL=O31);		// lever arm
	void etm(void);							// SINS error transform matrix coefficients
};

class CMahony
{
public:
	double tk, Kp, Ki;
	CQuat qnb;
	CMat3 Cnb;
	CVect3 exyzInt;

	CMahony(double tau=4.0, const CQuat &qnb0=qI);
	void SetTau(double tau=4.0);
	void Update(const CVect3 &gyro, const CVect3 &acc, const CVect3 &mag, double ts);
};

class CIIR
{
public:
	int n;
	double b[10], a[10], x[10], y[10];

	CIIR(void);
	CIIR(double *b0, double *a0, int n0);
	double Update(double x0);
};

class CAligni0
{
public:
	double tk;
	int t0, t1, t2;
	CVect3 wmm, vmm, vib0, vi0, Pib01, Pib02, Pi01, Pi02, tmpPib0, tmpPi0;
	CQuat qib0b;
	CEarth eth;
	CIMU imu;

	CAligni0(const CVect3 &pos=O31);
	CQuat Update(const CVect3 *wm, const CVect3 *vm, int nSamples, double ts);
};

class CKalman
{
public:
	double kftk;
	int nq, nr, measflag;
	CMat Ft, Pk, Hk;
	CVect Xk, Zk, Qt, Rt, rts, Pmax, Pmin,
		Rmax, Rmin, Rbeta, Rb,				// measurement noise R adaptive
		FBTau, FBMax, FBXk, FBTotal;		// feedback control

	CKalman(int nq0, int nr0);
	virtual void Init(void) = 0;				// initialize Qk,Rk,P0...
	virtual void SetFt(void) = 0;				// process matrix setting
	virtual void SetHk(void) = 0;				// measurement matrix setting
	virtual void SetMeas(void) = 0;				// set measurement
	virtual void Feedback(double fbts);			// feed back
	void TimeUpdate(double kfts, int fback=1);	// time update
	void MeasUpdate(double fading=1.0);			// measurement update
	void RAdaptive(int i, double r, double Pr); // Rt adaptive
	void SetMeasFlag(int flag);					// measurement flag setting
	void PkConstrain(void);						// Pk constrain: Pmin<diag(Pk)<Pmax
};

class CSINSKF:public CKalman
{
public:
	CSINS sins;

	CSINSKF(int nq0, int nr0);
	virtual void Init(void) {} ;
	virtual void Init(CSINS &sins0);
	virtual void SetFt(void);
	virtual void SetHk(void);
	virtual void Feedback(double fbts);
	void Update(const CVect3 *wm, const CVect3 *vm, int nSamples, double tk);	// KF Time&Meas Update 
};

class CSINSTDKF:public CSINSKF
{
public:
	double tdts;
	int iter, ifn, measRes;
	CMat Fk, Pk1; 
	CVect Pxz, Qk, Kk, Hi, tmeas;
	CVect3 meanfn;

	CSINSTDKF(int nq0, int nr0);
	void TDUpdate(CVect3 *wm, CVect3 *vm, int nSamples, double tk, int nStep=1);  // Time-Distributed Update
};

class CQEAHRS:public CKalman
{
public:
	CMat3 Cnb;

	CQEAHRS(double ts);
	void Update(const CVect3 &gyro, const CVect3 &acc, const CVect3 &mag, double ts);
};

#ifdef WIN32

class CFileRdWt
{
	FILE *f;
	static char dirIn[256], dirOut[256];
public:
	double buff[128];
	int columns;

	CFileRdWt(const char *dirI, const char *dirO=NULL);
	CFileRdWt(const char *fname0, int columns);
	int load(int lines=1);
	int IsEOF(void);
	CFileRdWt& operator<<(double d);
	CFileRdWt& operator<<(const CVect3 &v);
	CFileRdWt& operator<<(const CVect &v);
	CFileRdWt& operator<<(const CMat &m);
	CFileRdWt& operator<<(const CRAvar &R);
	CFileRdWt& operator<<(const CSINS &sins);
	CFileRdWt& operator<<(const CMahony &ahrs);
	CFileRdWt& operator<<(const CQEAHRS &ahrs);
	CFileRdWt& operator<<(const CKalman &kf);
	CFileRdWt& operator>>(double &d);
	CFileRdWt& operator>>(CVect3 &v);
	CFileRdWt& operator>>(CVect &v);
	CFileRdWt& operator>>(CMat &m);
	~CFileRdWt();
};

#endif //WIN32

CVect3 AlignCoarse(CVect3 wmm, CVect3 vmm, double latitude);

#endif
