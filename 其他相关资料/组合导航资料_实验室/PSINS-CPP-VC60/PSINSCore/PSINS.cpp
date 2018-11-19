#include "PSINS.h"

const CVect3 I31(1,1,1), O31(0,0,0);
const CQuat  qI(1.0,0,0,0);
const CMat3  I33(1,0,0, 0,1,0, 0,0,1), O33(0,0,0, 0,0,0, 0,0,0);
const CVect  On1(MMD,0.0);
const CGLV   glv;

/***************************  class CGLV  *********************************/
CGLV::CGLV(double Re, double f, double wie0, double g0)
{
	this->Re = Re; this->f = f; this->wie = wie0; this->g0 = g0;
	e = sqrt(2*f-f*f); e2 = e*e;
    mg = 1.0e-3*g0;
    ug = 1.0e-6*glv.g0;
    deg = PI/180.0;
    min = deg/60.0;
    sec = min/60.0;
    ppm = 1.0e-6;
    hur = 3600.0;
	dps = deg/1.0;
    dph = deg/hur;
    dpsh = deg/sqrt(hur);
    dphpsh = dph/sqrt(hur);
    ugpsHz = ug/sqrt(1.0);
    ugpsh = ug/sqrt(hur);
    mpsh = 1/sqrt(hur); 
    mpspsh = 1/1/sqrt(hur);
    ppmpsh = ppm/sqrt(hur);
    secpsh = sec/sqrt(hur);
}

/***************************  class CVect3  *********************************/
CVect3::CVect3(void)
{
}

CVect3::CVect3(double xx, double yy, double zz)
{
	i=xx, j=yy, k=zz;
}

CVect3::CVect3(const double *pdata)
{
	i=*pdata++, j=*pdata++, k=*pdata++;
}

BOOL CVect3::IsZero(double eps) const
{
	return (i<eps&&i>-eps && j<eps&&j>-eps && k<eps&&k>-eps);
}

BOOL CVect3::IsZeroXY(double eps) const
{
	return (i<eps&&i>-eps && j<eps&&j>-eps);
}

BOOL CVect3::IsNaN(void) const
{
	return 0; //(_isnan(i) || _isnan(j) || _isnan(k));
}

CVect3 CVect3::operator+(const CVect3 &v) const
{
	return CVect3(this->i+v.i, this->j+v.j, this->k+v.k);
}

CVect3 CVect3::operator-(const CVect3 &v) const
{
	return CVect3(this->i-v.i, this->j-v.j, this->k-v.k);
}

CVect3 CVect3::operator*(const CVect3 &v) const
{
	return CVect3(this->j*v.k-this->k*v.j, this->k*v.i-this->i*v.k, this->i*v.j-this->j*v.i);
}
	
CVect3 CVect3::operator*(double f) const
{
	return CVect3(i*f, j*f, k*f);
}
	
CVect3 CVect3::operator/(double f) const
{
	return CVect3(i/f, j/f, k/f);
}

CVect3& CVect3::operator+=(const CVect3 &v)
{ 
	i += v.i, j += v.j, k += v.k;
	return *this;
}

CVect3& CVect3::operator-=(const CVect3 &v)
{ 
	i -= v.i, j -= v.j, k -= v.k;
	return *this;
}

CVect3& CVect3::operator*=(double f)
{ 
	i *= f, j *= f, k *= f;
	return *this;
}

CVect3& CVect3::operator/=(double f)
{ 
	i /= f, j /= f, k /= f;
	return *this;
}

CVect3 operator*(double f, const CVect3 &v)
{
	return CVect3(v.i*f, v.j*f, v.k*f);
}
	
CVect3 operator-(const CVect3 &v)
{
	return CVect3(-v.i, -v.j, -v.k);
}

CVect3 sqrt(const CVect3 &v)
{
	return CVect3(sqrt(v.i), sqrt(v.j), sqrt(v.k));
}

double norm(const CVect3 &v)
{
	return sqrt(v.i*v.i + v.j*v.j + v.k*v.k);
}

double normXY(const CVect3 &v)
{
	return sqrt(v.i*v.i + v.j*v.j);
}

double dot(const CVect3 &v1, const CVect3 &v2)
{
	return (v1.i*v2.i + v1.j*v2.j + v1.k*v2.k);
}

CQuat rv2q(const CVect3 &rv)
{
#define F1	(   2 * 1)		// define: Fk=2^k*k! 
#define F2	(F1*2 * 2)
#define F3	(F2*2 * 3)
#define F4	(F3*2 * 4)
#define F5	(F3*2 * 5)
	double n2 = rv.i*rv.i+rv.j*rv.j+rv.k*rv.k, c, f;
	if(n2<(PI/180.0*PI/180.0))	// 0.017^2 
	{
		double n4=n2*n2;
		c = 1.0 - n2*(1.0/F2) + n4*(1.0/F4);
		f = 0.5 - n2*(1.0/F3) + n4*(1.0/F5);
	}
	else
	{
		double n_2 = sqrt(n2)/2.0;
		c = cos(n_2);
		f = sin(n_2)/n_2*0.5;
	}
	return CQuat(c, f*rv.i, f*rv.j, f*rv.k);
}

CMat3 askew(const CVect3 &v)
{
	return CMat3(0,  -v.k, v.j, 
				 v.k, 0.0,  -v.i,
				-v.j, v.i, 0);
}

CMat3 pos2Cen(const CVect3 &pos)
{
	double si = sin(pos.i), ci = cos(pos.i), sj = sin(pos.j), cj = cos(pos.j);
	return CMat3(	-sj, -si*cj,  ci*cj,  
					 cj, -si*sj,  ci*sj,  
					 0,   ci,     si      );	//Cen
}

CVect3 pp2vn(CVect3 &pos1, CVect3 &pos0, double ts, CEarth *pEth)
{
	double sl, cl, sl2, sq, sq2, RMh, RNh, clRNh;
	if(pEth)
	{
		RMh = pEth->RMh; clRNh = pEth->clRNh;
	}
	else
	{
		sl=sin(pos0.i); cl=cos(pos0.i); sl2=sl*sl;
		sq = 1-glv.e2*sl2; sq2 = sqrt(sq);
		RMh = glv.Re*(1-glv.e2)/sq/sq2+pos0.k;
		RNh = glv.Re/sq2+pos0.k;    clRNh = cl*RNh;
	}
    CVect3 vn = pos1 - pos0;
    return CVect3(vn.j*clRNh/ts, vn.i*RMh/ts, vn.k/ts);
}

/***************************  class CQuat  *********************************/
CQuat::CQuat(void)
{
}

CQuat::CQuat(double qq0, double qq1, double qq2, double qq3)
{
	q0=qq0, q1=qq1, q2=qq2, q3=qq3;
}

CQuat::CQuat(const double *pdata)
{
	q0=*pdata++, q1=*pdata++, q2=*pdata++, q3=*pdata++;
}

CQuat CQuat::operator+(const CVect3 &phi) const
{
	CQuat qtmp = rv2q(-phi);
	return qtmp*(*this);
}

CQuat CQuat::operator-(const CVect3 &phi) const
{
	CQuat qtmp = rv2q(phi);
	return qtmp*(*this);
}

CVect3 CQuat::operator-(CQuat &quat) const
{
	CQuat dq;
	
	dq = quat*(~(*this));
	if(dq.q0<0)
	{
		dq.q0=-dq.q0, dq.q1=-dq.q1, dq.q2=-dq.q2, dq.q3=-dq.q3;
	}
	double n2 = acos(dq.q0), f;
	if( sign(n2)!=0 )
	{
		f = 2.0/(sin(n2)/n2);
	}
	else
	{
		f = 2.0;
	}
	return CVect3(dq.q1,dq.q2,dq.q3)*f;
}

CQuat CQuat::operator*(const CQuat &quat) const
{
	CQuat qtmp;
	qtmp.q0 = q0*quat.q0 - q1*quat.q1 - q2*quat.q2 - q3*quat.q3;
	qtmp.q1 = q0*quat.q1 + q1*quat.q0 + q2*quat.q3 - q3*quat.q2;
	qtmp.q2 = q0*quat.q2 + q2*quat.q0 + q3*quat.q1 - q1*quat.q3;
	qtmp.q3 = q0*quat.q3 + q3*quat.q0 + q1*quat.q2 - q2*quat.q1;
	return qtmp;
}

CQuat& CQuat::operator*=(const CQuat &quat)
{
	return (*this=*this*quat);
}

CQuat& CQuat::operator-=(const CVect3 &phi)
{
	CQuat qtmp = rv2q(phi);
	return (*this=qtmp*(*this));
}

CQuat operator~(const CQuat &q)
{
	return CQuat(q.q0,-q.q1,-q.q2,-q.q3);
}

CVect3 CQuat::operator*(const CVect3 &v) const
{
	CQuat qtmp;
	CVect3 vtmp;
	qtmp.q0 =         - q1*v.i - q2*v.j - q3*v.k;
	qtmp.q1 = q0*v.i           + q2*v.k - q3*v.j;
	qtmp.q2 = q0*v.j           + q3*v.i - q1*v.k;
	qtmp.q3 = q0*v.k           + q1*v.j - q2*v.i;
	vtmp.i = -qtmp.q0*q1 + qtmp.q1*q0 - qtmp.q2*q3 + qtmp.q3*q2;
	vtmp.j = -qtmp.q0*q2 + qtmp.q2*q0 - qtmp.q3*q1 + qtmp.q1*q3;
	vtmp.k = -qtmp.q0*q3 + qtmp.q3*q0 - qtmp.q1*q2 + qtmp.q2*q1;
	return vtmp;
}

void normlize(CQuat *q)
{
	double nq=sqrt(q->q0*q->q0+q->q1*q->q1+q->q2*q->q2+q->q3*q->q3);
	q->q0 /= nq, q->q1 /= nq, q->q2 /= nq, q->q3 /= nq;
}

CVect3 q2rv(const CQuat &q)
{
	CQuat dq;
	dq = q;
	if(dq.q0<0)  { dq.q0=-dq.q0, dq.q1=-dq.q1, dq.q2=-dq.q2, dq.q3=-dq.q3; }
	if(dq.q0>1.0) dq.q0=1.0;
	double n2 = acos(dq.q0), f;
	if(n2>1.0e-20)
	{
		f = 2.0/(sin(n2)/n2);
	}
	else
	{
		f = 2.0;
	}
	return CVect3(dq.q1,dq.q2,dq.q3)*f;
}

/***************************  class CMat3  *********************************/
CMat3::CMat3(void)
{
}

CMat3::CMat3(double xx, double xy, double xz, 
		  double yx, double yy, double yz,
		  double zx, double zy, double zz )
{
	e00=xx,e01=xy,e02=xz; e10=yx,e11=yy,e12=yz; e20=zx,e21=zy,e22=zz;
}

CMat3::CMat3(const CVect3 &v0, const CVect3 &v1, const CVect3 &v2)
{
	e00 = v0.i, e01 = v0.j, e02 = v0.k;
	e10 = v1.i, e11 = v1.j, e12 = v1.k;
	e20 = v2.i, e21 = v2.j, e22 = v2.k;
}

CMat3 dv2att(CVect3 &va1, const CVect3 &va2, CVect3 &vb1, const CVect3 &vb2)
{
	CVect3 a=va1*va2, b=vb1*vb2, aa=a*va1, bb=b*vb1;
	CMat3 Ma(va1/norm(va1),a/norm(a),aa/norm(aa)), Mb(vb1/norm(vb1),b/norm(b),bb/norm(bb));
	return (~Ma)*(Mb);  //Cab
}

CMat3 operator-(const CMat3 &m)
{
	return CMat3(-m.e00,-m.e01,-m.e02,-m.e10,-m.e11,-m.e12,-m.e20,-m.e21,-m.e22);
}

CMat3 operator~(const CMat3 &m)
{
	return CMat3(m.e00,m.e10,m.e20, m.e01,m.e11,m.e21, m.e02,m.e12,m.e22);
}

CMat3 CMat3::operator*(const CMat3 &mat) const
{
	CMat3 mtmp;
	mtmp.e00 = e00*mat.e00 + e01*mat.e10 + e02*mat.e20;
	mtmp.e01 = e00*mat.e01 + e01*mat.e11 + e02*mat.e21;
	mtmp.e02 = e00*mat.e02 + e01*mat.e12 + e02*mat.e22;
	mtmp.e10 = e10*mat.e00 + e11*mat.e10 + e12*mat.e20;
	mtmp.e11 = e10*mat.e01 + e11*mat.e11 + e12*mat.e21;
	mtmp.e12 = e10*mat.e02 + e11*mat.e12 + e12*mat.e22;
	mtmp.e20 = e20*mat.e00 + e21*mat.e10 + e22*mat.e20;
	mtmp.e21 = e20*mat.e01 + e21*mat.e11 + e22*mat.e21;
	mtmp.e22 = e20*mat.e02 + e21*mat.e12 + e22*mat.e22;
	return mtmp;
}

CMat3 CMat3::operator+(const CMat3 &mat) const
{
	CMat3 mtmp;
	mtmp.e00 = e00 + mat.e00;  mtmp.e01 = e01 + mat.e01;  mtmp.e02 = e02 + mat.e02;  
	mtmp.e10 = e10 + mat.e10;  mtmp.e11 = e11 + mat.e11;  mtmp.e12 = e12 + mat.e12;  
	mtmp.e20 = e20 + mat.e20;  mtmp.e21 = e21 + mat.e21;  mtmp.e22 = e22 + mat.e22;  
	return mtmp;
}

CMat3 CMat3::operator-(const CMat3 &mat) const
{
	CMat3 mtmp;
	mtmp.e00 = e00 - mat.e00;  mtmp.e01 = e01 - mat.e01;  mtmp.e02 = e02 - mat.e02;  
	mtmp.e10 = e10 - mat.e10;  mtmp.e11 = e11 - mat.e11;  mtmp.e12 = e12 - mat.e12;  
	mtmp.e20 = e20 - mat.e20;  mtmp.e21 = e21 - mat.e21;  mtmp.e22 = e22 - mat.e22;  
	return mtmp;
}

CMat3 CMat3::operator*(double f) const
{
	return CMat3(e00*f,e01*f,e02*f, e10*f,e11*f,e12*f, e21*f,e20*f,e22*f);
}

CMat3 operator*(double f, const CMat3 &m)
{
	return CMat3(m.e00*f,m.e01*f,m.e02*f, m.e10*f,m.e11*f,m.e12*f, m.e20*f,m.e21*f,m.e22*f);
}

CVect3 CMat3::operator*(const CVect3 &v) const
{
	return CVect3(e00*v.i+e01*v.j+e02*v.k,e10*v.i+e11*v.j+e12*v.k,e20*v.i+e21*v.j+e22*v.k);
}

double det(const CMat3 &m)
{
	return m.e00*(m.e11*m.e22-m.e12*m.e21) - m.e01*(m.e10*m.e22-m.e12*m.e20) + m.e02*(m.e10*m.e21-m.e11*m.e20);
}

CQuat a2qua(double pitch, double roll, double yaw)
{
	pitch /= 2.0, roll /= 2.0, yaw /= 2.0;
    double	sp = sin(pitch), sr = sin(roll), sy = sin(yaw), 
			cp = cos(pitch), cr = cos(roll), cy = cos(yaw);
	CQuat qnb;
    qnb.q0 = cp*cr*cy - sp*sr*sy;
    qnb.q1 = sp*cr*cy - cp*sr*sy;
    qnb.q2 = cp*sr*cy + sp*cr*sy;
    qnb.q3 = cp*cr*sy + sp*sr*cy;
	return qnb;
}

CMat3 a2mat(const CVect3 &att)
{
	double	si = sin(att.i), ci = cos(att.i),
			sj = sin(att.j), cj = cos(att.j),
			sk = sin(att.k), ck = cos(att.k);
	CMat3 Cnb;
	Cnb.e00 =  cj*ck - si*sj*sk;	Cnb.e01 =  -ci*sk;	Cnb.e02 = sj*ck + si*cj*sk;
	Cnb.e10 =  cj*sk + si*sj*ck;	Cnb.e11 =  ci*ck;	Cnb.e12 = sj*sk - si*cj*ck;
	Cnb.e20 = -ci*sj;				Cnb.e21 =  si;		Cnb.e22 = ci*cj;
	return Cnb;
}

CQuat a2qua(const CVect3 &att)
{
	return a2qua(att.i, att.j, att.k);
}

CVect3 m2att(const CMat3 &Cnb)
{
	CVect3 att;
	att.i = asinEx(Cnb.e21);
	att.j = atan2Ex(-Cnb.e20, Cnb.e22);
	att.k = atan2Ex(-Cnb.e01, Cnb.e11);
	return att;
}

CQuat m2qua(const CMat3 &Cnb)
{
	double q0, q1, q2, q3, qq4;
    if(Cnb.e00>=Cnb.e11+Cnb.e22)
	{
        q1 = 0.5*sqrt(1+Cnb.e00-Cnb.e11-Cnb.e22);  qq4 = 4*q1;
        q0 = (Cnb.e21-Cnb.e12)/qq4; q2 = (Cnb.e01+Cnb.e10)/qq4; q3 = (Cnb.e02+Cnb.e20)/qq4;
	}
    else if(Cnb.e11>=Cnb.e00+Cnb.e22)
	{
        q2 = 0.5*sqrt(1-Cnb.e00+Cnb.e11-Cnb.e22);  qq4 = 4*q2;
        q0 = (Cnb.e02-Cnb.e20)/qq4; q1 = (Cnb.e01-Cnb.e10)/qq4; q3 = (Cnb.e12+Cnb.e21)/qq4;
	}
    else if(Cnb.e22>=Cnb.e00+Cnb.e11)
	{
        q3 = 0.5*sqrt(1-Cnb.e00-Cnb.e11+Cnb.e22);  qq4 = 4*q3;
        q0 = (Cnb.e10-Cnb.e01)/qq4; q1 = (Cnb.e02+Cnb.e20)/qq4; q2 = (Cnb.e12+Cnb.e21)/qq4;
	}
    else
	{
        q0 = 0.5*sqrt(1+Cnb.e00+Cnb.e11+Cnb.e22);  qq4 = 4*q0;
        q1 = (Cnb.e21-Cnb.e12)/qq4; q2 = (Cnb.e02-Cnb.e20)/qq4; q3 = (Cnb.e10-Cnb.e01)/qq4;
	}
	double nq = sqrt(q0*q0+q1*q1+q2*q2+q3*q3);
	q0 /= nq; q1 /= nq; q2 /= nq; q3 /= nq;
	return CQuat(q0, q1, q2, q3);
}

CVect3 q2att(const CQuat &qnb)
{
	double	q11 = qnb.q0*qnb.q0, q12 = qnb.q0*qnb.q1, q13 = qnb.q0*qnb.q2, q14 = qnb.q0*qnb.q3, 
			q22 = qnb.q1*qnb.q1, q23 = qnb.q1*qnb.q2, q24 = qnb.q1*qnb.q3,     
			q33 = qnb.q2*qnb.q2, q34 = qnb.q2*qnb.q3,  
			q44 = qnb.q3*qnb.q3;
	CVect3 att;
	att.i = asinEx(2*(q34+q12));
	att.j = atan2Ex(-2*(q24-q13), q11-q22-q33+q44);
	att.k = atan2Ex(-2*(q23-q14), q11-q22+q33-q44);
	return att;
}

CMat3 q2mat(const CQuat &qnb)
{
	double	q11 = qnb.q0*qnb.q0, q12 = qnb.q0*qnb.q1, q13 = qnb.q0*qnb.q2, q14 = qnb.q0*qnb.q3, 
			q22 = qnb.q1*qnb.q1, q23 = qnb.q1*qnb.q2, q24 = qnb.q1*qnb.q3,     
			q33 = qnb.q2*qnb.q2, q34 = qnb.q2*qnb.q3,  
			q44 = qnb.q3*qnb.q3;
	CMat3 Cnb;
    Cnb.e00 = q11+q22-q33-q44,  Cnb.e01 = 2*(q23-q14),     Cnb.e02 = 2*(q24+q13),
	Cnb.e10 = 2*(q23+q14),      Cnb.e11 = q11-q22+q33-q44, Cnb.e12 = 2*(q34-q12),
	Cnb.e20 = 2*(q24-q13),      Cnb.e21 = 2*(q34+q12),     Cnb.e22 = q11-q22-q33+q44;
	return Cnb;
}

CMat3 inv(const CMat3 &m)
{
	double nm;
	nm = m.e00*(m.e11*m.e22-m.e12*m.e21) - m.e01*(m.e10*m.e22-m.e12*m.e20) + m.e02*(m.e10*m.e21-m.e11*m.e20);
	CMat3 mtmp;
	mtmp.e00 =  (m.e11*m.e22-m.e12*m.e21)/nm;
	mtmp.e10 = -(m.e10*m.e22-m.e12*m.e20)/nm;
	mtmp.e20 =  (m.e10*m.e21-m.e11*m.e20)/nm;
	mtmp.e01 = -(m.e01*m.e22-m.e02*m.e21)/nm;
	mtmp.e11 =  (m.e00*m.e22-m.e02*m.e20)/nm;
	mtmp.e21 = -(m.e00*m.e21-m.e01*m.e20)/nm;
	mtmp.e02 =  (m.e01*m.e12-m.e02*m.e11)/nm;
	mtmp.e12 = -(m.e00*m.e12-m.e02*m.e10)/nm;
	mtmp.e22 =  (m.e00*m.e11-m.e01*m.e10)/nm;
	return mtmp;
}

CVect3 diag(const CMat3 &m)
{
	return CVect3(m.e00, m.e11, m.e22);
}

CMat3 diag(const CVect3 &v)
{
	return CMat3(v.i,0,0, 0,v.j,0, 0,0,v.k);
}

/***************************  class CMat  *********************************/
CMat::CMat(void)
{
#ifdef MAT_COUNT_STATISTIC
	if(iMax<++iCount) iMax = iCount;
#endif
}
	
CMat::CMat(int row0, int clm0)
{
#ifdef MAT_COUNT_STATISTIC
	if(iMax<++iCount) iMax = iCount;
#endif
	row=row0; clm=clm0; rc=row*clm;
}

CMat::CMat(int row0, int clm0, double f)
{
#ifdef MAT_COUNT_STATISTIC
	if(iMax<++iCount) iMax = iCount;
#endif
	row=row0; clm=clm0; rc=row*clm;
	for(double *pd=dd, *pEnd=&dd[rc]; pd<pEnd; pd++)  *pd = f;
}

CMat::CMat(int row0, int clm0, const double *pf)
{
#ifdef MAT_COUNT_STATISTIC
	if(iMax<++iCount) iMax = iCount;
#endif
	row=row0; clm=clm0; rc=row*clm;
	memcpy(dd, pf, rc*sizeof(double));
}

#ifdef MAT_COUNT_STATISTIC
int CMat::iCount=0, CMat::iMax=0;
CMat::~CMat(void)
{
	iCount--;
}
#endif

CMat CMat::operator*(const CMat &m0) const
{
#ifdef MAT_COUNT_STATISTIC
	++iCount;
#endif
	assert(this->clm==m0.row);
	CMat mtmp(this->row,m0.clm);
	int m=this->row, k=this->clm, n=m0.clm;
	double *p=mtmp.dd; const double *p1i=this->dd, *p2=m0.dd;
	for(int i=0; i<m; i++,p1i+=k)
	{
		for(int j=0; j<n; j++)
		{
			double f=0.0; const double *p1is=p1i, *p1isEnd=&p1i[k], *p2sj=&p2[j];
			for(; p1is<p1isEnd; p1is++,p2sj+=n)	f += (*p1is) * (*p2sj);
			*p++ = f;
		}
	}
	return mtmp;
}

CVect CMat::operator*(const CVect &v) const
{
	assert(this->clm==v.row);
	CVect vtmp(this->row);
	double *p=vtmp.dd, *pEnd=&vtmp.dd[vtmp.row]; const double *p1ij=this->dd, *p2End=&v.dd[v.row];
	for(; p<pEnd; p++)
	{
		double f=0.0; const double *p2j=v.dd;
		for(; p2j<p2End; p1ij++,p2j++)	f += (*p1ij) * (*p2j);
		*p = f;
	}
	return vtmp;
}

CMat CMat::operator+(const CMat &m0) const
{
#ifdef MAT_COUNT_STATISTIC
	++iCount;
#endif
	assert(row==m0.row&&clm==m0.clm);
	CMat mtmp(row,clm);
	double *p=mtmp.dd, *pEnd=&mtmp.dd[rc]; const double *p1=this->dd, *p2=m0.dd;
	while(p<pEnd)
	{ *p++ = (*p1++) + (*p2++); } 
	return mtmp;
}

CMat& CMat::operator+=(const CVect &v)
{
	assert(row==v.row||clm==v.clm);
	int row1 = row+1;
	double *p=dd, *pEnd=&dd[rc];
	for(const double *p1=v.dd; p<pEnd; p+=row1, p1++)	*p += *p1;
	return *this;
}

CMat CMat::operator-(const CMat &m0) const
{
#ifdef MAT_COUNT_STATISTIC
	++iCount;
#endif
	assert(row==m0.row&&clm==m0.clm);
	CMat mtmp(row,clm);
	double *p=mtmp.dd, *pEnd=&mtmp.dd[rc]; const double *p1=this->dd, *p2=m0.dd;
	while(p<pEnd)
	{ *p++ = (*p1++) - (*p2++); } 
	return mtmp;
}

CMat CMat::operator*(double f) const
{
#ifdef MAT_COUNT_STATISTIC
	++iCount;
#endif
	CMat mtmp(row,clm);
	double *p=mtmp.dd, *pEnd=&mtmp.dd[rc]; const double *p1=this->dd;
	while(p<pEnd)
	{ *p++ = (*p1++) * f; } 
	return mtmp;
}

CMat& CMat::operator+=(const CMat &m0)
{
	assert(row==m0.row&&clm==m0.clm);
	double *p=dd, *pEnd=&dd[rc]; const double *p1=m0.dd;
	while(p<pEnd)
	{ *p++ += *p1++; } 
	return *this;
}

CMat& CMat::operator-=(const CMat &m0)
{
	assert(row==m0.row&&clm==m0.clm);
	double *p=dd, *pEnd=&dd[rc]; const double *p1=m0.dd;
	while(p<pEnd)
	{ *p++ -= *p1++; } 
	return *this;
}

CMat& CMat::operator*=(double f)
{
	double *p=dd, *pEnd=&dd[rc];
	while(p<pEnd)
	{ *p++ *= f; } 
	return *this;
}

CMat& CMat::operator++()
{
	int row1=row+1;
	for(double *p=dd, *pEnd=&dd[rc]; p<pEnd; p+=row1)	*p += 1.0;
	return *this;
}

CMat operator~(const CMat &m0)
{
#ifdef MAT_COUNT_STATISTIC
	++CMat::iCount;
#endif
	CMat mtmp(m0.clm,m0.row);
	const double *pm=m0.dd;
	for(int i=0; i<m0.row; i++)
	{ for(int j=i; j<m0.rc; j+=m0.row) mtmp.dd[j] = *pm++; }
	return mtmp;
}

void symmetry(CMat &m)
{
	assert(m.row==m.clm);
	for(int i=0; i<m.clm; i++)
	{
		double *prow=&m.dd[i*m.clm+i+1], *prowEnd=&m.dd[i*m.clm+m.clm], *pclm=&m.dd[i*m.clm+i+m.clm];
		for(; prow<prowEnd; prow++,pclm+=m.clm)  *prow=*pclm=(*prow+*pclm)*0.5;
	}
}

double& CMat::operator()(int r, int c)
{
	return this->dd[r*this->clm+c];
}

void CMat::SetRow(int i, const CVect &v)
{
	assert(clm==v.clm);
	const double *p=v.dd;
	for(double *p1=&dd[i*clm],*pEnd=p1+clm; p1<pEnd; p++,p1++) *p1 = *p;
	return;
}

void CMat::SetClm(int j, const CVect &v)
{
	assert(row==v.row);
	const double *p=v.dd;
	for(double *p1=&dd[j],*pEnd=&dd[rc]; p1<pEnd; p++,p1+=clm) *p1 = *p;
	return;
}

void CMat::SetClmVect3(int i, int j, const CVect3 &v)
{
	double *p=&dd[i*clm+j];
	*p = v.i; p += clm;
	*p = v.j; p += clm;
	*p = v.k;
}

void CMat::SetRowVect3(int i, int j, const CVect3 &v)
{
	*(CVect3*)&dd[i*clm+j] = v;
}

void CMat::SetMat3(int i, int j, const CMat3 &m)
{
	double *p=&dd[i*clm+j];
	*(CVect3*)p = *(CVect3*)&m.e00;  p += clm;
	*(CVect3*)p = *(CVect3*)&m.e10;  p += clm;
	*(CVect3*)p = *(CVect3*)&m.e20;
}

CVect CMat::GetRow(int i) const
{
	CVect v;
	v.row=1; v.clm=clm;
	const double *p1=&dd[i*clm], *pEnd=p1+clm;
	for(double *p=v.dd; p1<pEnd; p++,p1++) *p = *p1;
	return v;
}

CVect CMat::GetClm(int j) const
{
	CVect v;
	v.row=row; v.clm=1;
	const double *p1=&dd[j], *pEnd=&dd[rc];
	for(double *p=v.dd; p1<pEnd; p++,p1+=clm) *p = *p1;
	return v;
}

void CMat::ZeroRow(int i)
{
	for(double *p=&dd[i*clm],*pEnd=p+clm; p<pEnd; p++) *p = 0.0;
	return;
}

void CMat::ZeroClm(int j)
{
	for(double *p=&dd[j],*pEnd=&dd[rc]; p<pEnd; p+=clm) *p = 0.0;
	return;
}

void CMat::SetDiag(double f, ...)
{
	va_list vl;
	va_start(vl, f);
	double *p=dd, *pEnd=&dd[rc];
	for(int row1=row+1; p<pEnd; p+=row1)
	{ *p = f;  f = va_arg(vl, double);	}
	va_end(vl);
}

void CMat::SetDiag2(double f, ...)
{
	va_list vl;
	va_start(vl, f);
	double *p=dd, *pEnd=&dd[rc];
	for(int row1=row+1; p<pEnd; p+=row1)
	{ *p = f*f;  f = va_arg(vl, double);	}
	va_end(vl);
}

double norm1(CMat &m)
{
	double n1=0.0;
	for(double *p=m.dd,*pEnd=&m.dd[m.rc]; p<pEnd; p++)
	{
		if(*p>0.0)	 n1 += *p;
		else  n1 -= *p;
	}
	return n1;
}

CVect diag(const CMat &m)
{
	int row1 = m.row+1;
	CVect vtmp(m.row,1);
	double *p=vtmp.dd, *pEnd=&vtmp.dd[vtmp.row];
	for(const double *p1=m.dd; p<pEnd; p++, p1+=row1)	*p = *p1;
	return vtmp;
}

void RowMul(CMat &m, const CMat &m0, const CMat &m1, int r)
{
	assert(m0.clm==m1.row);
	int rc0=r*m0.clm;
	double *p=&m.dd[rc0], *pEnd=p+m0.clm; const double *p0=&m0.dd[rc0], *p0End=p0+m0.clm, *p1j=m1.dd;
	for(; p<pEnd; p++)
	{
		double f=0.0; const double *p0j=p0, *p1jk=p1j++;
		for(; p0j<p0End; p0j++,p1jk+=m1.clm)	 f += (*p0j) * (*p1jk);
		*p = f;
	}
}

void RowMulT(CMat &m, const CMat &m0, const CMat &m1, int r)
{
	assert(m0.clm==m1.clm);
	int rc0=r*m0.clm;
	double *p=&m.dd[rc0], *pEnd=p+m0.clm; const double *p0=&m0.dd[rc0], *p0End=p0+m0.clm, *p1jk=m1.dd;
	for(; p<pEnd; p++)
	{
		double f=0.0; const double *p0j=p0;
		for(; p0j<p0End; p0j++,p1jk++)	 f += (*p0j) * (*p1jk);
		*p = f;
	}
}

CMat diag(const CVect &v)
{
#ifdef MAT_COUNT_STATISTIC
	++CMat::iCount;
#endif
	int rc = v.row>v.clm ? v.row : v.clm, rc1=rc+1;
	CMat mtmp(rc,rc,0.0);
	double *p=mtmp.dd;
	for(const double *p1=v.dd, *p1End=&v.dd[rc]; p1<p1End; p+=rc1, p1++)	*p = *p1;
	return mtmp;
}

/***************************  class CVect  *********************************/
CVect::CVect(void)
{
}

CVect::CVect(int row0, int clm0)
{
	if(clm0==1) { row=row0; clm=1;   }
	else		{ row=1;    clm=clm0;}
 }

CVect::CVect(int row0, double f)
{
	row=row0; clm=1;
	for(int i=0;i<row;i++) dd[i]=f;
}

CVect::CVect(int row0, const double *pf)
{
	row=row0; clm=1;
	memcpy(dd, pf, row*sizeof(double));
}

CVect::CVect(int row0, double f, double f1, ...)
{
	row=row0; clm=1;
	assert(row<=MMD&&clm<=MMD);
	va_list vl;
	va_start(vl, f);
	for(int i=0, rc=row>clm?row:clm; i<rc; i++)
	{ dd[i] = f;  f = va_arg(vl, double);	}
	va_end(vl);
}

CVect::CVect(const CVect3 &v)
{
	row=3; clm=1;
	dd[0]=v.i; dd[1]=v.j; dd[2]=v.k;
}

CVect::CVect(const CVect3 &v1, const CVect3 v2)
{
	row=6; clm=1;
	dd[0]=v1.i; dd[1]=v1.j; dd[2]=v1.k;
	dd[3]=v2.i; dd[4]=v2.j; dd[5]=v2.k;
}

CVect operator~(const CVect &v)
{
	CVect vtmp=v;
	vtmp.row=v.clm; vtmp.clm=v.row;
	return vtmp;
}

CVect CVect::operator*(const CMat &m) const
{
	assert(clm==m.row);
	CVect vtmp(row,clm);
	double *p=vtmp.dd; const double *p1End=&dd[clm];
	for(int j=0; j<clm; p++,j++)
	{
		double f=0.0; const double *p1j=dd, *p2jk=&m.dd[j];
		for(; p1j<p1End; p1j++,p2jk+=m.clm)	 f += (*p1j) * (*p2jk);
		*p = f;
	}
	return vtmp;
}

CMat CVect::operator*(const CVect &v) const
{
#ifdef MAT_STATISTIC
	++CMat::iCount;
#endif
	assert(clm==v.row);
	CMat mtmp(row,v.clm);
	if(row==1 && v.clm==1)  // (1x1) = (1xn)*(nx1)
	{
		double f = 0.0;
		for(int i=0; i<clm; i++)  f += dd[i]*v.dd[i];
		mtmp.dd[0] = f;
	}
	else    // (nxn) = (nx1)*(1xn)
	{
		double *p = mtmp.dd;
		for(int i=0; i<row; i++)
		{
			for(int j=0; j<v.clm; j++)  *p++ = dd[i]*v.dd[j];
		}
	}
	return mtmp;
}

CVect CVect::operator+(const CVect &v) const
{
	assert(row==v.row&&clm==v.clm);
	const double *p2=v.dd, *p1=dd, *p1End=&dd[row>clm?row:clm];
	CVect vtmp(row,clm);
	for(double *p=vtmp.dd; p1<p1End; p++,p1++,p2++)  { *p=*p1+*p2; }
	return vtmp;
}

CVect CVect::operator-(const CVect &v) const
{
	assert(row==v.row&&clm==v.clm);
	const double *p2=v.dd, *p1=dd, *p1End=&dd[row>clm?row:clm];
	CVect vtmp(row,clm);
	for(double *p=vtmp.dd; p1<p1End; p++,p1++,p2++)  { *p=*p1-*p2; }
	return vtmp;
}
	
CVect CVect::operator*(double f) const
{
	CVect vtmp(row,clm);
	const double *p1=dd,*p1End=&dd[row>clm?row:clm];
	for(double *p=vtmp.dd; p1<p1End; p++,p1++)  { *p=*p1*f; }
	return vtmp;
}

CVect& CVect::operator+=(const CVect &v)
{
	assert(row==v.row&&clm==v.clm);
	const double *p1 = v.dd;
	for(double *p=dd, *pEnd=&dd[row>clm?row:clm]; p<pEnd; p++,p1++)  { *p += *p1; }
	return *this;
}

CVect& CVect::operator-=(const CVect &v)
{
	assert(row==v.row&&clm==v.clm);
	const double *p1 = v.dd;
	for(double *p=dd, *pEnd=&dd[row>clm?row:clm]; p<pEnd; p++,p1++)  { *p -= *p1; }
	return *this;
}

CVect& CVect::operator*=(double f)
{
	for(double *p=dd, *pEnd=&dd[row>clm?row:clm]; p<pEnd; p++)  { *p *= f; }
	return *this;
}

double norm(const CVect &v)
{
	const double *p=v.dd, *pEnd=&v.dd[v.row>v.clm?v.row:v.clm];
	double f=0.0;
	for(; p<pEnd; p++)  { f += (*p)*(*p); }
	return sqrt(f);
}

double& CVect::operator()(int r)
{
	return this->dd[r];
}

void CVect::Set(double f, ...)
{
	assert(row<=MMD&&clm<=MMD);
	va_list vl;
	va_start(vl, f);
	for(int i=0, rc=row>clm?row:clm; i<rc; i++)
	{ dd[i] = f;  f = va_arg(vl, double);	}
	va_end(vl);
}

void CVect::Set2(double f, ...)
{
	assert(row<=MMD&&clm<=MMD);
	va_list vl;
	va_start(vl, f);
	for(int i=0, rc=row>clm?row:clm; i<rc; i++)
	{ dd[i] = f*f;  f = va_arg(vl, double);	}
	va_end(vl);
}

/***************************  class CIIR  *********************************/
CIIR::CIIR(void)
{
}

CIIR::CIIR(double *b0, double *a0, int n0)
{
	assert(n0<10);
	for(int i=0; i<n0; i++)  { b[i]=b0[i]/a0[0]; a[i]=a0[i]; x[i]=y[i]=0.0; }
	n = n0;
}

double CIIR::Update(double x0)
{
//	a(1)*y(n) = b(1)*x(n) + b(2)*x(n-1) + ... + b(nb+1)*x(n-nb)
//                        - a(2)*y(n-1) - ... - a(na+1)*y(n-na)
	double y0 = 0.0;
	for(int i=n-1; i>0; i--)
	{
		x[i] = x[i-1]; y[i] = y[i-1];
		y0 += b[i]*x[i] - a[i]*y[i];
	}
	x[0] = x0;
	y0 += b[0]*x0;
	y[0] = y0;
	return y0;
}

/***************************  class CRAvar  *********************************/
CRAvar::CRAvar()
{
}

CRAvar::CRAvar(int nR0, int maxCount0)
{
	assert(nR0<RAMAX);
	this->nR0 = nR0;
	maxCount = maxCount0;
}

void CRAvar::set(double r0, double tau, double rmax, double rmin, int i)
{
	this->R0[i] = r0*r0;
	this->tau[i] = tau;
	this->r0[i] = 0.0;  Rmaxflag[i] = maxCount;
	this->Rmax[i] = rmax==0.0 ? 100.0*this->R0[i] : rmax*rmax;
	this->Rmin[i] = rmin==0.0 ?  0.01*this->R0[i] : rmin*rmin;
}

void CRAvar::set(const CVect3 &r0, const CVect3 &tau, const CVect3 &rmax, const CVect3 &rmin)
{
	const double *pr0=&r0.i, *ptau=&tau.i, *prmax=&rmax.i, *prmin=&rmin.i;
	for(int i=0; i<3; i++,pr0++,ptau++,prmax++,prmin++)
		set(*pr0, *ptau, *prmax, *prmin, i);
}

void CRAvar::set(const CVect &r0, const CVect &tau, const CVect &rmax, const CVect &rmin)
{
	const double *pr0=r0.dd, *ptau=tau.dd, *prmax=rmax.dd, *prmin=rmin.dd;
	for(int i=0; i<nR0; i++,pr0++,ptau++,prmax++,prmin++)
		set(*pr0, *ptau, *prmax, *prmin, i);
}

void CRAvar::Update(double r, double ts, int i)
{
	double tstau = ts>tau[i] ? 1.0 : ts/tau[i];
	double dr2=r-r0[i]; dr2=dr2*dr2; r0[i]=r;
	if(dr2>R0[i]) R0[i]=dr2; else R0[i]=(1.0-tstau)*R0[i]+tstau*dr2;
	if(R0[i]<Rmin[i]) R0[i]=Rmin[i];
	if(R0[i]>Rmax[i]) {R0[i]=Rmax[i];Rmaxflag[i]=maxCount;} else {Rmaxflag[i]-=Rmaxflag[i]>0;}
}

void CRAvar::Update(const CVect3 &r, double ts)
{
	const double *pr=&r.i;
	for(int i=0; i<3; i++,pr++)
		Update(*pr, ts, i);
}

void CRAvar::Update(const CVect &r, double ts)
{
	const double *pr=r.dd;
	for(int i=0; i<nR0; i++,pr++)
		Update(*pr, ts, i);
}

double CRAvar::operator()(int k)
{
	return Rmaxflag[k] ? INF : sqrt(R0[k]);
}

/***************************  class CKalman  *********************************/
CKalman::CKalman(int nq0, int nr0)
{
	assert(nq0<=MMD&&nr0<=MMD);
	kftk = 0.0;
	nq = nq0; nr = nr0;
	Ft = Pk = CMat(nq,nq,0.0);
	Hk = CMat(nr,nq,0.0);
	Qt = Pmin = Xk = CVect(nq,0.0);  Pmax = CVect(nq,INF);
	Rt = Zk = CVect(nr,0.0);  rts = CVect(nr,1.0);
	Rmax = CVect(nr,INF); Rmin = Rb = CVect(nr,0.0); Rbeta = CVect(nr,1.0);
	FBTau = FBMax = CVect(nq,INF); FBXk = FBTotal = CVect(nq,0.0);
	measflag = 0;
}

void CKalman::TimeUpdate(double kfts, int fback)
{
	CMat Fk;
	kftk += kfts;
	SetFt();
	Fk=++(Ft*kfts);  // Fk = I+Ft*ts
	Xk = Fk * Xk;
	Pk = Fk*Pk*(~Fk);  Pk += Qt*kfts;
	if(fback)  Feedback(kfts);
}

void CKalman::SetMeasFlag(int flag)
{
	measflag = (flag==0) ? 0 : (measflag|flag);
}

void CKalman::MeasUpdate(double fading)
{
	CVect Pxz, Kk, Hi;
	SetMeas();
	for(int i=0; i<nr; i++)
	{
		if(measflag&(0x01<<i))
		{
			Hi = Hk.GetRow(i);
			Pxz = Pk*(~Hi);
			double Pz0 = (Hi*Pxz)(0,0), r=Zk(i)-(Hi*Xk)(0,0);
			RAdaptive(i, r, Pz0);
			double Pzz = Pz0+Rt.dd[i]/rts.dd[i];
			Kk = Pxz*(1.0/Pzz);
			Xk += Kk*r;
			Pk -= Kk*(~Pxz);
		}
	}
	if(fading>1.0) Pk *= fading;
	PkConstrain();
	symmetry(Pk);
	SetMeasFlag(0);
}

void CKalman::RAdaptive(int i, double r, double Pr)
{
	if(Rb.dd[i]>EPS)
	{
		double rr=r*r-Pr;
		if(rr<Rmin.dd[i])	rr = Rmin.dd[i];
		if(rr>Rmax.dd[i])	Rt.dd[i] = Rmax.dd[i];
		else				Rt.dd[i] = (1.0-Rbeta.dd[i])*Rt.dd[i]+Rbeta.dd[i]*rr;
		Rbeta.dd[i] = Rbeta.dd[i]/(Rbeta.dd[i]+Rb.dd[i]);
	}
}

void CKalman::PkConstrain(void)
{
	int i=0, nq1=nq+1;
	for(double *p=Pk.dd,*pmin=Pmin.dd,*pminEnd=&Pmin.dd[nq],*pmax=Pmax.dd; pmin<pminEnd; p+=nq1,pmin++,pmax++)
	{
		if(*p<*pmin && *p>EPS)
		{
			*p = *pmin;
		}
		else if(*p>*pmax)
		{
			double sqf=sqrt(*pmax/(*p))*0.5;
			for(double *prow=&Pk.dd[i*Pk.clm],*prowEnd=prow+nq,*pclm=&Pk.dd[i]; prow<prowEnd; prow++,pclm+=nq)
			{
				*prow *= sqf;
				*pclm *= sqf;
			}
		}
		i++;
	}
}

void CKalman::Feedback(double fbts)
{
	double *pTau=FBTau.dd, *pTotal=FBTotal.dd, *pMax=FBMax.dd, *pXk=FBXk.dd, *p=Xk.dd;
	for(int i=0; i<nq; i++, pTau++,pTotal++,pMax++,pXk++,p++)
	{
		if(*pTau<INF/2)
		{
			double afa = fbts<*pTau ? fbts/(*pTau) : 1.0;
			*pXk = *p*afa;
			if(*pTotal+*pXk>*pMax)			*pXk = *pMax-*pTotal;
			else if(*pTotal+*pXk<-*pMax)  	*pXk = -*pMax-*pTotal;
			*p -= *pXk;
			*pTotal += *pXk;
		}
		else
		{
			*pXk = 0.0;
		}
	}
}

/***************************  class CSINSKF  *********************************/
CSINSKF::CSINSKF(int nq0, int nr0):CKalman(nq0,nr0)
{
	sins = CSINS(qI, O31, O31);
}

void CSINSKF::Init(CSINS &sins0)
{
	sins = sins0;
	// a example for 15-state(phi,dvn,dpos,eb,db) inertial grade setting
	Pmax.Set2(10.0*glv.deg,10.0*glv.deg,30.0*glv.deg,    50.0,50.0,50.0,    1.0e4/glv.Re,1.0e4/glv.Re,1.0e4, 
		10.0*glv.dph,10.0*glv.dph,10.0*glv.dph,    10.0*glv.mg,10.0*glv.mg,10.0*glv.mg);
	Pmin.Set2(0.01*glv.min,0.01*glv.min,0.1*glv.min,    0.01,0.01,0.1,    1.0/glv.Re,1.0/glv.Re,0.1, 
		0.001*glv.dph,0.001*glv.dph,0.001*glv.dph,    10.0*glv.ug,10.0*glv.ug,10.0*glv.ug);
	Pk.SetDiag2(1.0*glv.deg,1.0*glv.deg,10.0*glv.deg,    1.0,1.0,1.0,     100.0/glv.Re,100.0/glv.Re,100.0, 
		1.0*glv.dph,1.0*glv.dph,1.0*glv.dph,    1.0*glv.mg,1.0*glv.mg,1.0*glv.mg);
	Qt.Set2(0.001*glv.dpsh,0.001*glv.dpsh,0.001*glv.dpsh,    10.0*glv.ugpsHz,10.0*glv.ugpsHz,10.0*glv.ugpsHz,    0.0,0.0,0.0,
		0.0*glv.dphpsh,0.0*glv.dphpsh,0.0*glv.dphpsh,    0.0*glv.ugpsh,0.0*glv.ugpsh,0.0*glv.ugpsh);
	Rt.Set2(0.2,0.2,0.6,   10.0/glv.Re,10.0/glv.Re,30.0);
	FBTau.Set(1.0,1.0,10.0,     1.0,1.0,1.0,     1.0,1.0,1.0,    10.0,10.0,10.0,    10.0,10.0,10.0);
	SetHk(); 
}

void CSINSKF::SetFt(void)
{
	sins.etm();
//	Ft = [ Maa    Mav    Map    -Cnb	O33 
//         Mva    Mvv    Mvp     O33    Cnb 
//         O33    Mpv    Mpp     O33    O33
//         zeros(6,9)  diag(-1./[ins.tauG;ins.tauA]) ];
	Ft.SetMat3(0,0,sins.Maa), Ft.SetMat3(0,3,sins.Mav), Ft.SetMat3(0,6,sins.Map), Ft.SetMat3(0,9,-sins.Cnb); 
	Ft.SetMat3(3,0,sins.Mva), Ft.SetMat3(3,3,sins.Mvv), Ft.SetMat3(3,6,sins.Mvp), Ft.SetMat3(3,12,sins.Cnb); 
							  Ft.SetMat3(6,3,sins.Mpv), Ft.SetMat3(6,6,sins.Mpp);
	Ft(9,9) = sins._tauGyro.i,  Ft(10,10) = sins._tauGyro.j, Ft(11,11) = sins._tauGyro.k;
	Ft(12,12) = sins._tauAcc.i, Ft(13,13) = sins._tauAcc.j,  Ft(14,14) = sins._tauAcc.k;
}

void CSINSKF::SetHk(void)
{
	// a example for SINS/GPS vn&pos measurement
	Hk(0,3) = Hk(1,4) = Hk(2,5) = 1.0;
	Hk(3,6) = Hk(4,7) = Hk(5,8) = 1.0;
}

void CSINSKF::Update(const CVect3 *wm, const CVect3 *vm, int nSamples, double ts)
{
	sins.Update(wm, vm, nSamples, ts);
	TimeUpdate(sins.nts);  kftk = sins.tk;
	MeasUpdate();
}

void CSINSKF::Feedback(double fbts)
{
	CKalman::Feedback(fbts);
	sins.qnb -= *(CVect3*)&FBXk.dd[0];  sins.vn -= *(CVect3*)&FBXk.dd[3];  sins.pos -= *(CVect3*)&FBXk.dd[6];
	sins.eb += *(CVect3*)&FBXk.dd[9];	sins.db += *(CVect3*)&FBXk.dd[12]; 
}

/***************************  class CTDSINSKF  *********************************/
CSINSTDKF::CSINSTDKF(int nq0, int nr0):CSINSKF(nq0,nr0)
{
	iter = -2;  ifn = 0;  tdts = 0.0;
	Fk = Pk1 = CMat(nq,nq, 0.0);
	Pxz = Qk = Kk = tmeas = CVect(nr, 0.0);
	meanfn = O31;
}

void CSINSTDKF::TDUpdate(CVect3 *wm, CVect3 *vm, int nSamples, double ts, int nStep)
{
	sins.Update(wm, vm, nSamples, ts);
	Feedback(sins.nts);

	measRes = 0;

	if(nStep<=0) { nStep=2*(nq+nr)+3; }

	tdts += sins.nts; kftk = sins.tk;
	meanfn = meanfn+sins.fn; ifn++;
	for(int i=0; i<nStep; i++)
	{
		if(iter==-2)			// -2: set measurements
		{
			if(ifn==0)	break;
			CVect3 vtmp=meanfn*(1.0/ifn); meanfn = O31; ifn = 0;
			sins.fn=vtmp; SetFt(); sins.fn = vtmp;			
			SetMeas();
		}
		else if(iter==-1)			// -1: discrete
		{
			Fk = ++(Ft*tdts); // Fk = I+Ft*ts
			Qk = Qt*tdts;
			Xk = Fk*Xk;
			tdts = 0.0;
		}
		else if(iter<nq)		// 0 -> (nq-1): Fk*Pk
		{
			int row=iter;
			RowMul(Pk1, Fk, Pk, row);
		}
		else if(iter<2*nq)		// nq -> (2*nq-1): Fk*Pk*Fk+Qk
		{
			int row=iter-nq;
			RowMulT(Pk, Pk1, Fk, row);
			if(row==nq-1) {	Pk += Qk; }
		}
		else if(iter<2*nq+2*nr)	// (2*nq) -> (2*nq+2*nr-1): sequential measurement updating
		{
			int row=(iter-2*Ft.row)/2;
			int flag = measflag&(0x01<<row);
			if(flag)
			{
				if((iter-2*Ft.row)%2==0)
				{
					Hi = Hk.GetRow(row);
					Pxz = Pk*(~Hi);
					double Pzz = (Hi*Pxz)(0,0) + Rt(row)/rts(row);
					Kk = Pxz*(1.0/Pzz);
				}
				else
				{
					measRes |= flag;
					double r = Zk(row)-(Hi*Xk)(0,0);
					RAdaptive(row, r, (Hi*Pxz)(0,0));
					Xk += Kk*r;
					Pk -= Kk*(~Pxz);
				}
			}
			else
			{
				nStep++;
			}
		}
		else if(iter>=2*(nq+nr))	// 2*(nq+nr): Pk constrain & symmetry
		{
			PkConstrain();
			symmetry(Pk);
			SetMeasFlag(0);
			iter = -3;
		}
		iter++;
	}
}

/***************************  class CEarth  *********************************/
CEarth::CEarth(double a0, double f0, double g0)
{
	a = a0;	f = f0; wie = glv.wie; 
	b = (1-f)*a;
	e = sqrt(a*a-b*b)/a;	e2 = e*e;
	gn = CVect3(0, 0, -g0);
}

void CEarth::Update(const CVect3 &pos, const CVect3 &vn)
{
	this->pos = pos;  this->vn = vn;
	sl = sin(pos.i), cl = cos(pos.i), tl = sl/cl;
	double sq = 1-e2*sl*sl, sq2 = sqrt(sq);
	RMh = a*(1-e2)/sq/sq2+pos.k;	f_RMh = 1.0/RMh;
	RNh = a/sq2+pos.k;    clRNh = cl*RNh;  f_RNh = 1.0/RNh; f_clRNh = 1.0/clRNh;
	wnie.i = 0,				wnie.j = wie*cl,		wnie.k = wie*sl;
	wnen.i = -vn.j*f_RMh,	wnen.j = vn.i*f_RNh,	wnen.k = wnen.j*tl;
	wnin = wnie + wnen;
	sl2 = sl*sl, sl4 = sl2*sl2;
	gn.k = -( glv.g0*(1+5.27094e-3*sl2+2.32718e-5*sl4)-3.086e-6*pos.k );
	gcc = gn - (wnie+wnin)*vn;
}

CVect3 CEarth::vn2dpos(const CVect3 &vn, double ts) const
{
	return CVect3(vn.j*f_RMh, vn.i*f_clRNh, vn.k)*ts;
}

/***************************  class CIMU  *********************************/
CIMU::CIMU(void)
{
	prefirst = 1;
}

void CIMU::Update(const CVect3 *wm, const CVect3 *vm, int nSamples)
{
	static double conefactors[5][4] = {				// coning coefficients
		{2./3},										// 2
		{9./20, 27./20},							// 3
		{54./105, 92./105, 214./105},				// 4
		{250./504, 525./504, 650./504, 1375./504}	// 5
		};
	int i;
	double *pcf = conefactors[nSamples-2];
	CVect3 cm(0.0), sm(0.0), wmm(0.0), vmm(0.0);

	this->nSamples = nSamples;
	if(nSamples==1)  // one-plus-previous sample
	{
		if(prefirst==1) {wm_1=wm[0]; vm_1=vm[0]; prefirst=0;}
		cm = 1.0/12*wm_1; wm_1=wm[0]; 
		sm = 1.0/12*vm_1; vm_1=vm[0];
	}
	if(nSamples>1) prefirst=1;
	for(i=0; i<nSamples-1; i++)
	{
		cm += pcf[i]*wm[i];
		sm += pcf[i]*vm[i];
		wmm += wm[i];
		vmm += vm[i];
	}
	wmm += wm[i];
	vmm += vm[i];
	phim = wmm + cm*wm[i];
	dvbm = vmm + 1.0/2*wmm*vmm + (cm*vm[i]+sm*wm[i]);
}

/***************************  class CSINS  *********************************/
CSINS::CSINS(const CQuat &qnb0, const CVect3 &vn0, const CVect3 &pos0, double tk0)
{
	tk = tk0;  nts = 0.0;
	qnb = qnb0;	vn = vn0, pos = pos0;
	eth.Update(pos0, vn0);
	Cnb = q2mat(qnb); att = m2att(Cnb); Cnb0 = Cnb; Cbn = ~Cnb; vb = Cbn*vn;
	Kg = Ka = I33; eb = db = _tauGyro = _tauAcc = O31;
	wib = fb = fn = an = wnb = web = O31;
	etm(); lever();
}

void CSINS::SetTauGA(CVect3 &tauG, CVect3 &tauA)
{
	_tauGyro.i = tauG.i>INF/2 ? 0.0 : 1.0/tauG.i;   // Gyro&Acc inverse correlation time for AR(1) model
	_tauGyro.j = tauG.j>INF/2 ? 0.0 : 1.0/tauG.j;
	_tauGyro.k = tauG.k>INF/2 ? 0.0 : 1.0/tauG.k;
	_tauAcc.i  = tauA.i>INF/2 ? 0.0 : 1.0/tauA.i;
	_tauAcc.j  = tauA.j>INF/2 ? 0.0 : 1.0/tauA.j;
	_tauAcc.k  = tauA.k>INF/2 ? 0.0 : 1.0/tauA.k;
}

void CSINS::Update(const CVect3 *wm, const CVect3 *vm, int nSamples, double ts)
{
	nts = nSamples*ts;	tk += nts;
	double nts2 = nts/2;
	imu.Update(wm, vm, nSamples);
	imu.phim = Kg*imu.phim - eb*nts; imu.dvbm = Ka*imu.dvbm - db*nts;  // IMU calibration
	CVect3 vn01 = vn+an*nts2, pos01 = pos+eth.vn2dpos(vn01,nts2);
	eth.Update(pos01, vn01);
	wib = imu.phim/nts; fb = imu.dvbm/nts;
	web = wib - Cbn*eth.wnie;
	wnb = wib - (qnb*rv2q(imu.phim/2))*eth.wnin;
	fn = qnb*fb;
	an = rv2q(-eth.wnin*nts2)*fn+eth.gcc;
	CVect3 vn1 = vn + an*nts;
	pos = pos + eth.vn2dpos(vn+vn1, nts2);	vn = vn1;
	Cnb0 = Cnb;
	qnb = rv2q(-eth.wnin*nts)*qnb*rv2q(imu.phim);
	Cnb = q2mat(qnb); att = m2att(Cnb); Cbn = ~Cnb; vb = Cbn*vn;
}

void CSINS::lever(const CVect3 &dL)
{
	Mpv = CMat3(0,eth.f_RMh,0, eth.f_clRNh,0,0, 0,0,1);
	CW = Cnb*askew(web), MpvCnb = Mpv*Cnb;
	vnL = vn + CW*dL; posL = pos + MpvCnb*dL;
}

void CSINS::etm(void)
{
	double tl=eth.tl, secl=1.0/eth.cl, secl2=secl*secl, 
		wN=eth.wnie.j, wU=eth.wnie.k, vE=vn.i, vN=vn.j;
	double f_RMh=eth.f_RMh, f_RNh=eth.f_RNh, f_clRNh=eth.f_clRNh, 
		f_RMh2=f_RMh*f_RMh, f_RNh2=f_RNh*f_RNh;
	CMat3 Avn=askew(vn),
		Mp1(0,0,0, -wU,0,0, wN,0,0),
		Mp2(0,0,vN*f_RMh2, 0,0,-vE*f_RNh2, vE*secl2*f_RNh,0,-vE*tl*f_RNh2);
//	CVect3 _wnin = -eth.wnin; 	Maa = askew(_wnin);  // for Keil/VS2017 ???
	Maa = askew(-eth.wnin);
	Mav = CMat3(0,-f_RMh,0, f_RNh,0,0, tl*f_RNh,0,0);
	Map = Mp1+Mp2;
	Mva = askew(fn);
//	CVect3 wnien = eth.wnie+eth.wnin;	Mvv = Avn*Mav - askew(wnien);
	Mvv = Avn*Mav - askew(eth.wnie+eth.wnin);
	Mvp = Avn*(Mp1+Map);
	double scl = eth.sl*eth.cl;
    Mvp.e20 = Mvp.e20-glv.g0*(5.27094e-3*2*scl+2.32718e-5*4*eth.sl2*scl); Mvp.e22 = Mvp.e22+3.086e-6;
	Mpv = CMat3(0,f_RMh,0, f_clRNh,0,0, 0,0,1);
	Mpp = CMat3(0,0,-vN*f_RMh2, vE*tl*f_clRNh,0,-vE*secl*f_RNh2, 0,0,0);
}

/*********************  class CMahony AHRS  ************************/
CMahony::CMahony(double tau, const CQuat &qnb0)
{
	SetTau(tau);
	qnb = qnb0;
	Cnb = q2mat(qnb);
	exyzInt = O31;
	tk = 0.0;
}

void CMahony::SetTau(double tau)
{
	double beta = 2.146/tau;
	Kp = 2.0f*beta, Ki = beta*beta;
}

void CMahony::Update(const CVect3 &gyro, const CVect3 &acc, const CVect3 &mag, double ts)
{
	double nm;
	CVect3 acc0, mag0, exyz, bxyz, wxyz;

	nm = norm(acc);
	acc0 = nm>0.1 ? acc/nm : O31;
	nm = norm(mag);
	mag0 = nm>0.1 ? mag/nm : O31;
	bxyz = Cnb*mag0;
	bxyz.j = normXY(bxyz); bxyz.i = 0.0;
	wxyz = (~Cnb)*bxyz;
	exyz = *((CVect3*)&Cnb.e20)*acc0 + wxyz*mag0;
	exyzInt += exyz*(Ki*ts);
	qnb *= rv2q((gyro*glv.dps-Kp*exyz-exyzInt)*ts);
	Cnb = q2mat(qnb);
	tk += ts;
}

/*********************  class Quat&EKF based AHRS  ************************/
CQEAHRS::CQEAHRS(double ts):CKalman(7,3)
{
	double sts = sqrt(ts);
	Pmax.Set2(2.0,2.0,2.0,2.0, 1000*glv.dph,1000.0*glv.dph,1000.0*glv.dph);
	Pmin.Set2(0.001,0.001,0.001,0.001, 10.0*glv.dph,10.0*glv.dph,10.0*glv.dph);
	Pk.SetDiag2(1.0,1.0,1.0,1.0, 1000.0*glv.dph,1000.0*glv.dph,1000.0*glv.dph);
	Qt.Set2(10.0*glv.dpsh,10.0*glv.dpsh,10.0*glv.dpsh, 10.0*glv.dphpsh,10.0*glv.dphpsh,10.0*glv.dphpsh);
	Rt.Set2(100.0*glv.mg/sts,100.0*glv.mg/sts, 1.0*glv.deg/sts);
	Xk(0) = 1.0;
	Cnb = q2mat(*(CQuat*)&Xk.dd[0]);
}

void CQEAHRS::Update(const CVect3 &gyro, const CVect3 &acc, const CVect3 &mag, double ts)
{
	double q0, q1, q2, q3, wx, wy, wz, fx, fy, fz, mx, my, mz, h11, h12, h21, h22; 
	q0 = Xk.dd[0],		 q1 = Xk.dd[1],		  q2 = Xk.dd[2],		q3 = Xk.dd[3];
	wx = gyro.i*glv.dps, wy = gyro.j*glv.dps, wz = gyro.k*glv.dps; 
	fx = acc.i,			 fy = acc.j,		  fz = acc.k; 
	mx = mag.i,			 my = mag.j,		  mz = mag.k; 
	// Ft
	                0, Ft.dd[ 1] = -wx/2, Ft.dd[ 2] = -wy/2, Ft.dd[ 3] = -wz/2,  Ft.dd[ 4] =  q1/2, Ft.dd[ 5] =  q2/2, Ft.dd[ 6] =  q3/2; 
	Ft.dd[ 7] =  wx/2,                 0, Ft.dd[ 9] =  wz/2, Ft.dd[10] = -wy/2,  Ft.dd[11] = -q0/2, Ft.dd[12] =  q3/2, Ft.dd[13] = -q2/2; 
	Ft.dd[14] =  wy/2, Ft.dd[15] = -wz/2,                 0, Ft.dd[17] =  wx/2,  Ft.dd[18] = -q3/2, Ft.dd[18] = -q0/2, Ft.dd[20] =  q1/2; 
	Ft.dd[21] =  wz/2, Ft.dd[22] =  wy/2, Ft.dd[23] = -wx/2,                 0,  Ft.dd[25] =  q2/2, Ft.dd[26] = -q1/2, Ft.dd[27] = -q0/2; 
	// Hk
    h11 = fx*q0-fy*q3+fz*q2;  h12 = fx*q1+fy*q2+fz*q3;
    h21 = fx*q3+fy*q0-fz*q1;  h22 = fx*q2-fy*q1-fz*q0;
    Hk.dd[ 0] = h11*2,  Hk.dd[ 1] = h12*2,  Hk.dd[ 2] = -h22*2,  Hk.dd[ 3] = -h21*2;
    Hk.dd[ 7] = h21*2,  Hk.dd[ 8] = h22*2,  Hk.dd[ 9] =  h12*2,  Hk.dd[10] =  h11*2;
/*	CVect3 magH = Cnb*mag;
	double C11=Cnb.e11, C01=Cnb.e01, CC=C11*C11+C01*C01;
	if(normXY(magH)>0.01 && CC>0.25)  // CC>0.25 <=> pitch<60deg
	{
		double f2=2.0/CC;
        Hk.dd[14] = (q3*C11+q0*C01)*f2,  Hk.dd[15] = (-q2*C11-q1*C01)*f2,  Hk.dd[16] = (-q1*C11+q2*C01)*f2,  Hk.dd[17] = (q0*C11-q3*C01)*f2;
		Zk.dd[2] = atan2(magH.i, magH.j);
	}
	else
	{
        Hk.dd[14] = Hk.dd[15] = Hk.dd[16] = Hk.dd[17] = 0.0;
		Zk.dd[2] = 0.0;
	}*/

	SetMeasFlag(0x03);
	TimeUpdate(ts);
	MeasUpdate();
	PkConstrain();
	normlize((CQuat*)&Xk.dd[0]);
	Cnb = q2mat(*(CQuat*)&Xk.dd[0]);
}

/******************************  File Read or Write *********************************/
#ifdef WIN32

char CFileRdWt::dirIn[256] = {0}, CFileRdWt::dirOut[256] = {0};

CFileRdWt::CFileRdWt(const char *dirI, const char *dirO)  // set dir
{
	f = 0;
	int len = strlen(dirI);
	memcpy(dirIn, dirI, len);
	if(dirIn[len-1]!='\\') { dirIn[len]='\\'; dirIn[len+1]='\0'; }
	if(dirO)
	{
		len = strlen(dirO);
		memcpy(dirOut, dirO, len);
		if(dirOut[len-1]!='\\') { dirOut[len]='\\'; dirOut[len+1]='\0'; }
	}
	else
		memcpy(dirOut, dirIn, strlen(dirIn));
}

CFileRdWt::CFileRdWt(const char *fname0, int columns)
{
	char fname[256]={0};
	int findc=0, len0=strlen(fname0);
	for(int i=0; i<len0; i++)	{ if(fname0[i]=='\\') { findc=1; break; } }
	if(columns==0)
	{	if(dirOut[0]!=0&&findc==0)	{ strcat(fname, dirOut); } }
	else
	{	if(dirIn[0]!=0&&findc==0)	{ strcat(fname, dirIn); } }
	strcat(fname, fname0);
	if(columns==-1)		// bin file read
	{
		f = fopen(fname, "rb");
	}
	else if(columns==0)  // bin file write
	{
		f = fopen(fname, "wb");
	}
	else if(columns>0)  // txt file read
	{
		f = fopen(fname, "rt");
		unsigned char cpre='\n', c;
		while(1)  // skip comments
		{
			fscanf(f, "%c", &c);
			if(c==' ') continue;
			if(cpre=='\n' && c!='%') break;
			cpre = c;
		}
		int res0 = fseek(f, -1L, SEEK_CUR);
		int res1 = fseek(f, -1L, SEEK_CUR);
		this->columns = columns;
	}
	else
	{
		f = 0;
	}
}

int CFileRdWt::load(int lines)
{
	for(int i=0; i<lines; i++)
	{
		for(int j=0; j<columns; j++)	{ fscanf(f, "%lf,", &buff[j]); }		
//		for(int j=0; j<columns-1; j++)	{ fscanf(f, "%lf,", &buff[j]); }	fscanf(f, "%lf ", &buff[j]);	
		if(feof(f))  return 0;
	}
	return 1;
}

int CFileRdWt::IsEOF(void)
{
	return feof(f);
}

CFileRdWt& CFileRdWt::operator<<(double d)
{
	fwrite(&d, 1, sizeof(double), f);
	return *this;
}

CFileRdWt& CFileRdWt::operator<<(const CVect3 &v)
{
	fwrite(&v, 1, sizeof(v), f);
	return *this;
}

CFileRdWt& CFileRdWt::operator<<(const CVect &v)
{
	fwrite(v.dd, v.clm*v.row, sizeof(double), f);
	return *this;
}

CFileRdWt& CFileRdWt::operator<<(const CMat &m)
{
	fwrite(m.dd, m.clm*m.row, sizeof(double), f);
	return *this;
}

CFileRdWt& CFileRdWt::operator<<(const CRAvar &R)
{
	fwrite(R.R0, R.nR0, sizeof(double), f);
	return *this;
}

CFileRdWt& CFileRdWt::operator<<(const CSINS &sins)
{
	return *this<<sins.att<<sins.vn<<sins.pos<<sins.eb<<sins.db<<sins.tk;
}

CFileRdWt& CFileRdWt::operator<<(const CMahony &ahrs)
{
	return *this<<m2att(ahrs.Cnb)<<ahrs.exyzInt<<ahrs.tk;
}

CFileRdWt& CFileRdWt::operator<<(const CQEAHRS &ahrs)
{
	return *this<<m2att(ahrs.Cnb)<<*(CVect3*)&ahrs.Xk.dd[4]<<diag(ahrs.Pk)<<ahrs.kftk;
}

CFileRdWt& CFileRdWt::operator<<(const CKalman &kf)
{
	return *this<<kf.Xk<<diag(kf.Pk)<<kf.kftk;
}

CFileRdWt::~CFileRdWt()
{
	if(f) fclose(f); 
}

CFileRdWt& CFileRdWt::operator>>(double &d)
{
	fread(&d, 1, sizeof(double), f);
	return *this;
}

CFileRdWt& CFileRdWt::operator>>(CVect3 &v)
{
	fread(&v, 1, sizeof(v), f);
	return *this;
}

CFileRdWt& CFileRdWt::operator>>(CVect &v)
{
	fread(v.dd, v.clm*v.row, sizeof(double), f);
	return *this;
}

CFileRdWt& CFileRdWt::operator>>(CMat &m)
{
	fread(m.dd, m.clm*m.row, sizeof(double), f);
	return *this;
}

#endif //WIN32

/***************************  function AlignCoarse  *********************************/
CVect3 AlignCoarse(CVect3 wmm, CVect3 vmm, double latitude)
{
	double T11, T12, T13, T21, T22, T23, T31, T32, T33;
	double cl = cos(latitude), tl = tan(latitude), nn;
	CVect3 wbib = wmm / norm(wmm),  fb = vmm / norm(vmm);
	T31 = fb.i,				 T32 = fb.j,			 	T33 = fb.k;
	T21 = wbib.i/cl-T31*tl,	 T22 = wbib.j/cl-T32*tl,	T23 = wbib.k/cl-T33*tl;		nn = sqrt(T21*T21+T22*T22+T23*T23);  T21 /= nn, T22 /= nn, T23 /= nn;
	T11 = T22*T33-T23*T32,	 T12 = T23*T31-T21*T33,		T13 = T21*T32-T22*T31;		nn = sqrt(T11*T11+T12*T12+T13*T13);  T11 /= nn, T12 /= nn, T13 /= nn;
	CMat3 Cnb(T11, T12, T13, T21, T22, T23, T31, T32, T33);
	return m2att(Cnb);
}

CAligni0::CAligni0(const CVect3 &pos)
{
	eth.Update(pos);
	tk = 0;
	t0 = t1 = 10, t2 = 0; 
	wmm = vmm = vib0 = vi0 = Pib01 = Pib02 = Pi01 = Pi02 = O31;
	qib0b = CQuat(1.0);
}

CQuat CAligni0::Update(const CVect3 *wm, const CVect3 *vm, int nSamples, double ts)
{
	double nts = nSamples*ts;
	imu.Update(wm, vm, nSamples);
	wmm = wmm + imu.phim;  vmm = vmm + imu.dvbm;
	// vtmp = qib0b * (vm + 1/2 * wm X vm)
	CVect3 vtmp = qib0b*imu.dvbm;
	// vtmp1 = qni0' * [dvn+(wnin+wnie)Xvn-gn] * ts;
	tk += nts;
	CMat3 Ci0n = pos2Cen(CVect3(eth.pos.i,eth.wie*tk,0.0));
	CVect3 vtmp1 = Ci0n*(-eth.gn*nts);
	// Pib02 = Pib02 + vib0*ts, Pi02 = Pi02 + vi0*ts
	vib0 = vib0 + vtmp,		 vi0 = vi0 + vtmp1;
	Pib02 = Pib02 + vib0*nts, Pi02 = Pi02 + vi0*nts;
	//
	if(++t2>3*t0)
	{
		t0 = t1, Pib01 = tmpPib0, Pi01 = tmpPi0;
	}
	else if(t2>2*t0 && t1==t0)
	{
		t1 = t2, tmpPib0 = Pib02, tmpPi0 = Pi02;
	}
	//
	qib0b = qib0b*rv2q(imu.phim);
	// qnb=qni0*qiib0*qib0b
	CQuat qnb;
	if(t2<100)
	{
		qnb = CQuat(1.0);
	}
	else if(t2<1000)
	{
		qnb = a2qua(AlignCoarse(wmm, vmm, eth.pos.i));
	}
	else
	{
		qnb = (~m2qua(Ci0n))*m2qua(dv2att(Pi01, Pi02, Pib01, Pib02))*qib0b;
	}
	return qnb;
}

#ifndef ASSERT_NULL
BOOL assert(BOOL b)
{
	int res;

	if(b)
	{
		res = 1;
	}
	else
	{
		res = 0;
	}
	return res;
}
#endif

// determine the sign of 'val' with the sensitivity of 'eps'
int sign(double val, double eps)
{
	int s;

	if(val<-eps)
	{
		s = -1;
	}
	else if(val>eps)
	{
		s = 1;
	}
	else
	{
		s = 0; 
	}
	return s;
}

// set double value 'val' between range 'minVal' and 'maxVal'
double range(double val, double minVal, double maxVal)
{
	double res;

	if(val<minVal)
	{ 
		res = minVal; 
	}
	else if(val>maxVal)	
	{ 
		res = maxVal; 
	}
	else
	{ 
		res = val;
	}
	return res;
}

double atan2Ex(double y, double x)
{
	double res;

	if((sign(y)==0) && (sign(x)==0))
	{
		res = 0.0;
	}
	else
	{
		res = atan2(y, x);
	}
	return res;
}

double diffYaw(double yaw, double yaw0)
{
	double dyaw = yaw-yaw0;
	if(dyaw>=PI) dyaw-=_2PI;
	else if(dyaw<=-PI) dyaw+=_2PI;
	return dyaw;
}
