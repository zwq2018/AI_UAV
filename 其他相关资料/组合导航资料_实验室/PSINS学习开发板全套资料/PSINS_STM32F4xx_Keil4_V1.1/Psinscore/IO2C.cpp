#include "kfapp.h"


/////////////////////////////////////////// kf ////////////////////////////////////////////////
static CKFApp kf;

extern "C" void PSINSInit(double *att, double *vn, double *pos, double t0)
{
	CSINS sins = CSINS(a2qua(*(CVect3*)att), *(CVect3*)vn, *(CVect3*)pos, t0);
	kf.Init16(sins);
}

extern "C" void PSINSSetMeas(double *attm, double *vnm, double *posm, double tm)
{
	kf.SetMeas((CVect3*)attm, (CVect3*)vnm, (CVect3*)posm, tm);
}

extern "C" void PSINSUpdate(double *wm, double *vm, int n, double tk, int nStep)
{
	kf.TDUpdate((CVect3*)wm, (CVect3*)vm, n, tk, nStep);
}

extern "C" void PSINSOut(double *att, double *vn, double *pos)
{
	*(CVect3*)att = kf.sins.att;  *(CVect3*)vn = kf.sins.vn;  *(CVect3*)pos = kf.sins.pos;
}


/////////////////////////////////////////// ahrs ////////////////////////////////////////////////
static CMahony ahrs;

extern "C" void MahonyInit(float tau, float *att)
{
	ahrs = CMahony(tau, a2qua(CVect3(att[0],att[1],att[2]))); 
}

extern "C" void MahonyUpdate(float *gyro, float *acc, float *mag, float ts)
{
	ahrs.Update(CVect3(gyro[0],gyro[1],gyro[2]), CVect3(acc[0],acc[1],acc[2]), CVect3(mag[0],mag[1],mag[2]), ts);
}

extern "C" float MahonyGetEuler(float *att)
{
	CVect3 attv = m2att(ahrs.Cnb);
	float deg = 3.14159/180;
	att[0] = (float)attv.i/deg; att[1] = (float)attv.j/deg; att[2] = (float)attv.k/deg;
	return (float)ahrs.tk;
}

extern "C" void MahonyGetDrift(float *drift)
{
	float dph = 3.14159/180/3600;
	drift[0] = (float)ahrs.exyzInt.i/dph; drift[1] = (float)ahrs.exyzInt.j/dph; drift[2] = (float)ahrs.exyzInt.k/dph;
}
