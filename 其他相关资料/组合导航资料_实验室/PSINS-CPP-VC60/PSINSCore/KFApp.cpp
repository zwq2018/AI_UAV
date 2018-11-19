#include "KFApp.h"


/***************************  class CCarAHRS  *********************************/
CKFApp::CKFApp(void):CSINSTDKF(15,6)
{
}

void CKFApp::Init16(const CSINS &sins0)  // MEMS
{
	// phi(3), dvn(3), dpos(3), eb(3), db(3)
	sins = sins0;  kftk = sins.tk;
	measGPSVn = measGPSPos = O31;
	Pmax.Set2(10.0*glv.deg,10.0*glv.deg,30.0*glv.deg, 50.0,50.0,50.0, 1.0e4/glv.Re,1.0e4/glv.Re,1.0e4, 
		1000.0*glv.dph,1000.0*glv.dph,1000.0*glv.dph, 100.0*glv.mg,100.0*glv.mg,100.0*glv.mg);
	Pmin.Set2(1.0*glv.min,1.0*glv.min,1.0*glv.min, 0.01,0.01,0.1, 1.0/glv.Re,1.0/glv.Re,1.0, 
		1.0*glv.dph,1.0*glv.dph,1.0*glv.dph, 0.1*glv.mg,0.1*glv.mg,0.1*glv.mg);
	Pk.SetDiag2(1.0*glv.deg,1.0*glv.deg,30.0*glv.deg, 1.0,1.0,1.0, 100.0/glv.Re,100.0/glv.Re,100.0, 
		100.0*glv.dph,100.0*glv.dph,100.0*glv.dph, 10.0*glv.mg,10.0*glv.mg,10.0*glv.mg);
	Qt.Set2(1.0*glv.dpsh,1.0*glv.dpsh,1.0*glv.dpsh, 100.0*glv.ugpsHz,100.0*glv.ugpsHz,100.0*glv.ugpsHz, 0.0,0.0,0.0,
		0.0,0.0,0.0, 0.0,0.0,0.0);
	Rt.Set2(0.5, 0.5, 0.5, 10.0/glv.Re, 10.0/glv.Re, 10.0);
	Rmax = Rt*100; Rmin = Rt*0.01;	Rb.Set(0.9,0.9,0.9,  0.9,0.9,0.9);
	FBTau.Set(1.0,1.0,10.0,  1.0,1.0,1.0,  1.0,1.0,1.0,  10.0,10.0,10.0,  10.0,10.0,10.0);
	FBMax.Set(INF,INF,INF,  INF,INF,INF,  INF,INF,INF,  3000.0*glv.dph,3000.0*glv.dph,3000.0*glv.dph,  50.0*glv.mg,50.0*glv.mg,50.0*glv.mg); 
}

void CKFApp::SetMeas(void)
{
	double dt = sins.tk - tmeas -0.01*0;
	if(fabs(dt)>0.5) return;
	if(!measGPSVn.IsZero())
	{
		*(CVect3*)&Zk.dd[0] = sins.vn - measGPSVn - sins.an*dt;
		SetMeasFlag(007);
	}
	if(!measGPSPos.IsZero())
	{
		*(CVect3*)&Zk.dd[3] = sins.pos - measGPSPos - sins.eth.vn2dpos(sins.vn,dt);
		SetMeasFlag(070);
	}
	if(this->measflag)
	{
		SetHk();
		measGPSVn = measGPSPos = O31;
	}
}

void CKFApp::SetMeas(CVect3 *vnm, CVect3 *posm, double tm)
{
	if(vnm) measGPSVn = *vnm;
	if(posm) measGPSPos = *posm;
	this->tmeas = tm;
}
