#include "KFApp.h"


/***************************  class CCarAHRS  *********************************/
CKFApp::CKFApp(void):CSINSTDKF(22,9)
{
}

void CKFApp::Init28(CSINS &sins0)  // FOG
{
	// phi(3), dvn(3), dpos(3), eb(3), db(3), d(3)L, mu(3), dt(1)
	sins = sins;  kftk = sins.tk;
	measAtt = measGPSVn = measGPSPos = O31;
	Pmax.Set2(10.0*glv.deg,10.0*glv.deg,30.0*glv.deg, 50.0,50.0,50.0, 1.0e4/glv.Re,1.0e4/glv.Re,1.0e4, 
		1000.0*glv.dph,1000.0*glv.dph,1000.0*glv.dph, 100.0*glv.mg,100.0*glv.mg,100.0*glv.mg,
		10.0,10.0,10.0, 10.0*glv.deg,10.0*glv.deg,10.0*glv.deg, 1.0);
	Pmin.Set2(.10*glv.min,.10*glv.min,0.3*glv.min, 0.01,0.01,0.1, 1.0/glv.Re,1.0/glv.Re,1.0, 
		0.1*glv.dph,0.1*glv.dph,0.1*glv.dph, 0.1*glv.mg,0.1*glv.mg,0.1*glv.mg,
		0.0,0.0,0.0, 1.0*glv.min,1.0*glv.min,1.0*glv.min, 0.001);
	Pk.SetDiag2(1.0*glv.deg,1.0*glv.deg,10.0*glv.deg, 1.0,1.0,1.0, 100.0/glv.Re,100.0/glv.Re,100.0, 
		2.0*glv.dph,2.0*glv.dph,2.0*glv.dph, 10.0*glv.mg,10.0*glv.mg,10.0*glv.mg,
		0.0,0.0,0.0, 1.0*glv.deg,1.0*glv.deg,10.0*glv.deg, 0.1);
	Qt.Set2(.1*glv.dpsh,.1*glv.dpsh,.1*glv.dpsh, 100.0*glv.ugpsHz,100.0*glv.ugpsHz,100.0*glv.ugpsHz, 0.0,0.0,0.0,
		0.0,0.0,0.0, 0.0*glv.ugpsh,0.0*glv.ugpsh,0.0*glv.ugpsh, 0.0*glv.ppmpsh,
		0.0,0.0,0.0, 0.0,0.0,0.0, 0.0);
	Rt.Set2(10.0*glv.min, 10.0*glv.min, 10.0*glv.min, 0.5, 0.5, 0.5, 10.0/glv.Re, 10.0/glv.Re, 10.0);
	FBTau.Set(1.0,1.0,10.0,  1.0,1.0,1.0,  1.0,1.0,1.0,  10.0,10.0,10.0,  10.0,10.0,10.0,
		INF,INF,INF, INF,INF,INF, INF);
	SetHk(); 
}

void CKFApp::Init16(CSINS &sins0)  // MEMS
{
	// phi(3), dvn(3), dpos(3), eb(3), db(3), d(3)L, mu(3), dt(1)
	sins = sins0;  kftk = sins.tk;
	measAtt = measGPSVn = measGPSPos = O31;
	Pmax.Set2(10.0*glv.deg,10.0*glv.deg,30.0*glv.deg, 50.0,50.0,50.0, 1.0e4/glv.Re,1.0e4/glv.Re,1.0e4, 
		1000.0*glv.dph,1000.0*glv.dph,1000.0*glv.dph, 100.0*glv.mg,100.0*glv.mg,100.0*glv.mg,
		10.0,10.0,10.0, 10.0*glv.deg,10.0*glv.deg,10.0*glv.deg, 1.0);
	Pmin.Set2(1.0*glv.min,1.0*glv.min,1.0*glv.min, 0.01,0.01,0.1, 1.0/glv.Re,1.0/glv.Re,1.0, 
		1.0*glv.dph,1.0*glv.dph,1.0*glv.dph, 0.1*glv.mg,0.1*glv.mg,0.1*glv.mg,
		0.0,0.0,0.0, 1.0*glv.min,1.0*glv.min,1.0*glv.min, 0.001);
	Pk.SetDiag2(1.0*glv.deg,1.0*glv.deg,30.0*glv.deg, 1.0,1.0,1.0, 100.0/glv.Re,100.0/glv.Re,100.0, 
		100.0*glv.dph,100.0*glv.dph,100.0*glv.dph, 10.0*glv.mg,10.0*glv.mg,10.0*glv.mg,
		0.50,0.50,0.50, 1.0*glv.deg,1.0*glv.deg,10.0*glv.deg, 0.1);
	Qt.Set2(1.0*glv.dpsh,1.0*glv.dpsh,1.0*glv.dpsh, 100.0*glv.ugpsHz,100.0*glv.ugpsHz,100.0*glv.ugpsHz, 0.0,0.0,0.0,
		0.0,0.0,0.0, 0.0*glv.ugpsh,0.0*glv.ugpsh,0.0*glv.ugpsh, 0.0*glv.ppmpsh,
		0.0,0.0,0.0, 0.0,0.0,0.0, 0.0);
	Rt.Set2(10.0*glv.min, 10.0*glv.min, 10.0*glv.min, 0.5, 0.5, 0.5, 10.0/glv.Re, 10.0/glv.Re, 10.0);
	FBTau.Set(1.0,1.0,10.0,  1.0,1.0,1.0,  1.0,1.0,1.0,  10.0,10.0,10.0,  10.0,10.0,10.0,
		INF,INF,INF, INF,INF,INF, INF);
	SetHk(); 
	Ratt = Rvn = Rpos = CRAvar(3);
	Ratt.set(sqrt(*(CVect3*)&Rt.dd[0]), I31*10.0);
	Rvn.set(sqrt(*(CVect3*)&Rt.dd[3]), I31*10.0);
	Rpos.set(sqrt(*(CVect3*)&Rt.dd[6]), I31*10.0);
}

void CKFApp::SetFt(void)
{
	CSINSKF::SetFt();
}

void CKFApp::SetHk(void)
{
	sins.lever();
	Hk(0,0) = Hk(1,1) = Hk(2,2) = 1.0;   Hk.SetMat3(0,18,-sins.Cnb);
	Hk(3,3) = Hk(4,4) = Hk(5,5) = 1.0;   Hk.SetMat3(3,15,-sins.CW);			Hk.SetClmVect3(3,21,-sins.an);
	Hk(6,6) = Hk(7,7) = Hk(8,8) = 1.0;   Hk.SetMat3(6,15,-sins.MpvCnb);		Hk.SetClmVect3(6,21,-(sins.Mpv*sins.vn));
}

void CKFApp::SetMeas(void)
{
	double dt = sins.tk - tm -0.01*0;
	if(fabs(dt)>0.5) return;
	if(!measAtt.IsZero())
	{
		CQuat qa = a2qua(measAtt);
		*(CVect3*)&Zk.dd[0] = sins.qnb*rv2q(sins.wnb*(-dt))-qa;
		Ratt.Update(measAtt, 1.0);
		SetMeasFlag(0007);
	}
	if(!measGPSVn.IsZero())
	{
		*(CVect3*)&Zk.dd[3] = sins.vn - measGPSVn - sins.an*dt;
		Rvn.Update(measGPSVn, 1.0);
		SetMeasFlag(0070);
	}
	if(!measGPSPos.IsZero())
	{
		*(CVect3*)&Zk.dd[6] = sins.pos - measGPSPos - sins.eth.vn2dpos(sins.vn,dt);
		Rpos.Update(measGPSPos, 1.0);
		SetMeasFlag(0700);
	}
	if(this->measflag)
	{
		SetHk();
		measAtt = measGPSVn = measGPSPos = O31;
	}
}

void CKFApp::SetMeas(CVect3 *attm, CVect3 *vnm, CVect3 *posm, double tm)
{
	if(attm) measAtt = *attm;
	if(vnm) measGPSVn = *vnm;
	if(posm) measGPSPos = *posm;
	this->tm = tm;
}
