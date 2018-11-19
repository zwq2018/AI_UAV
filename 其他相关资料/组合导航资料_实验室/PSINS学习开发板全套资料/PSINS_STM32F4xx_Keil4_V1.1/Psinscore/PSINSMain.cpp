#include "kfapp.h"

CCarAHRS kf(TS_IMU);

int main1(void)
{
//	MyTest();
	CFileRdWt dir("H:\\ygm2016\\¾«×¼\\PSINS-CPP\\"), fin("data1.txt",29), fins("ins.bin",0), fkf("kf.bin",0);
		
	sensor.clear();
	for(int i=0; i<2500/TS_IMU; i++)
	{
		if(!sensor.set(fin)) break;
		kf.Update(10);
//		if(kf.yawAlignOK)
		{
			fins<<kf.sins<<kf.raINS<<kf.raGPSVn<<kf.raGPSPos<<kf.raGPSYaw<<kf.measRes<<(double)kf.navState;
			fkf<<kf;
		}
		if(i%10000==0)	printf("%d\n",i/100);
	}
	return 0;
}

/******************************** IO to c file ***********************************/
extern "C" void PSINSInit(void)
{
	sensor.clear();
}

extern "C" void PSINSSetIMU(double *ga, double ts)
{
	sensor.imuValid = 1;  sensor.nn =1; sensor.ts = ts;
	sensor.wm[0] = CVect3(&ga[0])*glv.dps*ts;  sensor.vm[0] = CVect3(&ga[3])*ts;  sensor.nts = sensor.nn*ts;
	sensor.imut += sensor.nts;
}

extern "C" void PSINSUpdate(int nStep)
{
	kf.Update(nStep);
}

extern "C" void PSINSOut(double *avp)
{
	avp[0] = kf.sins.att.i/glv.deg;  avp[1] = kf.sins.att.j/glv.deg;  avp[2] = kf.sins.att.k/glv.deg;
	avp[3] = kf.sins.vn.i;  avp[4] = kf.sins.vn.j;  avp[5] = kf.sins.vn.k;
	avp[6] = kf.sins.pos.i/glv.deg;  avp[7] = kf.sins.pos.j/glv.deg;  avp[8] = kf.sins.pos.k;	
	avp[9] = kf.sins.tk;
}

/* Use the following command to show the results in Matlab/PSINS Toolbox:
glvs
psinstypedef(153);
fid=fopen('ins.bin'); avp=fread(fid, [27,inf], 'double')'; fclose(fid); insplot(avp(:,1:16));
fid=fopen('kf.bin'); xkpk=fread(fid, [33,inf], 'double')'; fclose(fid); kfplot(xkpk(:,[1:15,17:31,end]));
*/
