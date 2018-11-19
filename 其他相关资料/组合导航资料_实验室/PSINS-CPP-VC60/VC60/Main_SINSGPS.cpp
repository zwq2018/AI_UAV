#include "..\PSINSCore\kfapp.h"

CKFApp kf;
CFileRdWt dir("H:\\ygm2018\\¾«×¼\\PSINS-CPP\\Data\\");

int main(void)
{
	CFileRdWt fin("data1.txt",29), fins("ins.bin",0), fkf("kf.bin",0), frk("rk.bin",0);
	CVect3 wm, vm, gpsvn, gpspos;
	double timu=0, ts = 0.01;

	gpspos = CVect3(34.196255*glv.deg,108.875677*glv.deg, 410.70);
	kf.Init16(CSINS(a2qua(CVect3(-0.821,3.690,6.960)*glv.deg), O31, gpspos, timu));
	for(int i=1; i<2000*100; i++)
	{
		fin.load();	if(fin.IsEOF()) break;
		timu = fin.buff[0];
		wm = *(CVect3*)&fin.buff[1];  vm = *(CVect3*)&fin.buff[4];
		wm = wm*glv.dps*ts; vm = vm*ts;

		if(fin.buff[19]>0 && timu>60)
		{
			gpsvn = *(CVect3*)&fin.buff[16]; gpspos = *(CVect3*)&fin.buff[19];
			gpspos.i *=glv.deg; gpspos.j *=glv.deg; 
			kf.SetMeas(&gpsvn, &gpspos, timu);
		}
		kf.TDUpdate(&wm, &vm, 1, ts, 10);
//		kf.Update(&wm, &vm, 1, ts);
		fkf<<kf;
		fins<<kf.sins;
		frk<<kf.Rt; 
		if(i%1000==0) printf("%d\n", i/100);
	}

	return 0;
}
