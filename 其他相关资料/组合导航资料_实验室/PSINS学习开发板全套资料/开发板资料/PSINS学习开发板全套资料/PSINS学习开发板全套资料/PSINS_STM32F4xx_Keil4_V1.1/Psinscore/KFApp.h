/* KFApp c++ hearder file KFApp.h */
/*
	By     : Yan Gongmin @ NWPU
	Date   : 2017-04-29
	From   : College of Automation, 
	         Northwestern Polytechnical University, 
			 Xi'an 710072, China
*/

#ifndef _KFAPP_H
#define _KFAPP_H

#include "PSINS.h"


class CKFApp:public CSINSTDKF
{
public:
	double tm;
	CVect3 measAtt, measGPSVn, measGPSPos;
	CRAvar Ratt, Rvn, Rpos;

	CKFApp(void);
	void Init16(CSINS &sins0);
	void Init28(CSINS &sins0);
	virtual void SetFt(void);
	virtual void SetHk(void);
	virtual void SetMeas(void);
	void SetMeas(CVect3 *attm, CVect3 *vnm, CVect3 *posm, double tm);
};

#endif

