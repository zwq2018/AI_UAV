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
	double tmeas;
	CVect3 measGPSVn, measGPSPos;

	CKFApp(void);
	void Init16(const CSINS &sins0);
	virtual void SetMeas(void);
	void SetMeas(CVect3 *vnm, CVect3 *posm, double tm);
};

#endif

