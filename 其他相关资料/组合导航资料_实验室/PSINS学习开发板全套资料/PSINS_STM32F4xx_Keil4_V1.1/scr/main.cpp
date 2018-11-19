#include "main.h"
#include "PSINS.h"

CMahony ahrs;
CVect3 wm, vm, att;

int main(void)
{
	mcu_init();
	
	ahrs = CMahony(4.0);
	double ts = 0.01;
		 
	while(1)
	{
		if(GAMT_OK_flag==1)
		{
			wm.i = mpu_Data_value.Gyro[0];  wm.j = mpu_Data_value.Gyro[1];  wm.k = mpu_Data_value.Gyro[2];
			vm.i = mpu_Data_value.Accel[0]; vm.j = mpu_Data_value.Accel[1]; vm.k = mpu_Data_value.Accel[2]; 
			ahrs.Update(wm, vm, O31, ts);
			att = q2att(ahrs.qnb)/glv.deg; 
			out_data.Att[0] = att.i; out_data.Att[1] = att.j; out_data.Att[2] = att.k;
			GAMT_OK_flag = 0;
		}	
	}
}
