#include "main.h"

void USARTx_SendStr(USART_TypeDef* USARTx,u8 str[], u16 size)
{ 
	u16 i;
	
	while(USART_GetFlagStatus(USARTx, USART_FLAG_TC)==RESET){;}  //等待发送完成
	for(i=0;i<size;i++)
	{		
		USART_SendData(USARTx,str[i]);
		while(USART_GetFlagStatus(USARTx, USART_FLAG_TC)==RESET);  //等待发送完成	
	}
}

void GPS_message(void)
{
	s32 GPS_temp;
  double GPS_temp1;
	
	gps_Data_value.GPS_ITOW=(u32)((Rx2_data[9]<<24)+(Rx2_data[8]<<16)+(Rx2_data[7]<<8)+Rx2_data[6]);
	
	///////////////////////////////////////////////////GPS_vleE/N/U	
	GPS_temp=((Rx2_data[61]<<24)+(Rx2_data[60]<<16)+(Rx2_data[59]<<8)+(Rx2_data[58]));
	gps_Data_value.GPS_Vn[0]=(double)GPS_temp/(double)1000;
	GPS_temp=((Rx2_data[57]<<24)+(Rx2_data[56]<<16)+(Rx2_data[55]<<8)+(Rx2_data[54]));
	gps_Data_value.GPS_Vn[1]=(double)GPS_temp/(double)1000;
	GPS_temp=((Rx2_data[65]<<24)+(Rx2_data[64]<<16)+(Rx2_data[63]<<8)+(Rx2_data[62]));
	gps_Data_value.GPS_Vn[2]=(double)GPS_temp/(double)-1000;
	
	///////////////////////////////////////////////////GPS_Lon/Lan
	GPS_temp=((Rx2_data[33]<<24)+(Rx2_data[32]<<16)+(Rx2_data[31]<<8)+(Rx2_data[30]));
	gps_Data_value.GPS_Pos[1]=(double)GPS_temp*DEG/(double)10000000;
	GPS_temp=((Rx2_data[37]<<24)+(Rx2_data[36]<<16)+(Rx2_data[35]<<8)+(Rx2_data[34]));
	gps_Data_value.GPS_Pos[0]=(double)GPS_temp*DEG/(double)10000000;
  GPS_temp=((Rx2_data[45]<<24)+(Rx2_data[44]<<16)+(Rx2_data[43]<<8)+(Rx2_data[42]));
	gps_Data_value.GPS_Pos[2]=(double)GPS_temp/(double)1000;
	
	///////////////////////////////////////////////////headMot
	GPS_temp=((Rx2_data[73]<<24)+(Rx2_data[72]<<16)+(Rx2_data[71]<<8)+(Rx2_data[70]));
	GPS_temp1=(double)GPS_temp/(double)100000;;
	if(GPS_temp1>180.0)
	{
		GPS_temp1=GPS_temp1-360.0;
	}
	gps_Data_value.GPS_Mot=GPS_temp1*DEG;
	/////////////////////////////////////////////////GPS_fixType		
	gps_Data_value.GPS_fixType=Rx2_data[26]&0x07;
	/////////////////////////////////////////////////GPS_flags		
	gps_Data_value.GPS_flags=Rx2_data[27]&0x01;
  /////////////////////////////////////////////////GPS_numSV		
	GPS_temp1=Rx2_data[29];
	if(GPS_temp1>15)
	{
		GPS_temp1=15;
	}
	gps_Data_value.GPS_numSV=GPS_temp1;
  /////////////////////////////////////////////////GPS_pDOP		
	GPS_temp=((Rx2_data[83]<<24)+(Rx2_data[82]<<16))>>16;
	gps_Data_value.GPS_pDOP=(double)GPS_temp/(double)100;
	
	GPS_temp=((Rx2_data[49]<<24)+(Rx2_data[48]<<16)+(Rx2_data[47]<<8)+(Rx2_data[46]));
	gps_Data_value.GPS_hAcc=(double)GPS_temp/(double)1000;
	GPS_temp=((Rx2_data[53]<<24)+(Rx2_data[52]<<16)+(Rx2_data[51]<<8)+(Rx2_data[50]));
	gps_Data_value.GPS_vAcc=(double)GPS_temp/(double)1000;
	GPS_temp=((Rx2_data[81]<<24)+(Rx2_data[80]<<16)+(Rx2_data[79]<<8)+(Rx2_data[78]));
	gps_Data_value.GPS_headAcc=(double)GPS_temp/(double)100000;
	GPS_temp=((Rx2_data[77]<<24)+(Rx2_data[76]<<16)+(Rx2_data[75]<<8)+(Rx2_data[74]));
	gps_Data_value.GPS_sAcc=(double)GPS_temp/(double)1000;
	GPS_temp=((Rx2_data[69]<<24)+(Rx2_data[68]<<16)+(Rx2_data[67]<<8)+(Rx2_data[66]));
	gps_Data_value.GPS_gSpeed=(double)GPS_temp/(double)1000;
	
}

void GPS_state(void)
{
   if(PPs_cnt==2)
	 {
		 GPS_break_cnt=GPS_break_cnt+10;
	 }
	 if(GPS_break_cnt==1000)
   {
		 GPS_break_cnt=0;
		 PPs_cnt=0;		 
	 }

}

void Data_updata(void)
{
	float Data_F;
	double Data_D;
	u32 Data_U;
//////////////////////////////////////////////////	
	out_data.OUT_cnt=(float)((double)OUT_cnt/(double)1000);
	out_data.Gyro[0]=(float)mpu_Data_value.Gyro[0];
	out_data.Gyro[1]=(float)mpu_Data_value.Gyro[1];
	out_data.Gyro[2]=(float)mpu_Data_value.Gyro[2];
	
	out_data.Accel[0]=(float)(mpu_Data_value.Accel[0]*(double)9.8);
	out_data.Accel[1]=(float)(mpu_Data_value.Accel[1]*(double)9.8);
	out_data.Accel[2]=(float)(mpu_Data_value.Accel[2]*(double)9.8);
	
	out_data.Magn[0]=(float)mpu_Data_value.Mag[0];
	out_data.Magn[1]=(float)mpu_Data_value.Mag[1];
	out_data.Magn[2]=(float)mpu_Data_value.Mag[2];
	
	out_data.mBar=(float)(mpu_Data_value.Pressure);
	
	out_data.GPS_Vn[0]=(float)(gps_Data_value.GPS_Vn[0]);
	out_data.GPS_Vn[1]=(float)(gps_Data_value.GPS_Vn[1]);
	out_data.GPS_Vn[2]=(float)(gps_Data_value.GPS_Vn[2]);
	
	Data_D=gps_Data_value.GPS_Pos[1]/DEG;
	Data_U=(u32)Data_D;
	Data_F=(float)(Data_D-(double)Data_U);
	out_data.GPS_Pos[0]=(float)Data_U;
	out_data.GPS_Pos[1]=Data_F;
	
	Data_D=gps_Data_value.GPS_Pos[0]/DEG;
	Data_U=(u32)Data_D;
	Data_F=(float)(Data_D-(double)Data_U);
	out_data.GPS_Pos[2]=(float)Data_U;
	out_data.GPS_Pos[3]=Data_F;
	
	out_data.GPS_Pos[4]=(float)gps_Data_value.GPS_Pos[2];
	
	if(gps_Data_value.GPS_pDOP>99){gps_Data_value.GPS_pDOP=99;}
	out_data.GPS_status=(float)gps_Data_value.GPS_numSV*(float)1000+gps_Data_value.GPS_pDOP;
	
	out_data.GPS_delay=(float)GPS_Delay/(float)1000;
	
	out_data.Temp=mpu_Data_value.Temp;
//////////////////////////////////////////////////	
	*(  u32*)&Usart1_out_DATA[ 0*4] = 0x56aa55aa;
	*(float*)&Usart1_out_DATA[ 1*4] = out_data.OUT_cnt;
	*(float*)&Usart1_out_DATA[ 2*4] = out_data.Gyro[0];
	*(float*)&Usart1_out_DATA[ 3*4] = out_data.Gyro[1];
	*(float*)&Usart1_out_DATA[ 4*4] = out_data.Gyro[2];
	*(float*)&Usart1_out_DATA[ 5*4] = out_data.Accel[0];
	*(float*)&Usart1_out_DATA[ 6*4] = out_data.Accel[1];
	*(float*)&Usart1_out_DATA[ 7*4] = out_data.Accel[2];
	*(float*)&Usart1_out_DATA[ 8*4] = out_data.Magn[0];
	*(float*)&Usart1_out_DATA[ 9*4] = out_data.Magn[1];
	*(float*)&Usart1_out_DATA[10*4] = out_data.Magn[2];
	*(float*)&Usart1_out_DATA[11*4] = out_data.mBar;
	*(float*)&Usart1_out_DATA[12*4] = out_data.Att[0];
	*(float*)&Usart1_out_DATA[13*4] = out_data.Att[1];
	*(float*)&Usart1_out_DATA[14*4] = out_data.Att[2];
	*(float*)&Usart1_out_DATA[15*4] = out_data.Vn[0];
	*(float*)&Usart1_out_DATA[16*4] = out_data.Vn[1];
	*(float*)&Usart1_out_DATA[17*4] = out_data.Vn[2];
	*(float*)&Usart1_out_DATA[18*4] = out_data.Pos[0];
	*(float*)&Usart1_out_DATA[19*4] = out_data.Pos[1];
	*(float*)&Usart1_out_DATA[20*4] = out_data.Pos[2];
	*(float*)&Usart1_out_DATA[21*4] = out_data.Pos[3];
	*(float*)&Usart1_out_DATA[22*4] = out_data.Pos[4];
	*(float*)&Usart1_out_DATA[23*4] = out_data.GPS_Vn[0];
	*(float*)&Usart1_out_DATA[24*4] = out_data.GPS_Vn[1];
	*(float*)&Usart1_out_DATA[25*4] = out_data.GPS_Vn[2];
	*(float*)&Usart1_out_DATA[26*4] = out_data.GPS_Pos[0];
	*(float*)&Usart1_out_DATA[27*4] = out_data.GPS_Pos[1];
	*(float*)&Usart1_out_DATA[28*4] = out_data.GPS_Pos[2];
	*(float*)&Usart1_out_DATA[29*4] = out_data.GPS_Pos[3];
	*(float*)&Usart1_out_DATA[30*4] = out_data.GPS_Pos[4];
	*(float*)&Usart1_out_DATA[31*4] = out_data.GPS_status;
	*(float*)&Usart1_out_DATA[32*4] = out_data.GPS_delay;
	*(float*)&Usart1_out_DATA[33*4] = out_data.Temp;
	
	Usart1_out_Length=34*4;	
}
