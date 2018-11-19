#ifndef __MAIN_H
#define __MAIN_H

#include "mcu_init.h"
#include "stm32f4xx.h"
#include "arm_math.h" 
#include "stm32f4xx_it.h"
#include "mpu9250.h"

extern MPU_AD_value		mpu_AD_value;
extern MPU_Data_value mpu_Data_value;
extern GPS_Data_value gps_Data_value; 
extern INS_Data_value ins_Data_value;
extern Out_Data				out_data;

extern u8 MS5611_cnt;

extern u8 Usart1_out_DATA[200];
extern u8 Usart1_out_Length;

extern u8  Rx2_data[120];                     
extern u8  Rx2_complete;                 
extern u16 Length2;

extern u8  PPs_cnt;
extern u8  GPS_exist;
extern u16 GPS_break_cnt;

extern u32 OUT_cnt;
extern u32 GPS_Delay;

extern u8  GAMT_OK_flag;
extern u8  GPS_OK_flag;
extern u8  Bar_OK_flag;

#endif /* __MAIN_H */

