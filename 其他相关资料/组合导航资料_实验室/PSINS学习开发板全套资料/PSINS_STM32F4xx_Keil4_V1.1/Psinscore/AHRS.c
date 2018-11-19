#include "math.h"

// 最简单的航姿仪算法，所谓“最简单”就意味着在加速机动环境下效果会不太好，没做任何容错处理。
// 注意: 传感器/机体坐标系(b):x右-y前-z上; 导航坐标系(n):E东-N北-U天
// gx,gy,gz陀螺仪采样(rad/s), ax,ay,az加计采样(可为任意单位或归一化单位)
// mx,my,mz磁采样(可为任意单位或归一化单位)，若三轴磁均输入零则表示不利用磁信息
// 根据网上程序整理优化 by Yan Gongmin, 2017-06-07
void AHRSInit(void);
void AHRSupdate(float gx, float gy, float gz, float ax, float ay, float az, float mx, float my, float mz, float ts);
void AHRSGetEuler(float *pitch, float *roll, float *yaw);
void AHRSGetDrift(float *driftx, float *drifty, float *driftz);
float AHRSGetT(void);

static float q0,q1,q2,q3, exInt,eyInt,ezInt, tk;
static float C00,C01,C02, C10,C11,C12, C20,C21,C22;
static float Kp, Ki, Km;
static void qua2cnb(void);

void AHRSInit(void)
{
	Kp = 1.0f, Ki = 0.53f, Km = 1.0f;
  	q0 = 1.0f, q1 = q2 = q3 = 0.0f;
	qua2cnb();
  	exInt = eyInt = ezInt = 0.0;
	tk = 0.0;
}

void AHRSupdate(float gx, float gy, float gz, float ax, float ay, float az, float mx, float my, float mz, float ts)
{
	float norm;
	float by, bz, wx, wy, wz;
	float ex, ey, ez, halfT, KihalfT;
	float _q0, _q1, _q2, _q3;

	tk += ts;
	halfT =  ts/2.0f;  KihalfT = Ki*halfT;
	
	norm = 1.0f/sqrt(ax*ax + ay*ay + az*az);       // 加计归一化
	ax = ax * norm,  ay = ay * norm,  az = az * norm;
	
	norm = sqrt(mx*mx + my*my + mz*mz);       // 磁归一化
	if(norm>0.0001f)  norm = 1.0f/norm;
	mx = mx * norm,  my = my * norm,  mz = mz * norm;
	
	bz = C20*mx + C21*my + C22*mz;  // 磁场的天向投影分量
	by = sqrt(1.0f-bz*bz);          // 磁场的水平（磁北）投影分量
//	bx = 0.0f;                      // 磁场的磁东向投影分量
	
	wx = C10*by + C20*bz;			// 磁场的理想b系投影分量
	wy = C11*by + C21*bz;
	wz = C12*by + C22*bz;
	
	ex = (ay*C22-az*C21) + (my*wz-mz*wy)*Km;	// 加计测水平误差、磁测方位误差
	ey = (az*C20-ax*C22) + (mz*wx-mx*wz)*Km;
	ez = (ax*C21-ay*C20) + (mx*wy-my*wx)*Km;
	
	exInt += ex * KihalfT;
	eyInt += ey * KihalfT;
	ezInt += ez * KihalfT;
	gx += Kp*ex + exInt;		// 陀螺修正
	gy += Kp*ey + eyInt;
	gz += Kp*ez + ezInt;
	
	_q0 = q0 + (-q1*gx - q2*gy - q3*gz)*halfT;		// 解四元数微分方程
	_q1 = q1 + ( q0*gx + q2*gz - q3*gy)*halfT;
	_q2 = q2 + ( q0*gy - q1*gz + q3*gx)*halfT;
	_q3 = q3 + ( q0*gz + q1*gy - q2*gx)*halfT; 
	
	norm = 1.0f/sqrt(_q0*_q0 + _q1*_q1 + _q2*_q2 + _q3*_q3);	// 四元数归一化
	q0 = _q0 * norm, q1 = _q1 * norm, q2 = _q2 * norm, q3 = _q3 * norm;

	qua2cnb();
}

void AHRSGetEuler(float *pitch, float *roll, float *yaw)
{
	*pitch = asinf(C21) * 57.2957795f;   // deg
	*roll  = atan2f(-C20, C22) * 57.2957795f;
	*yaw   = atan2f(-C01, C11) * 57.2957795f;		// +-180, 北偏西为正 
}

void AHRSGetDrift(float *driftx, float *drifty, float *driftz)
{
	*driftx = exInt / 4.848e-6f;   // deg/hur
	*drifty = eyInt / 4.848e-6f;
	*driftz = ezInt / 4.848e-6f;
}

float AHRSGetT(void)
{
	return tk;
}

static void qua2cnb(void)
{
	float q00 = q0*q0, q01 = q0*q1, q02 = q0*q2, q03 = q0*q3;
	float q11 = q1*q1, q12 = q1*q2, q13 = q1*q3;
	float q22 = q2*q2, q23 = q2*q3;
	float q33 = q3*q3;
	C00 = q00+q11-q22-q33,  C01 = 2*(q12-q03),     C02 = 2*(q13+q02);  // 四元数转姿态阵
	C10 = 2*(q12+q03),      C11 = q00-q11+q22-q33, C12 = 2*(q23-q01);
	C20 = 2*(q13-q02),      C21 = 2*(q23+q01),     C22 = q00-q11-q22+q33;
}
