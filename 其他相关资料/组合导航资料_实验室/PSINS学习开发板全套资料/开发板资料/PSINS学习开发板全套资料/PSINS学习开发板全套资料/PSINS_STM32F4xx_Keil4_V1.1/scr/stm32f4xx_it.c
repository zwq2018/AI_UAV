/**
  ******************************************************************************
  * @file    Project/STM32F4xx_StdPeriph_Template/stm32f4xx_it.c 
  * @author  MCD Application Team
  * @version V1.0.1
  * @date    13-April-2012
  * @brief   Main Interrupt Service Routines.
  *          This file provides template for all exceptions handler and 
  *          peripherals interrupt service routine.
  ******************************************************************************
  * @attention
  *
  * <h2><center>&copy; COPYRIGHT 2012 STMicroelectronics</center></h2>
  *
  * Licensed under MCD-ST Liberty SW License Agreement V2, (the "License");
  * You may not use this file except in compliance with the License.
  * You may obtain a copy of the License at:
  *
  *        http://www.st.com/software_license_agreement_liberty_v2
  *
  * Unless required by applicable law or agreed to in writing, software 
  * distributed under the License is distributed on an "AS IS" BASIS, 
  * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  * See the License for the specific language governing permissions and
  * limitations under the License.
  *
  ******************************************************************************
  */

/* Includes ------------------------------------------------------------------*/
#include "stm32f4xx_it.h"
#include "main.h"

/** @addtogroup Template_Project
  * @{
  */

/* Private typedef -----------------------------------------------------------*/
/* Private define ------------------------------------------------------------*/
/* Private macro -------------------------------------------------------------*/
/* Private variables ---------------------------------------------------------*/
/* Private function prototypes -----------------------------------------------*/
/* Private functions ---------------------------------------------------------*/

/******************************************************************************/
/*            Cortex-M4 Processor Exceptions Handlers                         */
/******************************************************************************/

/**
  * @brief   This function handles NMI exception.
  * @param  None
  * @retval None
  */
void NMI_Handler(void)
{
}

/**
  * @brief  This function handles Hard Fault exception.
  * @param  None
  * @retval None
  */
void HardFault_Handler(void)
{
  /* Go to infinite loop when Hard Fault exception occurs */
  while (1)
  {
  }
}

/**
  * @brief  This function handles Memory Manage exception.
  * @param  None
  * @retval None
  */
void MemManage_Handler(void)
{
  /* Go to infinite loop when Memory Manage exception occurs */
  while (1)
  {
  }
}

/**
  * @brief  This function handles Bus Fault exception.
  * @param  None
  * @retval None
  */
void BusFault_Handler(void)
{
  /* Go to infinite loop when Bus Fault exception occurs */
  while (1)
  {
  }
}

/**
  * @brief  This function handles Usage Fault exception.
  * @param  None
  * @retval None
  */
void UsageFault_Handler(void)
{
  /* Go to infinite loop when Usage Fault exception occurs */
  while (1)
  {
  }
}

/**
  * @brief  This function handles SVCall exception.
  * @param  None
  * @retval None
  */
void SVC_Handler(void)
{
}

/**
  * @brief  This function handles Debug Monitor exception.
  * @param  None
  * @retval None
  */
void DebugMon_Handler(void)
{
}

/**
  * @brief  This function handles PendSVC exception.
  * @param  None
  * @retval None
  */
void PendSV_Handler(void)
{
}

/**
  * @brief  This function handles SysTick Handler.
  * @param  None
  * @retval None
  */
void SysTick_Handler(void)
{
 // TimingDelay_Decrement();
}

/******************************************************************************/
/*                 STM32F4xx Peripherals Interrupt Handlers                   */
/*  Add here the Interrupt Handler for the used peripheral(s) (PPP), for the  */
/*  available peripheral interrupt handler's name please refer to the startup */
/*  file (startup_stm32f4xx.s).                                               */
/******************************************************************************/

/**
  * @brief  This function handles PPP interrupt request.
  * @param  None
  * @retval None
  */
/*void PPP_IRQHandler(void)
{
}*/
void EXTI0_IRQHandler(void)
{
	u16 TIMCounter = 20;
	
	if(EXTI_GetITStatus(EXTI_Line0) != RESET)
  {	
		EXTI_ClearITPendingBit(EXTI_Line0);
		if(PPs_cnt<2)
    {
       PPs_cnt++;
		}
		if(PPs_cnt==2)
		{	
			TIM_SetCounter(TIM2, TIMCounter);
		}
	}	
}

void TIM2_IRQHandler(void)
{
	
	if (TIM_GetITStatus(TIM2, TIM_IT_Update) != RESET) 
  {
		 TIM_ClearITPendingBit(TIM2, TIM_IT_Update);
		
		 GPIO_SetBits(GPIOC,GPIO_Pin_3);
		
		 READ_MPU9250_A_T_G();
		 READ_MPU9250_MAG();
		 GAMT_OK_flag=1;
		 OUT_cnt=OUT_cnt+10;
		 if(PPs_cnt==0)
		 {
			 GPS_Delay=GPS_Delay+10;
		 }
		 else
		 {
			 GPS_Delay=0;
		 }
		 if(Rx2_complete==1)
		  {
			 GPS_message();
			 GPS_OK_flag=1;
			 GPS_exist=1;
			 Rx2_complete=0;
		  }
		 GPS_state();
		 Data_updata();
		 USART1_DIA_OUT_Configuration();
			
		 GPIO_ResetBits(GPIOC,GPIO_Pin_3);
	}
	
}

void TIM3_IRQHandler(void)
{
	
	if (TIM_GetITStatus(TIM3, TIM_IT_Update) != RESET) 
  {
		 //TIM_ClearITPendingBit(TIM3, TIM_IT_Update);
		 MS5611_cnt++;
		 if(MS5611_cnt==1)
			{
				MS561101BA_start_Temperature();
			}
			if(MS5611_cnt==2)
			{
				MS561101BA_getTemperature();
			}
			if(MS5611_cnt==3)
			{
				MS561101BA_start_Pressure();
			}
			if(MS5611_cnt==4)
			{
				MS561101BA_getPressure();					
			}
			if(MS5611_cnt==5)
			{
				Bar_OK_flag=1;
				MS5611_cnt=0;
			}
			TIM_ClearITPendingBit(TIM3, TIM_IT_Update);
			TIM_SetCounter(TIM3, 0);
	}
	
}

void USART2_IRQHandler(void)
{
	char temp;
	
	if(USART_GetITStatus(USART2, USART_IT_RXNE) != RESET)	   //判断读寄存器是否非空
  {	
      temp = USART_ReceiveData(USART2);   //将读寄存器的数据缓存到接收缓冲区里
      USART_ClearITPendingBit(USART2, USART_IT_RXNE);//清楚中断标志
		
		  if(PPs_cnt>0)
	    {
				Rx2_data[Length2]=temp;
				Length2++;
				if((Length2==1)&&(Rx2_data[0]!=0xb5))
				{
					Rx2_data[0]=0;
					Length2=0;
				}
				if((Length2==2)&&(Rx2_data[1]!=0x62))
				{
					Rx2_data[0]=Rx2_data[1]=0;
					Length2=0;
				}	
				if(Length2==100)
				{
					Rx2_complete=1;
					Length2=0;
				}		
			}
  }
	
}

void USART1_IRQHandler(void)
{
	
	if(USART_GetITStatus(USART1, USART_IT_RXNE) != RESET)	   //判断读寄存器是否非空
  {	
      USART_ReceiveData(USART1);   //将读寄存器的数据缓存到接收缓冲区里
      USART_ClearITPendingBit(USART1, USART_IT_RXNE);//清楚中断标志
  }
	
}

/**
  * @}
  */ 


/************************ (C) COPYRIGHT STMicroelectronics *****END OF FILE****/
