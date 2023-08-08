#ifndef _MATRIXMUL_H_
#define _MATRIXMUL_H_

#define CHECK_RESULT 1
#define ENABLE_NAIVE 1

#define BLOCK_SIZE 16

#define VECTOR_SIZE 4


#define WA (32 * BLOCK_SIZE) 
#define HA (16 * BLOCK_SIZE)
#define WB (24 * BLOCK_SIZE)
#define HB WA
#define WC WB
#define HC HA

#endif
