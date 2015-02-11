//Author: Baze Ilijoskki
//Program to search AB matrix within MN one using OpenCL in GPU (A<M, B<N)
 
#include "stdafx.h"
 
#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include < time.h >
#include < ctime >
 
#define MN_MAX_ROW	128
#define MN_MAX_COL	128

#define AB_MAX_ROW	128
#define AB_MAX_COL	128

 
#ifdef __APPLE__
#include < OpenCL/opencl.h >
#else
#include < CL/cl.h >
#endif
 
#define MEM_SIZE (128)
#define MAX_SOURCE_SIZE (0x100000)
 
int SearchMatrix(char *matrix_mn, int mn_row, int mn_col, char *matrix_ab, int ab_row, int ab_col, char *matrix_res)
{
	// Check parameter
	if (!matrix_mn || !matrix_ab || !matrix_res)
		return -1;
	if ((mn_row<ab_row) || (mn_col<ab_col))
		return -1;
	
	cl_device_id device_id = NULL;
	cl_context context = NULL;
	cl_command_queue command_queue = NULL;
	cl_mem memobjMN = NULL;
	cl_mem memobjAB = NULL;
	cl_mem memobjRES = NULL;
	cl_mem rowMN = NULL;
	cl_mem colMN = NULL;
	cl_mem rowAB = NULL;
	cl_mem colAB = NULL;
	cl_program program = NULL;
	cl_kernel kernel = NULL;
	cl_platform_id platform_id = NULL;
	cl_uint ret_num_devices;
	cl_uint ret_num_platforms;
	cl_int ret;

	//char string[MEM_SIZE];
	FILE *fp;
	char fileName[] = "./matrix.cl";
	char *source_str;
	size_t source_size;
	/* Load the source code containing the kernel*/
	fp = fopen(fileName, "r");
	if (!fp) {
		fprintf(stderr, "Failed to load kernel.\n");
		return -1;
	}
	
	source_str = (char*)malloc(MAX_SOURCE_SIZE);
	source_size = fread( source_str, 1, MAX_SOURCE_SIZE, fp);
	fclose( fp );
	
	/* Get Platform and Device Info */
	ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
	ret = clGetDeviceIDs( platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &ret_num_devices);

	/* Create OpenCL context */
	context = clCreateContext( NULL, 1, &device_id, NULL, NULL, &ret);

	/* Create Command Queue */
	command_queue = clCreateCommandQueue(context, device_id, 0, &ret);

	/* Create Memory Buffer */
	memobjMN = clCreateBuffer(context, CL_MEM_READ_WRITE, MN_MAX_ROW * MN_MAX_COL * sizeof(char), NULL, &ret);
	memobjAB = clCreateBuffer(context, CL_MEM_READ_WRITE, AB_MAX_ROW * AB_MAX_COL * sizeof(char), NULL, &ret);
	memobjRES = clCreateBuffer(context, CL_MEM_READ_WRITE, MN_MAX_ROW * MN_MAX_COL * sizeof(char), NULL, &ret);
	rowMN = clCreateBuffer(context, CL_MEM_READ_WRITE,  sizeof(int), NULL, &ret);
	colMN = clCreateBuffer(context, CL_MEM_READ_WRITE,  sizeof(int), NULL, &ret);
	rowAB = clCreateBuffer(context, CL_MEM_READ_WRITE,  sizeof(int), NULL, &ret);
	colAB = clCreateBuffer(context, CL_MEM_READ_WRITE,  sizeof(int), NULL, &ret);
 
  // Copy the lists MN and AB to their respective memory buffers
    ret = clEnqueueWriteBuffer(command_queue,memobjMN, CL_TRUE, 0,
           MN_MAX_ROW * MN_MAX_COL * sizeof(char), matrix_mn, 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(command_queue, memobjAB, CL_TRUE, 0,
            AB_MAX_ROW * AB_MAX_COL * sizeof(char), matrix_ab, 0, NULL, NULL);
	ret = clEnqueueWriteBuffer(command_queue, memobjRES, CL_TRUE, 0,
		    MN_MAX_ROW * MN_MAX_COL * sizeof(char), matrix_res, 0, NULL, NULL);
	int max_mn_row = MN_MAX_ROW;
	int max_mn_col = MN_MAX_COL;
	ret = clEnqueueWriteBuffer(command_queue, rowMN, CL_TRUE, 0, sizeof(int), &max_mn_row, 0, NULL, NULL);
	ret = clEnqueueWriteBuffer(command_queue, colMN, CL_TRUE, 0, sizeof(int), &max_mn_col, 0, NULL, NULL);
	ret = clEnqueueWriteBuffer(command_queue, rowAB, CL_TRUE, 0, sizeof(int), &ab_row, 0, NULL, NULL);
	ret = clEnqueueWriteBuffer(command_queue, colAB, CL_TRUE, 0, sizeof(int), &ab_col, 0, NULL, NULL);
	
	/* Create Kernel Program from the source */
	program = clCreateProgramWithSource(context, 1, (const char **)&source_str,
		(const size_t *)&source_size, &ret);
	
	/* Build Kernel Program */
	ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);

	/* Create OpenCL Kernel */
	kernel = clCreateKernel(program, "matrixSearch", &ret);
	
	/* Set OpenCL Kernel Arguments */
	ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&memobjMN);
	ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&memobjAB);
	ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&memobjRES);
	ret = clSetKernelArg(kernel, 3, sizeof(int), (void *)&rowMN);
	ret = clSetKernelArg(kernel, 4, sizeof(int), (void *)&colMN);
	ret = clSetKernelArg(kernel, 5, sizeof(int), (void *)&rowAB);
	ret = clSetKernelArg(kernel, 6, sizeof(int), (void *)&colAB);
	
	/* Execute OpenCL Kernel */
	//ret = clEnqueueTask(command_queue, kernel, 0, NULL,NULL);
	size_t globalThreads[2] = {mn_row, mn_col};
	size_t localThreads[2] = {16,16};
	
	clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, globalThreads, localThreads, NULL, 0, NULL);
	/* Copy results from the memory buffer */
	ret = clEnqueueReadBuffer(command_queue, memobjRES, CL_TRUE, 0,
                            MN_MAX_ROW * MN_MAX_COL * sizeof(char),matrix_res, 0, NULL, NULL);
 
	ret = clFlush(command_queue);
	ret = clFinish(command_queue);
	ret = clReleaseKernel(kernel);
	ret = clReleaseProgram(program);
	ret = clReleaseMemObject(memobjMN);
	ret = clReleaseMemObject(memobjAB);
	ret = clReleaseMemObject(memobjRES);
	ret = clReleaseCommandQueue(command_queue);
	ret = clReleaseContext(context);
	
	free(source_str);
	
	return 0;
}

int main()
{
	FILE *fp = NULL;
	char lineStr[1024];
	char *matrixMN = NULL;
	char *matrixAB = NULL;
	char *matrixDST = NULL;
	int colCnt, rowCnt;
	int mnRowCnt, mnColCnt;
	int abRowCnt, abColCnt;
	int i, j, res;

	// malloc matrixMN
	matrixMN = malloc(MN_MAX_ROW*MN_MAX_COL);
	if (!matrixMN)
	{
		printf("Could not allocate memory for matrixMN\n");
		goto MAIN_EXIT;
	}

	// malloc matrixAB
	matrixAB = malloc(AB_MAX_ROW*AB_MAX_COL);
	if (!matrixAB)
	{
		printf("Could not allocate memory for matrixAB\n");
		goto MAIN_EXIT;
	}

	// malloc matrixDST
	matrixDST = malloc(MN_MAX_ROW*MN_MAX_COL);
	if (!matrixDST)
	{
		printf("Could not allocate memory for matrixDST\n");
		goto MAIN_EXIT;
	}

	// Matrix MN

	// Open MxN matrix file
	if ((fp = fopen("matrixMN.txt", "r")) == NULL)
	{
		printf("Could not open matrixMN.txt\n");
		goto MAIN_EXIT;
	}

	// Read line
	colCnt = MN_MAX_COL+1;
	rowCnt = 0;
	while(fgets(lineStr, 1024, fp) != NULL)
	{
		int lineCnt = 0, mnCnt = 0;
		while ((lineStr[lineCnt]!='\n') && (lineCnt<1024) && (mnCnt<MN_MAX_COL))
		{
			if (lineStr[lineCnt] != ' ')
			{
				matrixMN[rowCnt*MN_MAX_ROW+mnCnt] = lineStr[lineCnt];
				mnCnt++;
			}
			lineCnt++;			
		}

		// Select MIN col count
		if (mnCnt < colCnt)
		{
			colCnt = mnCnt;
		}

		rowCnt++;
	}

	// Failed to read
	if (colCnt == (MN_MAX_COL+1))
	{
		printf("Could not read matrixMN.txt\n");
		goto MAIN_EXIT;
	}

	// Set Row, Col count of matrixMN
	mnRowCnt = rowCnt;
	mnColCnt = colCnt;

	fclose(fp);
	fp = NULL;

	// Matrix AB

	// Open AxB matrix file
	if ((fp = fopen("matrixAB.txt", "r")) == NULL)
	{
		printf("Could not open matrixAB.txt\n");
		goto MAIN_EXIT;
	}

	// Read line
	colCnt = AB_MAX_COL+1;
	rowCnt = 0;
	while(fgets(lineStr, 1024, fp) != NULL)
	{
		int lineCnt = 0, abCnt = 0;
		while ((lineStr[lineCnt]!='\n') && (lineCnt<1024) && (abCnt<AB_MAX_COL))
		{
			if (lineStr[lineCnt] != ' ')
			{
				matrixAB[rowCnt*AB_MAX_ROW+abCnt] = lineStr[lineCnt];
				abCnt++;
			}
			lineCnt++;			
		}

		// Select MIN col count
		if (abCnt < colCnt)
		{
			colCnt = abCnt;
		}

		rowCnt++;
	}

	// Failed to read
	if (colCnt == (AB_MAX_COL+1))
	{
		printf("Could not read matrixAB.txt\n");
		goto MAIN_EXIT;
	}

	// Set Row, Col count of matrixMN
	abRowCnt = rowCnt;
	abColCnt = colCnt;

	fclose(fp);
	fp = NULL;

	// Check A<M, B<N
	if ((abColCnt>mnColCnt) || (abRowCnt>mnRowCnt))
	{
		printf("matrixAB > matrixMN\n");
		goto MAIN_EXIT;
	}

	// Start the search
	memset(matrixDST, 0, MN_MAX_ROW*MN_MAX_COL);
	if (SearchMatrix(matrixMN, mnRowCnt, mnColCnt, matrixAB, abRowCnt, abColCnt, matrixDST) < 0)
	{
		printf("Search Matrix failed\n");
		goto MAIN_EXIT;
	}

	// Print the result
	res = 0;
	for (i=0; i<MN_MAX_ROW; i++)
	{
		for (j=0; j<MN_MAX_COL; j++)
		{
			if (matrixDST[i*MN_MAX_COL+j] != 0)
			{
				printf("Found at %d x %d\n", i, j);
				res = 1;
			}
		}
	}
	if (!res)
	{
		printf("No found\n");
	}

MAIN_EXIT:
	if (fp) fclose(fp);
	if (matrixMN) free(matrixMN);
	if (matrixAB) free(matrixAB);
	if (matrixDST) free(matrixDST);

	return 0;
}