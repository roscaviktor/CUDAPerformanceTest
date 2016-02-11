/***********************************************************************
 Author: Victor Rosca
 Date: 2016-02-10

                     CUDA. Testing Performance.
 
 This application generate an image using CUDA. 
 CUDA is a parallel computing platform and application programming interface (API) model created by NVIDIA.
 It allows software developers to use a CUDA-enabled graphics processing unit (GPU) for general purpose 
 processing – an approach known as GPGPU. The CUDA platform is a software layer that gives direct access 
 to the GPU's virtual instruction set and parallel computational elements.

 ************************************************************************/


#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <conio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

////////////////////////////////////////////////////////////////////
//#define IMAGE_TYPE_1
//#define IMAGE_TYPE_2
#define IMAGE_TYPE_3
//#define IMAGE_TYPE_4

#define IMAGE_WIDTH		5000
#define IMAGE_HEIGHT	5000
#define IMAGE_LEN		IMAGE_WIDTH * IMAGE_HEIGHT

// If is defined TESTING_PERFORMANCE then the program will generate a HTML file that will contain the result of testing. 
#define TESTING_PERFORMANCE

#ifdef TESTING_PERFORMANCE
#define BLOCK_STEP 10
#define INITIAL_BLOCKS 400
#define FINAL_BLOCKS IMAGE_LEN
#define MAX_IMAGES_GENERATED 2
#endif

cudaError_t generateImage(float *time, int calledNo);

#ifndef TESTING_PERFORMANCE
////////////////////////////////////////////////////////////////////
bool drawBMP(char *filename, char *blue, char *green, char *red) {
	unsigned int headers[13];
	FILE * outfile;
	int extrabytes;
	int paddedsize;
	int x; int y; int n;

	// How many bytes of padding to add to each
	extrabytes = 4 - ((IMAGE_WIDTH * 3) % 4);                
	// horizontal line - the size of which must
	// be a multiple of 4 bytes.
	if (extrabytes == 4)
		extrabytes = 0;

	paddedsize = ((IMAGE_WIDTH * 3) + extrabytes) * IMAGE_HEIGHT;

	// Headers...
	// Note that the "BM" identifier in bytes 0 and 1 is NOT included in these "headers".
	headers[0] = paddedsize + 54;			// bfSize (whole file size)
	headers[1] = 0;							// bfReserved (both)
	headers[2] = 54;						// bfOffbits
	headers[3] = 40;						// biSize
	headers[4] = IMAGE_WIDTH;				// biWidth
	headers[5] = IMAGE_HEIGHT;				// biHeight

	// Would have biPlanes and biBitCount in position 6, but they're shorts.
	// It's easier to write them out separately (see below) than pretend
	// they're a single int, especially with endian issues...
	headers[7] = 0;							// biCompression
	headers[8] = paddedsize;				// biSizeImage
	headers[9] = 0;							// biXPelsPerMeter
	headers[10] = 0;						// biYPelsPerMeter
	headers[11] = 0;						// biClrUsed
	headers[12] = 0;						// biClrImportant

	outfile = fopen(filename, "wb");

	// Headers begin...
	// When printing ints and shorts, we write out 1 character at a time to avoid endian issues.
	fprintf(outfile, "BM");
	for (n = 0; n <= 5; n++){
		fprintf(outfile, "%c", headers[n] & 0x000000FF);
		fprintf(outfile, "%c", (headers[n] & 0x0000FF00) >> 8);
		fprintf(outfile, "%c", (headers[n] & 0x00FF0000) >> 16);
		fprintf(outfile, "%c", (headers[n] & (unsigned int)0xFF000000) >> 24);
	}

	// These next 4 characters are for the biPlanes and biBitCount fields.
	fprintf(outfile, "%c", 1);
	fprintf(outfile, "%c", 0);
	fprintf(outfile, "%c", 24);
	fprintf(outfile, "%c", 0);

	for (n = 7; n <= 12; n++){
		fprintf(outfile, "%c", headers[n] & 0x000000FF);
		fprintf(outfile, "%c", (headers[n] & 0x0000FF00) >> 8);
		fprintf(outfile, "%c", (headers[n] & 0x00FF0000) >> 16);
		fprintf(outfile, "%c", (headers[n] & (unsigned int)0xFF000000) >> 24);
	}

	// Headers done, now write the data...
	int i;

	// BMP image format is written from bottom to top...
	for (y = IMAGE_HEIGHT - 1; y >= 0; y--){		
		for (x = 0; x <= IMAGE_WIDTH - 1; x++){
			i = y * x;			
			// Also, it's written in (b,g,r) format...
			fprintf(outfile, "%c", blue[i]);
			fprintf(outfile, "%c", green[i]);
			fprintf(outfile, "%c", red[i]);
		}

		if (extrabytes){
			// See above - BMP lines must be of lengths divisible by 4.
			for (n = 1; n <= extrabytes; n++){
				fprintf(outfile, "%c", 0);
			}
		}
	}
	fclose(outfile);
	return true;
}
#endif

////////////////////////////////////////////////////////////////////
char *time_stamp(){
	char *timestamp = (char *)malloc(sizeof(char)* 16);
	time_t ltime;
	ltime = time(NULL);
	struct tm *tm;
	tm = localtime(&ltime);

	sprintf(timestamp, "%04d%02d%02d%02d%02d%02d", tm->tm_year + 1900, tm->tm_mon,
		tm->tm_mday, tm->tm_hour, tm->tm_min, tm->tm_sec);
	return timestamp;
}



////////////////////////////////////////////////////////////////////
// This function is running in GPU.
__global__ void runKernel(char *blue, char *green, char *red, long blockSize, long width, long height)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
#ifdef IMAGE_TYPE_1
	blue[i] = (char)((255));
	green[i] = (char)((255));
	red[i] = (char)(((width*height) / i) % 256);
#endif
#ifdef IMAGE_TYPE_2
	blue[i] = (char)(width * height / i * 256) % 256;
	green[i] = (char)(width * height / i * 128) % 256;
	red[i] = (char)(width * height / i * 64) % 256;
#endif
#ifdef IMAGE_TYPE_3
	blue[i] = (char)((i * 256) / (width * height));
	green[i] = 64;
	red[i] = 64;
#endif
#ifdef IMAGE_TYPE_4
	long y = i / width;
	switch (threadIdx.x % 3){
	case 0:
		blue[i] = 255;
		green[i] = 0;
		red[i] = 0;
		//blue[i] = (char)(blockIdx.x * 256 / blockDim.x);
		//green[i] = (char)(blockIdx.x * 64 / blockDim.x);
		//red[i] = (char)(blockIdx.x * 64 / blockDim.x);
		break;
	case 1:
		blue[i] = 0;
		green[i] = 255;
		red[i] = 0;
		//blue[i] = (char)(blockIdx.x * 64 / blockDim.x);
		//green[i] = (char)(blockIdx.x * 256 / blockDim.x);
		//red[i] = (char)(blockIdx.x * 64 / blockDim.x);
		break;
	case 2:
		blue[i] = 0;
		green[i] = 0;
		red[i] = 255;
		//blue[i] = (char)(blockIdx.x * 64 / blockDim.x);
		//green[i] = (char)(blockIdx.x * 64 / blockDim.x);
		//red[i] = (char)(blockIdx.x * 256 / blockDim.x);
		break;
	}
#endif
}

////////////////////////////////////////////////////////////////////
// Helper function for using CUDA to add vectors in parallel.
cudaError_t generateImage(float *time, int calledNo)
{
	char *dev_blue = 0;
	char *dev_green = 0;
	char *dev_red = 0;
    cudaError_t cudaStatus;

	char *blue = nullptr;
	char *green = nullptr;
	char *red = nullptr;

	blue = new char[IMAGE_LEN];
	green = new char[IMAGE_LEN];
	red = new char[IMAGE_LEN];

	//long start;
	cudaEvent_t start, stop;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_blue, IMAGE_LEN * sizeof(char));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

	cudaStatus = cudaMalloc((void**)&dev_green, IMAGE_LEN * sizeof(char));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

	cudaStatus = cudaMalloc((void**)&dev_red, IMAGE_LEN * sizeof(char));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Launch a kernel.
	long blockSize = IMAGE_LEN / calledNo;

	cudaEventRecord(start);
	runKernel << <blockSize, calledNo >> >(dev_blue, dev_green, dev_red, blockSize, IMAGE_WIDTH, IMAGE_HEIGHT);
	cudaEventRecord(stop);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "runKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching runKernel!\n", cudaStatus);
        goto Error;
    }

	cudaEventSynchronize(stop);
	cudaEventElapsedTime(time, start, stop);

    // Deploy from GPU buffer to host memory.
#ifndef TESTING_PERFORMANCE
	cudaStatus = cudaMemcpy(blue, dev_blue, IMAGE_LEN * sizeof(char), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!"); goto Error;
	}
	cudaStatus = cudaMemcpy(green, dev_green, IMAGE_LEN * sizeof(char), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!"); goto Error;
	}
	cudaStatus = cudaMemcpy(red, dev_red, IMAGE_LEN * sizeof(char), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!"); goto Error;
    }

	// Create a BMP file.
	char fileStr[25];
	sprintf(fileStr, "%s.bmp", time_stamp());
	if (drawBMP(fileStr, blue, green, red)){
		printf("Image was printed!\n");
	}
#endif

Error:
	cudaFree(dev_blue);
	cudaFree(dev_green);
	cudaFree(dev_red);
	delete blue;
	delete green;
	delete red;
    
    return cudaStatus;
}


////////////////////////////////////////////////////////////////////
int main()
{
	float time = 0.0;

#ifdef TESTING_PERFORMANCE
	int blocks = INITIAL_BLOCKS;
	FILE *file;
	cudaError_t cudaStatus;
	char fileStr[25];
	sprintf(fileStr, "%s.html", time_stamp());
	file = fopen(fileStr, "wb");
	fprintf(file, "<html>\n \
	<head>\n \
		<title>CUDA - Testing Performance, Victor Rosca</title>\n \
		<script type = \"text/javascript\" src = \"https://www.gstatic.com/charts/loader.js\"></script>\n \
		<script type = \"text/javascript\">\n \
		google.charts.load('current', { 'packages':['corechart'] });\n \
	google.charts.setOnLoadCallback(drawChart);\n \
	\n\
	function drawChart() { \n\
		var data = google.visualization.arrayToDataTable([ \n\
			['Time (msec)', 'Generate %d image(s).'], \n\
				", MAX_IMAGES_GENERATED);
	printf("Testing CUDA. \n\n", blocks, time);
	printf("Image size: ~ %lu Mb\n\n", (IMAGE_LEN * 3 / 1024 / 1024));
	printf("-------------------------------------------\n");
	printf("|  Blocks of threads |     Time (msec)    |\n");
	printf("|-----------------------------------------|\n");
	while (blocks < FINAL_BLOCKS){
		long blockSize = IMAGE_LEN / blocks;
		int i = 0;
		bool stop = false;
		float totalTime = 0.0;
		while (i++ < MAX_IMAGES_GENERATED){
			cudaStatus = generateImage(&time, blocks);
			totalTime += time;
			if (cudaStatus != cudaSuccess){
				stop = true;
				break;
			}
		}
		if (stop)
			break;
		fprintf(file, "['<<<%d, %d>>>', %f],\n", blocks, blockSize, totalTime);
		printf("|%*d|%*f|\n", 20, blocks, 20, totalTime);
		blocks += BLOCK_STEP;
	}
	printf("-------------------------------------------\n");

	fprintf(file, "]); \n\
		var options = { \n\
		title: 'CUDA Performance, file: %s, author: Victor Rosca', \n\
		   curveType : 'function', \n\
				   legend : { position: 'bottom' } \n\
		}; \n\
		var chart = new google.visualization.LineChart(document.getElementById('curve_chart')); \n\
		chart.draw(data, options); \n\
	} \n\
	</script> \n\
		</head> \n\
		<body> \n\
			<p>Image size: ~ %lu Mb, </br> \n\
			<p>Initial blocks of threads = %d</br> \n\
			Final blocks of threads = %d</p> \n\
		<div id = \"curve_chart\" style = \"width: 1200px; height: 700px\"></div> \n\
		</body> \n\
	</html>", fileStr, (long)(IMAGE_LEN * 3 / 1024 / 1024), (int)INITIAL_BLOCKS, blocks);
	
	fclose(file);

	if (cudaStatus != cudaSuccess){
		fprintf(stderr, "generateImage failed!");
		printf("END. Press any key.\n");
		getch();
		return 1;
	}
#else
	cudaError_t cudaStatus = generateImage(&time, 1000);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "generateImage failed!");
		printf("END. Press any key.\n");
		getch();
		return 1;
	}
#endif

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		printf("END. Press any key.\n");
		getch();
		return 1;
	}
	printf("END. Press any key.\n");
	getch();
	return 0;
}
