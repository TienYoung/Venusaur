#include <nvjpeg.h>
#include <vector>
#include <fstream>

#include "RayTracer.h"

#include "Exception.h"


int         width = 256;
int         height = 256;


int main(int argc, char* argv[])
{

	std::vector<unsigned char> red;
	std::vector<unsigned char> green;
	std::vector<unsigned char> blue;

	Trace(width, height, red, green, blue);


	nvjpegHandle_t nv_handle;
	nvjpegEncoderState_t nv_enc_state;
	nvjpegEncoderParams_t nv_enc_params;
	cudaStream_t stream = nullptr;

	//CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
	CUDA_CHECK(cudaStreamSynchronize(stream));

	// initialize nvjpeg structures
	CHECK_NVJPEG(nvjpegCreateSimple(&nv_handle));
	CHECK_NVJPEG(nvjpegEncoderStateCreate(nv_handle, &nv_enc_state, stream));
	CHECK_NVJPEG(nvjpegEncoderParamsCreate(nv_handle, &nv_enc_params, stream));
	CHECK_NVJPEG(nvjpegEncoderParamsSetEncoding(nv_enc_params, NVJPEG_ENCODING_BASELINE_DCT, stream));
	CHECK_NVJPEG(nvjpegEncoderParamsSetQuality(nv_enc_params, 70, stream));
	CHECK_NVJPEG(nvjpegEncoderParamsSetOptimizedHuffman(nv_enc_params, 0, stream));
	CHECK_NVJPEG(nvjpegEncoderParamsSetSamplingFactors(nv_enc_params, NVJPEG_CSS_444, stream));


	nvjpegImage_t nv_image;
	// Fill nv_image with image data, let¡¯s say 640x480 image in RGB format
	nv_image.channel[0] = red.data();
	nv_image.channel[1] = green.data();
	nv_image.channel[2] = blue.data();
	//nv_image.channel[3] = &data->w;
	nv_image.pitch[0] = width;
	nv_image.pitch[1] = width;
	nv_image.pitch[2] = width;
	//nv_image.pitch[3] = sizeof(uchar4) * width;


	// Compress image
	CHECK_NVJPEG(nvjpegEncodeImage(nv_handle, nv_enc_state, nv_enc_params, &nv_image, NVJPEG_INPUT_RGB, width, height, stream));


	// get compressed stream size
	size_t length;
	CHECK_NVJPEG(nvjpegEncodeRetrieveBitstream(nv_handle, nv_enc_state, NULL, &length, stream));
	// get stream itself
	CUDA_CHECK(cudaStreamSynchronize(stream));
	std::vector<char> jpeg(length);
	CHECK_NVJPEG(nvjpegEncodeRetrieveBitstream(nv_handle, nv_enc_state, (unsigned char*)jpeg.data(), &length, 0));

	// write stream to file
	CUDA_CHECK(cudaStreamSynchronize(stream));
	std::ofstream output_file("test.jpg", std::ios::out | std::ios::binary);
	output_file.write(jpeg.data(), length);
	output_file.close();



	return 0;
}