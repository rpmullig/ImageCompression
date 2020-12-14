#include <cuda.h>
#include <iostream>
#include <map>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <vector>
#include <sstream>
#include <iterator>
#include <utility>
#include <bitset> 
#include "lodepng.h"
#include "lodepng.cu"
using namespace std;

class Pixel {
public:
	unsigned char r; 
	unsigned char g; 
	unsigned char b;
	unsigned char a; 
	Pixel(unsigned char r, unsigned char g, unsigned char b, unsigned char a){
		this->r = r; 
		this->g = g; 
		this->b = b; 
		this->a = a; 
	}
	
} ;

__host__ void encodeWithState(const char* filename, std::vector<unsigned char>& image, unsigned width, unsigned height); 
__global__ void decompress(int n, char * data, unsigned char * output, int height, int width, 
							unsigned char * red_map, unsigned char * green_map, unsigned char * blue_map, unsigned char * alpha_map); 


int main(int argc, char **argv)
{
	/*
	Pass important pieces of data to the cuda file
	Importantly, the huffman map data structure was saved in a text file
	*/
	const char* mapFilename = argv[1]; 
	const char* binaryFilename = argv[2]; 
	const int imageHeight = stoi(argv[3]); 
	const int imageWidth = stoi(argv[4]); 
	const int codeBitLength = stoi(argv[5]); 
	const int N = stoi(argv[6]); 
	const char* outputFilename = argv[7]; 


	/*
	The huffman map data structure was saved with the key and the RGBA values
	spereated all by new lines for lazy parsing. 
	*/	
	string line;
	ifstream mapFile; 
	mapFile.open(mapFilename, ios::binary | ios::in);
	std::vector<std::string> elements; 

	if (mapFile.is_open()){
		while ( getline (mapFile, line) ){
			elements.emplace_back(line); 
		}
		mapFile.close();
	}


	/*
	Editing the map
	
	In CUDA GPU programming, there are no strings; which is how the codes were 
	represented at times in Go and in the text fiel for transfer and conveincne. Here
	below a codemap will contain an integer value and a custom struct Pixel to house the data. 
	
	The data has been standardized to 24 bits, because dividing up the task for the GPU
	requires a standard format. If there was an offset, then the task would be more complex to index
	and locate the propert data alignment. 
	*/
	map<int, Pixel> codeMap; 
	
	int mapSize = 0;
	int maxMapElement = 0; 
	int minMapElement = 16777216; // sentitenl value with max 2^24  

	for (int i = 0; i < elements.size(); i += 5){	
		Pixel newPixel((unsigned char)stoi(elements[i+1]), (unsigned char)stoi(elements[i+2]), (unsigned char)stoi(elements[i+3]), (unsigned char)stoi(elements[i+4]));  			
		unsigned long bitNum = std::bitset<24>(elements[i]).to_ulong(); // bitset will convert a bitstring into a number 
		int intNum = (int)bitNum; // cast the above to an int 
		codeMap.insert(pair<int, Pixel>(intNum, newPixel)); // had to conver the bit string to an in, with no strings in cuda
		
		if (maxMapElement > intNum) { maxMapElement = intNum; }
		if (minMapElement < intNum) { minMapElement = intNum; }
		++mapSize; 
	}
	
	/* Since C++ does not allow high level data structures, have to convert a map
	 to an array for lookup on the key index. By using the range of the bit there can be an array of pixel values that 
	 can be indexed by the integer reprsentation code. 

	Note: the redundancy with the map was done because this was an after the fact fix. Technically, 
	this could replace the codeMap entierly--but used for conveince.  
	*/
	int keyRange = maxMapElement - minMapElement + 1; 
	
	unsigned char * red_map = (unsigned char*)malloc(keyRange * sizeof(unsigned char)); 
	unsigned char * green_map = (unsigned char*)malloc(keyRange * sizeof(unsigned char));
	unsigned char * blue_map = (unsigned char*)malloc(keyRange * sizeof(unsigned char));
	unsigned char * alpha_map = (unsigned char*)malloc(keyRange * sizeof(unsigned char));
	for (const auto& kv : codeMap) {
		red_map[kv.first - minMapElement] = kv.second.r;
		green_map[kv.first - minMapElement] = kv.second.g;
		blue_map[kv.first - minMapElement] = kv.second.b;
		alpha_map[kv.first - minMapElement] = kv.second.a;
	}
	
	/*
	Read in the compressed binary file

	source: http://www.cplusplus.com/doc/tutorial/files/
	*/
	streampos size;
	char * data;

	ifstream binFile (binaryFilename, ios::in|ios::binary|ios::ate);
	if (binFile.is_open()){
		size = binFile.tellg();
		data = new char [size];
		binFile.seekg (0, ios::beg);
		binFile.read (data, size);
		binFile.close();
	}

	/*
	Create an output for the GPU to decompress the data to 
	*/
	unsigned char * output; 
	int pngSize = imageWidth * imageHeight * 4; // 4 bytes for RGBA 
	output = (unsigned char*)malloc(pngSize * sizeof(unsigned char));
	
	// cuda malloc the output for the GPU's to write to 
	// device output
	unsigned char * d_output; 
	cudaMalloc((void**) &d_output, pngSize * sizeof(unsigned char)); 
	
	char * d_data; 
	cudaMalloc((void**) &d_data, size * sizeof(char));; // see binaryfile read that created size variable
	cudaMemcpy((char *) &d_data, data, keyRange * sizeof(unsigned char), cudaMemcpyHostToDevice);
	
	/* copy over the maps to the device/GPU to use 
		 1. declare device maps them on the host with the d_* prefix convention
		 2. cudaMalloc/initiallize them using cudaMalloc
		 3. copy over the data from host to cuda with cudaMemcpy
		 4. pass the device maps to the host
		 5. free the memory on cuda 
	*/
	unsigned char * d_red_map;  
	unsigned char * d_green_map; 
	unsigned char * d_blue_map;  
	unsigned char * d_alpha_map;
	
	cudaMalloc((void**) &d_red_map, keyRange * sizeof(unsigned char));	
	cudaMalloc((void**) &d_green_map, keyRange * sizeof(unsigned char));
	cudaMalloc((void**) &d_blue_map,  keyRange * sizeof(unsigned char));
	cudaMalloc((void**) &d_alpha_map, keyRange * sizeof(unsigned char));
	
	cudaMemcpy((unsigned char *) &d_red_map, red_map, keyRange * sizeof(unsigned char), cudaMemcpyHostToDevice);	
	cudaMemcpy((unsigned char *) &d_green_map, green_map, keyRange * sizeof(unsigned char), cudaMemcpyHostToDevice);
	cudaMemcpy((unsigned char *) &d_blue_map, blue_map, keyRange * sizeof(unsigned char), cudaMemcpyHostToDevice);
	cudaMemcpy((unsigned char *) &d_alpha_map, alpha_map, keyRange * sizeof(unsigned char), cudaMemcpyHostToDevice);
	
	// run the GPU program with N blocks with N threads each
	// will split up the image and then operate on the images with a given index 
	decompress<<<N, N>>>(N, d_data, d_output, imageHeight, imageWidth, d_red_map, d_green_map, d_blue_map, d_alpha_map); 
	
	// copy the data to the cpu/host to the output for the host to export this data to a png file
	cudaMemcpy(d_output, output, pngSize * sizeof(unsigned char), cudaMemcpyDeviceToHost);
	
	cudaFree(d_output);
	cudaFree(d_data);
	cudaFree(d_red_map);
	cudaFree(d_green_map);
	cudaFree(d_blue_map);
	cudaFree(d_alpha_map);
	
	/*
	No STL objects are allowed inside kernels, so I must convert the output to a vector 
	in order to get the image in a png format through the lodepng 3rd party library imported
	Modified usage from the lodepng. file https://github.com/lvandeve/lodepng/blob/master/examples/example_encode.cpp
	
	The encodeWithState was modified from the original author as well.
	
	The following code will convert the GPU output of reading the compressed data and put into 
	a vector for the PNG export. 
	*/

	/*
	To confirm tha the png works, you can comment out the cuda section and past ethe following. 
	This proves that the png works. Can fill the output array with data for a png output file
	for(int i = 0; i < pngSize; i++){
		output[i] = i % 255; 
	}
	*/

	unsigned width = (unsigned)imageWidth, height = (unsigned)imageHeight;
	std::vector<unsigned char> image;
	image.resize(width * height * 4);
	for(unsigned y = 0; y < height; y++) {
		for(unsigned x = 0; x < width; x++) {
			image[4 * width * y + 4 * x + 0] = output[imageWidth * y + (5 * x) + 0];
			image[4 * width * y + 4 * x + 1] = output[imageWidth * y + (5 * x) + 1];   
			image[4 * width * y + 4 * x + 2] = output[imageWidth * y + (5 * x) + 2]; 
			image[4 * width * y + 4 * x + 3] = output[imageWidth * y + (5 * x) + 3]; 
		}
	}

	encodeWithState(outputFilename, image, width, height); // exports the image 

	// free up memory
	free(output);
	free(red_map); 
	free(green_map); 
	free(blue_map); 
	free(alpha_map);
	image.clear();

	return 0; 
}

/*
Note this function is global, so that the host can call it. Any subfunctions stemming from this would be __device__ 
and work only within the GPU, but that was not needed

Every input data is an array data structure, and the char is just a byte-sized primitave. 

The indexing comes from the cude kernel itself, which will get the location relative to the image. Then, each thread within the block 
will do a certain amount of work (represneted by the while loop) and populate the output with the uncoding of the data from the 
data array as the key.

In the parellel verison: this essentially will converts 24 bits to 32 bits--since there had to be alignment. 
*/
__global__ void decompress(int n, char * data, unsigned char * output, int height, int width, 
							unsigned char * red_map, unsigned char * green_map, unsigned char * blue_map, unsigned char * alpha_map){

	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int i = 0; 
	while (i <  (int)((height * width) / (n*n)) && index < (height * width)){ // a certain amount of work for each worker. 
		int lookupIndex = data[index + i]; 
		output[index + (5*i) + 0] = red_map[lookupIndex]; 
		output[index + (5*i) + 1] = red_map[lookupIndex]; 
		output[index + (5*i) + 2] = red_map[lookupIndex]; 
		output[index + (5*i) + 3] = red_map[lookupIndex]; 
		i += blockDim.x;  // iterate by the number of threads in a block.  
	}
	__syncthreads(); // wait for all threads to finish. Not needed for calculations but acts like a waiting group
	return; 
}

// modified from the lodepng library examples
__host__ void  encodeWithState(const char* filename, std::vector<unsigned char>& image, unsigned width, unsigned height) {
	std::vector<unsigned char> png;

	// we're going to encode with a state rather than a convenient function, because enforcing a color type requires setting options
	lodepng::State state;
	// input color type
	state.info_raw.colortype = LCT_RGBA;
	state.info_raw.bitdepth = 8;
	// output color type
	state.info_png.color.colortype = LCT_RGBA;
	state.info_png.color.bitdepth = 8;
	state.encoder.auto_convert = 0; // without this, it would ignore the output color type specified above and choose an optimal one instead

  
	unsigned error = lodepng::encode(png, image, width, height, state);
	if(!error) lodepng::save_file(png, filename);

	//if there's an error, display it
	if(error) std::cout << "encoder error " << error << ": "<< lodepng_error_text(error) << std::endl;
}