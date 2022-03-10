#pragma once


#include <vector_types.h>
#include <vector>

void Init();
 
uchar4* Launch(int width, int height);

void Cleanup();

void Trace(int width, int height, std::vector<unsigned char>& red, std::vector<unsigned char>& green, std::vector<unsigned char>& blue);