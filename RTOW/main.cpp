#include <fstream>

#include <application.h>

#include "renderer_normal.h"

int main(int argc, char* argv[])
{
	auto aspect_ratio = 16.0 / 9.0;
	int image_width = 400;

	// Calculate the image height, and ensure that it's at least 1.
	int image_height = int(image_width / aspect_ratio);
	image_height = (image_height < 1) ? 1 : image_height;

	auto app = std::make_unique<Venusaur::Application>(image_width, image_height);
	
	std::ifstream file{"normal.optixir", std::ios::binary};
	std::vector<char> optixIR(std::istreambuf_iterator<char>(file), {});
	file.close();
	
	app->SetRenderer(std::make_shared<RayTracingInOneWeekend::RendererNormal>(app->GetOutputBuffer(), optixIR));

	while(app->IsRunning())
	{
		app->Update();
    }

	return 0;
}