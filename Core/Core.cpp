#include "Application.h"


std::shared_ptr<Venusaur::Application> app;

int main(int argc, char* argv[])
{
	app = std::make_shared<Venusaur::Application>();

	// Rendering.
	do
	{
		app->Update();
	} while (app->IsRunning());

	return 0;
}