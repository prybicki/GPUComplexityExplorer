#include <fmt/format.h>
#include <kernels.hpp>
#include <lib.hpp>

#include <Magnum/GL/DefaultFramebuffer.h>
#include <Magnum/Platform/GlfwApplication.h>

using namespace Magnum;

struct THCWindow : public Platform::Application
{
	explicit THCWindow(const Arguments& arguments)
	: Platform::Application{arguments}
	{
		setWindowCentered();
		setWindowTitle("Tiny Hadron Collider");

	}

private:
	void setWindowCentered()
	{
		const GLFWvidmode* videoMode = glfwGetVideoMode(glfwGetPrimaryMonitor());
		glfwSetWindowPos(window(), (videoMode->width - windowSize().x()) / 2, (videoMode->height - windowSize().y()) / 2);
	}

	void drawEvent() override
	{
		GL::defaultFramebuffer.clear(GL::FramebufferClear::Color);
		swapBuffers();
	}
};

int main(int argc, char** argv)
{
	runSync1D(64, 32, kHelloWorld, 42);
	cudaFuncAttributes attrs;
	CHECK_CUDA(cudaFuncGetAttributes(&attrs, reinterpret_cast<const void*>(kHelloWorld)));
	fmt::print("BinaryVersion={} PtxVersion={}\n", attrs.binaryVersion, attrs.ptxVersion);

	THCWindow app({argc, argv});
	return app.exec();
}
