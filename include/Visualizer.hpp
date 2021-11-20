#pragma once

#include <cassert>
#include <queue>
#include <functional>

#include <MagnumPlugins/AnyImageImporter/AnyImageImporter.h>
#include <Magnum/GL/DefaultFramebuffer.h>
#include <Magnum/GL/Mesh.h>
#include <Magnum/GL/Renderer.h>
#include <Magnum/Math/Angle.h>
#include <Magnum/Math/Color.h>
#include <Magnum/Math/Matrix3.h>
#include <Magnum/MeshTools/Compile.h>
#include <Magnum/PixelFormat.h>
#include <Magnum/Platform/GlfwApplication.h>
#include <Magnum/Primitives/Circle.h>
#include <Magnum/Shaders/FlatGL.h>
#include <Magnum/Trade/ImageData.h>
#include <Magnum/Trade/MeshData.h>

#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>

#include <internal/aliases.hpp>
#include <internal/macros.hpp>
#include <cuda/kernels.hpp>
#include <Vector.hpp>
#include <Matrix.hpp>
#include <ComputeManager.hpp>

using namespace Magnum;

struct Visualizer : public Platform::Application
{
private:
	GL::Mesh circle;
	Shaders::FlatGL2D shader;
	std::queue<std::function<void()>> drawQueue;

	GL::Buffer colorBuffer;
	GL::Buffer transformBuffer;
	cudaGraphicsResource_t colorResource;
	cudaGraphicsResource_t transformResource;

public:
	explicit Visualizer(const Arguments& arguments)
	: Platform::Application{arguments, makeWindowConfig(), makeOpenGLConfig()}
	, circle(MeshTools::compile(Primitives::circle2DSolid(8)))
	, shader(Shaders::FlatGL2D::Flag::InstancedTransformation | Shaders::FlatGL2D::Flag::VertexColor)
	, colorBuffer()
	, transformBuffer()
	, colorResource(nullptr)
	, transformResource(nullptr)
	{
		const GLFWvidmode* videoMode = glfwGetVideoMode(glfwGetPrimaryMonitor());
		glfwSetWindowPos(window(), (videoMode->width - windowSize().x()) / 2, (videoMode->height - windowSize().y()) / 2);

		GL::Renderer::enable(GL::Renderer::Feature::Blending);
		GL::Renderer::setBlendFunction(
				GL::Renderer::BlendFunction::SourceAlpha, /* or SourceAlpha for non-premultiplied */
				GL::Renderer::BlendFunction::DestinationAlpha);
	}

	~Visualizer()
	{
		if (transformResource != nullptr) {
			CHECK_CUDA_NO_THROW(cudaGraphicsUnregisterResource(colorResource));
			CHECK_CUDA_NO_THROW(cudaGraphicsUnregisterResource(transformResource));
		}
	}

	void renderParticles(count_t count, Vec2f* dPosition, float* dRadius, Vec4f* dColor)
	{
		std::function drawLambda = [=](){
			if (transformBuffer.size() / sizeof(Mat3x3f) < static_cast<unsigned>(count)) {
				if (transformResource != nullptr) {
					CHECK_CUDA(cudaGraphicsUnregisterResource(colorResource));
					CHECK_CUDA(cudaGraphicsUnregisterResource(transformResource));
				}
				transformBuffer.setStorage(sizeof(Mat3x3f) * count, GL::Buffer::StorageFlags{});
				colorBuffer.setStorage(sizeof(Vec4f) * count, GL::Buffer::StorageFlags{});
				CHECK_CUDA(cudaGraphicsGLRegisterBuffer(&colorResource, colorBuffer.id(), cudaGraphicsRegisterFlagsWriteDiscard));
				CHECK_CUDA(cudaGraphicsGLRegisterBuffer(&transformResource, transformBuffer.id(), cudaGraphicsRegisterFlagsWriteDiscard));
			}
			size_t dColorSize = 0;
			size_t dTransformSize = 0;
			Vec4f* colorBufferPtr = nullptr;
			Mat3x3f* transformBufferPtr = nullptr;

			CHECK_CUDA(cudaGraphicsMapResources(1, &colorResource));
			CHECK_CUDA(cudaGraphicsMapResources(1, &transformResource));
			CHECK_CUDA(cudaGraphicsResourceGetMappedPointer((void**) &colorBufferPtr, &dColorSize, colorResource));
			CHECK_CUDA(cudaGraphicsResourceGetMappedPointer((void**) &transformBufferPtr, &dTransformSize, transformResource));
			assert(dColorSize == sizeof(Vec4f) * count);
			assert(dTransformSize == sizeof(Mat3x3f) * count);

			CHECK_CUDA(cudaMemcpy(colorBufferPtr, dColor, sizeof(Vec4f) * count, cudaMemcpyDeviceToDevice));
			cm.runSync1D(count, 256, kPos2DToTransform3x3, count, dPosition, dRadius, transformBufferPtr);

			CHECK_CUDA(cudaGraphicsUnmapResources(1, &transformResource));
			CHECK_CUDA(cudaGraphicsUnmapResources(1, &colorResource));

			circle.addVertexBufferInstanced(transformBuffer, 1, 0,Shaders::FlatGL2D::TransformationMatrix {});
			circle.addVertexBufferInstanced(colorBuffer, 1, 0,Shaders::FlatGL2D::Color4 {});
			circle.setInstanceCount(count);
			shader.draw(circle);
		};
		drawQueue.push(drawLambda);
	}

private:
	void drawEvent() override
	{
		GL::defaultFramebuffer.clear(GL::FramebufferClear::Color);
		while (!drawQueue.empty()) {
			std::invoke(drawQueue.front());
			drawQueue.pop();
		}
		swapBuffers();
	}

	Configuration makeWindowConfig()
	{
		Configuration cfg;
		cfg.setSize({1024, 1024});
		cfg.setTitle("Tiny Hadron Collider");
		return cfg;
	}

	GLConfiguration makeOpenGLConfig()
	{
		GLConfiguration cfg;
		return cfg;
	}
};