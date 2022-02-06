#pragma once

#include <cassert>
#include <functional>
#include <set>
#include <queue>

#include <MagnumPlugins/AnyImageImporter/AnyImageImporter.h>
#include <Magnum/DebugTools/FrameProfiler.h>
#include <Magnum/GL/DefaultFramebuffer.h>
#include <Magnum/GL/Mesh.h>
#include <Magnum/GL/Renderer.h>
#include <Magnum/GL/TextureFormat.h>
#include <Magnum/Math/Angle.h>
#include <Magnum/Math/Color.h>
#include <Magnum/Math/Matrix3.h>
#include <Magnum/Math/Matrix4.h>
#include <Magnum/MeshTools/Compile.h>
#include <Magnum/PixelFormat.h>
#include <Magnum/Platform/GlfwApplication.h>
#include <Magnum/Primitives/Circle.h>
#include <Magnum/SceneGraph/Object.h>
#include <Magnum/SceneGraph/Camera.h>
#include <Magnum/SceneGraph/MatrixTransformation2D.hpp>
#include <Magnum/Shaders/FlatGL.h>
#include <Magnum/Trade/ImageData.h>
#include <Magnum/Trade/MeshData.h>
#include <Magnum/ImGuiIntegration/Context.hpp>


#include <implot.h>

#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>

#include <internal/aliases.hpp>
#include <internal/macros.hpp>
#include <cuda/kernels.hpp>
#include <Vector.hpp>
#include <Matrix.hpp>
#include <ComputeManager.hpp>
#include <Corrade/Utility/Arguments.h>
#include <Magnum/GL/RectangleTexture.h>
#include <Field2DShader.hpp>


using namespace Magnum;

struct Visualizer : public Platform::Application
{
	// TODO: Add documentation (units!)
	constexpr static float INITIAL_RENDER_DISTANCE = 16.0f;
	constexpr static Vector2 INITAL_CAMERA_POSITION = {2048.f, 2048.f};
	constexpr static float CAMERA_PAN_SPEED = 0.016f;
	constexpr static float CAMERA_ZOOM_SPEED = 0.032f;

private:
	Utility::Arguments cliArgs;
	GL::Mesh circle;
	Shaders::FlatGL2D flatShader;
	Field2DShader field2DShader;
	std::queue<std::function<void()>> drawQueue;

	std::set<KeyEvent::Key> pressedKeys;
	SceneGraph::Object<SceneGraph::MatrixTransformation2D> cameraObject;
	SceneGraph::Camera2D camera;
	float currentZoom = 1.0f;

	GL::Buffer colorBuffer;
	GL::Buffer transformBuffer;
	cudaGraphicsResource_t colorResource;
	cudaGraphicsResource_t transformResource;

	std::optional<std::function<void()>> userGUI;
	ImGuiIntegration::Context _imgui{NoCreate};
	ImPlotContext* implotCtx;
	DebugTools::FrameProfilerGL profiler { DebugTools::FrameProfilerGL::Value::FrameTime, 10};

public:
	Visualizer(int argc, char** argv) : Visualizer({argc, argv}, makeCLIArgs({argc, argv})) {}

	Visualizer(const Arguments& args, const Utility::Arguments& cliArgs)
	: Platform::Application{args, makeWindowConfig(cliArgs), makeOpenGLConfig()}
	, circle(MeshTools::compile(Primitives::circle2DSolid(6)))
	, flatShader(Shaders::FlatGL2D::Flag::InstancedTransformation | Shaders::FlatGL2D::Flag::VertexColor)
	, field2DShader()
	, camera(cameraObject)
	, colorBuffer()
	, transformBuffer()
	, colorResource(nullptr)
	, transformResource(nullptr)
	{
		const GLFWvidmode* videoMode = glfwGetVideoMode(glfwGetPrimaryMonitor());
		glfwSetWindowPos(window(), (videoMode->width - windowSize().x()) / 2, (videoMode->height - windowSize().y()) / 2);

		// GL::Renderer::setBlendFunction(
		// 		GL::Renderer::BlendFunction::SourceAlpha,
		// 		GL::Renderer::BlendFunction::DestinationAlpha);
		currentZoom = INITIAL_RENDER_DISTANCE / std::min(windowSize().x(), windowSize().y());
		cameraObject.translate(INITAL_CAMERA_POSITION);

		_imgui = ImGuiIntegration::Context(Vector2{windowSize()} / dpiScaling(), windowSize(), framebufferSize());
		implotCtx = ImPlot::CreateContext();

		// Critical for GUI!
		GL::Renderer::enable(GL::Renderer::Feature::Blending);
		GL::Renderer::setBlendEquation(GL::Renderer::BlendEquation::Add,GL::Renderer::BlendEquation::Add);
		GL::Renderer::setBlendFunction(GL::Renderer::BlendFunction::SourceAlpha, GL::Renderer::BlendFunction::OneMinusSourceAlpha);

		profiler.beginFrame();
	}

	void setUserGUI(std::function<void()> userGUI)
	{
		this->userGUI = userGUI;
	}

	static Utility::Arguments makeCLIArgs(const Arguments& args)
	{
		Utility::Arguments cliArgs;
		// TODO: add options:
		// - Sample count
		// - Particle segment count
		cliArgs.addBooleanOption('f', "fullscreen");
		cliArgs.addSkippedPrefix("magnum", "Magnum options");
		cliArgs.parse(args.argc, args.argv);
		return cliArgs;
	}

	~Visualizer()
	{
		ImPlot::DestroyContext(implotCtx);
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

			flatShader.setTransformationProjectionMatrix(camera.projectionMatrix() * camera.cameraMatrix());
			flatShader.draw(circle);
		};
		drawQueue.push(drawLambda);
	}

	// devBytes should be 1 byte per pixel
	void renderTexture(float posX, float posY, int sizeX, int sizeY, void* devBytes)
	{
		std::function drawLambda = [=](){
			GL::RectangleTexture texture;
			cudaGraphicsResource_t resource;
			cudaArray_t array;

			texture.setStorage(GL::TextureFormat::R8, {sizeX, sizeY});
			texture.setMinificationFilter(GL::SamplerFilter::Nearest);
			texture.setMagnificationFilter(GL::SamplerFilter::Nearest);
			CHECK_CUDA(cudaGraphicsGLRegisterImage(&resource,
			                                       texture.id(),
			                                       GL_TEXTURE_RECTANGLE,
			                                       cudaGraphicsRegisterFlagsWriteDiscard));
			CHECK_CUDA(cudaGraphicsMapResources(1, &resource));
			CHECK_CUDA(cudaGraphicsSubResourceGetMappedArray(&array, resource, 0, 0));
			CHECK_CUDA(cudaMemcpy2DToArray(array, 0, 0, devBytes, sizeX, sizeX, sizeY, cudaMemcpyDeviceToDevice));
			CHECK_CUDA(cudaGraphicsUnmapResources(1, &resource));
			CHECK_CUDA(cudaGraphicsUnregisterResource(resource));

			field2DShader.setTransformationProjectionMatrix(camera.projectionMatrix() * camera.cameraMatrix());
			field2DShader.drawTexture(texture, posX, posY);
		};
		drawQueue.push(drawLambda);
	}
private:

	void viewportEvent(ViewportEvent& event) override
	{
		// fmt::print("windowSize=({}, {}) framebufferSize=({}, {}) dpiScale=({}, {})\n",
		// 	event.windowSize().x(), event.windowSize().y(),
		// 	event.framebufferSize().x(), event.framebufferSize().y(),
		// 	event.dpiScaling().x(), event.dpiScaling().y());
		GL::defaultFramebuffer.setViewport({{}, event.framebufferSize()});
		updateProjectionMatrix();

		_imgui.relayout(Vector2{event.windowSize()} / event.dpiScaling(), event.windowSize(), event.framebufferSize());
	}

	void keyPressEvent(KeyEvent& event) override
	{
		if(_imgui.handleKeyPressEvent(event)) {
			return;
		}
		pressedKeys.insert(event.key());
	}

	void keyReleaseEvent(KeyEvent& event) override
	{
		if(_imgui.handleKeyReleaseEvent(event)) {
			return;
		}
		pressedKeys.erase(event.key());
	}

	void mousePressEvent(MouseEvent& event) override {
		if(_imgui.handleMousePressEvent(event)) return;
	}

	void mouseReleaseEvent(MouseEvent& event) override {
		if(_imgui.handleMouseReleaseEvent(event)) return;
	}

	void mouseMoveEvent(MouseMoveEvent& event) override {
		if(_imgui.handleMouseMoveEvent(event)) return;
	}

	void mouseScrollEvent(MouseScrollEvent& event) override {
		if(_imgui.handleMouseScrollEvent(event)) {
			/* Prevent scrolling the page */
			event.setAccepted();
			return;
		}
	}
	void handleKeyboard()
	{
		float cameraPanSpeed = CAMERA_PAN_SPEED * std::min(camera.projectionSize().x(), camera.projectionSize().y());
		if (pressedKeys.contains(KeyEvent::Key::A)) {
			cameraObject.translate({-cameraPanSpeed, 0});
		}
		if (pressedKeys.contains(KeyEvent::Key::D)) {
			cameraObject.translate({+cameraPanSpeed, 0});
		}
		if (pressedKeys.contains(KeyEvent::Key::W)) {
			cameraObject.translate({0, +cameraPanSpeed});
		}
		if (pressedKeys.contains(KeyEvent::Key::S)) {
			cameraObject.translate({0, -cameraPanSpeed});
		}
		if (pressedKeys.contains(KeyEvent::Key::Q)) {
			currentZoom *= (1+CAMERA_ZOOM_SPEED);
		}
		if (pressedKeys.contains(KeyEvent::Key::E)) {
			currentZoom *= (1-CAMERA_ZOOM_SPEED);
		}
		if (pressedKeys.contains(KeyEvent::Key::Esc)) {
			this->exit(0);
		}
		updateProjectionMatrix();
	};

	 void updateProjectionMatrix()
	 {
		auto projectionMatrix = Matrix3::projection(Vector2(windowSize()) * currentZoom);
		camera.setProjectionMatrix(projectionMatrix);
	 }

	void drawEvent() override
	{
		handleKeyboard();
		GL::defaultFramebuffer.clear(GL::FramebufferClear::Color);
		while (!drawQueue.empty()) {
			std::invoke(drawQueue.front());
			drawQueue.pop();
		}

		drawGUI();
		swapBuffers();
	}

	void drawGUI()
	{
	 	static double FPS = 0.0;
	 	profiler.endFrame();
	 	if (profiler.isMeasurementAvailable(DebugTools::FrameProfilerGL::Value::FrameTime)) {
	 		FPS = 1e9 / profiler.frameTimeMean();
	 	}
	 	profiler.beginFrame();

		_imgui.newFrame();
		if(ImGui::GetIO().WantTextInput && !isTextInputActive())
			startTextInput();
		else if(!ImGui::GetIO().WantTextInput && isTextInputActive())
			stopTextInput();

		ImGui::Begin("Stats");
		ImGui::Text("FPS: %.1f", FPS);
		ImGui::End();

	 	if (userGUI.has_value()) {
	 		std::invoke(this->userGUI.value());
	 	}

	 	// GL::Renderer::enable(GL::Renderer::Feature::Blending);
	 	GL::Renderer::enable(GL::Renderer::Feature::ScissorTest);
	 	// GL::Renderer::disable(GL::Renderer::Feature::FaceCulling);
	 	// GL::Renderer::disable(GL::Renderer::Feature::DepthTest);

	 	_imgui.drawFrame();

	 	// GL::Renderer::enable(GL::Renderer::Feature::DepthTest);
	 	// GL::Renderer::enable(GL::Renderer::Feature::FaceCulling);
	 	GL::Renderer::disable(GL::Renderer::Feature::ScissorTest);
	 	// GL::Renderer::disable(GL::Renderer::Feature::Blending);

	}

	Configuration makeWindowConfig(const Utility::Arguments& cliArgs)
	{
		Configuration cfg;
		GlfwApplication::Configuration::WindowFlags flags;
		cfg.setTitle("Tiny Hadron Collider");

		glfwInit(); // TODO: Temporary hack: need to find a way to query desktop resolution before glfwInitialization
		const GLFWvidmode* videoMode = glfwGetVideoMode(glfwGetPrimaryMonitor());
		if (cliArgs.isSet("fullscreen")) {
			flags |= Platform::GlfwApplication::Configuration::WindowFlag::Fullscreen;
			cfg.setSize({videoMode->width, videoMode->height});
		}
		else {
			flags |= GlfwApplication::Configuration::WindowFlag::Resizable;
			// TODO: this is flawed approach when dpiScaling != 1.0, fix it
			cfg.setSize({3 * videoMode->width/4, 3 * videoMode->height/4});
		}
		cfg.setWindowFlags(flags);
		return cfg;
	}

	GLConfiguration makeOpenGLConfig()
	{
		GLConfiguration cfg;
		return cfg;
	}
};
