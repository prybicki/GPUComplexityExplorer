#include <random>

#include <fmt/format.h>

#include <Visualizer.hpp>
#include <Vector.hpp>
#include <cuda/kernels.hpp>

#include <Magnum/DebugTools/FrameProfiler.h>

std::default_random_engine engine;

struct Particles
{
	count_t count;
	float* radius;
	Vec2f* position;
	Vec2f* velocity;
	Vec4f* color;

	std::vector<float> hRad;
	std::vector<Vec2f> hPos, hVel;
	std::vector<Vec4f> hCol;

	Particles(count_t count, Vec2f posOrigin, float posRad, float radMin, float radMax, Vec4f col) : count(count)
	{
		std::uniform_real_distribution<float> dRad {radMin, radMax};
		for (count_t i = 0; i < count; ++i) {

			hPos.emplace_back(posOrigin + randomVectorInRing(0.0f, posRad));
			hVel.emplace_back(randomVectorInRing(0.1f, 1.0f));
			hCol.emplace_back(col);
			hRad.emplace_back(dRad(engine));
		}
		CHECK_CUDA(cudaMalloc(&position, sizeof(Vec2f) * count));
		CHECK_CUDA(cudaMalloc(&velocity, sizeof(Vec2f) * count));
		CHECK_CUDA(cudaMalloc(&color, sizeof(Vec4f) * count));
		CHECK_CUDA(cudaMalloc(&radius, sizeof(float) * count));

		CHECK_CUDA(cudaMemcpy(position, hPos.data(), sizeof(Vec2f) * count, cudaMemcpyHostToDevice));
		CHECK_CUDA(cudaMemcpy(velocity, hVel.data(), sizeof(Vec2f) * count, cudaMemcpyHostToDevice));
		CHECK_CUDA(cudaMemcpy(color, hCol.data(), sizeof(Vec4f) * count, cudaMemcpyHostToDevice));
		CHECK_CUDA(cudaMemcpy(radius, hRad.data(), sizeof(float) * count, cudaMemcpyHostToDevice));
	}

private:
	Vec2f randomVectorInRing(float r, float R)
	{
		std::uniform_real_distribution<float> dAngle {0, 2 * M_PI};
		std::uniform_real_distribution<float> dLen {r, R};

		float angle = dAngle(engine);
		return dLen(engine) * Vec2f(sinf(angle), cosf(angle));
	}
};

using namespace std::literals;

int main(int argc, char** argv)
{
	constexpr count_t count = 131072 * 6;
	Visualizer vis({argc, argv});

	Vec4f r = {1.0f, 0.0f, 0.0f, 0.8f};
	Vec4f g = {0.0f, 1.0f, 0.0f, 0.8f};
	Vec4f b = {0.0f, 0.0f, 1.0f, 0.8f};

	// Fill background to avoid the ugly.
	vis.redraw();
	vis.mainLoopIteration();

	Particles left =  { count, {-0.7f, -0.3f}, 0.1f, 0.001, 0.002, r};
	Particles top =   { count, {+0.0f, +0.7f}, 0.1f, 0.001, 0.002, g};
	Particles right = { count, {+0.7f, -0.3f}, 0.1f, 0.001, 0.002, b};

	DebugTools::FrameProfilerGL profiler { DebugTools::FrameProfilerGL::Value::FrameTime, 10};
	bool shouldContinue;
	do {
		profiler.beginFrame();
		cm.runSync1D(count, 256, kApplyVelocity, count, 0.005f, top.velocity, top.position);
		cm.runSync1D(count, 256, kApplyVelocity, count, 0.005f, left.velocity, left.position);
		cm.runSync1D(count, 256, kApplyVelocity, count, 0.005f, right.velocity, right.position);
		vis.renderParticles(count, top.position, top.radius, top.color);
		vis.renderParticles(count, left.position, left.radius, left.color);
		vis.renderParticles(count, right.position, right.radius, right.color);
		vis.redraw();
		shouldContinue = vis.mainLoopIteration();
		profiler.endFrame();

		if (profiler.isMeasurementAvailable(DebugTools::FrameProfilerGL::Value::FrameTime)) {
			fmt::print("FPS={:4.2f}\n", 1e9 / profiler.frameTimeMean());
		}
	}
	while (shouldContinue);
	return 0;
}
