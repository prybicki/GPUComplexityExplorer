#include <graphics/Visualizer.hpp>
#include <core/MemoryManager.hpp>
#include <macros/todo.hpp>
#include <thrust/reduce.h>
#include <thrust/device_ptr.h>
#include <compute/kernels.hpp>

#include <unistd.h>
#include <data/Field2D.hpp>

#include <compute/Accessors.hpp>

template struct DoPrint<LinMemAcc2D<float>>;

int main(int argc, char** argv)
{
	Visualizer v {argc, argv};
	Field2D heatField {
		NCube2f::centeredAt({0.0f, 0.0f}, {1024.0f, 1024.0f}),
		Vec2c {128, 128},
		PropertySet {
			Property {f32, "heat"}
		}
	};

	auto heatFieldArea = heatField.area();
	v.cameraLookAtWithPadding(heatFieldArea, 0.1f);

	auto square = heatFieldArea.scaled(0.1f);
	auto rectVertical = heatFieldArea.scaled({0.1f, 0.2f});
	auto rectHorizontal = heatFieldArea.scaled({0.2f, 0.1f});
	auto p2 = square.placedIn(heatFieldArea, {0.5f, 0.5f});
	auto p1 = square.placedIn(heatFieldArea, {0.1f, 0.1f});
	auto p3 = square.placedIn(heatFieldArea, {0.9f, 0.9f});
	auto p4 = rectVertical.placedIn(heatFieldArea, {0.33f, 0.66f});
	auto p5 = rectHorizontal.placedIn(heatFieldArea, {0.66f, 0.33f});

	fmt::print("{}\n{}\n{}\n{}\n{}\n", p1, p2, p3, p4, p5);

	LinMemAcc2D<float> map = heatField.property<float, LinMemAcc2D<float>>("heat");

	DoPrint<LinMemAcc2D<float>> doPrint({.map = map});
	auto d = heatField.dims();
	mm.run(ThreadsLayout(std::array{d.x(), d.y()}), doPrint.ptr, doPrint.args);

	// mm.run(ThreadsLayout({d.x(), d.y()}), kTmpSetNCube, d.x(), d.y(), fieldCurr, rect1, 1.0f);
	// rm.run(ThreadsLayout(std::array<int, 2>{sizeX, sizeY}), kTmpSetNCube, sizeX, sizeY, fieldCurr, rect2, 0.9f);
	// rm.run(ThreadsLayout(std::array<int, 2>{sizeX, sizeY}), kTmpSetNCube, sizeX, sizeY, fieldCurr, rect3, 0.8f);
    //
	// rm.run(ThreadsLayout(std::array<int, 2>{sizeX, sizeY}), kTmpSetNCube, sizeX, sizeY, fieldCurr, rect4, 0.8f);
	// rm.run(ThreadsLayout(std::array<int, 2>{sizeX, sizeY}), kTmpSetNCube, sizeX, sizeY, fieldCurr, rect5, 0.8f);
    //
	// while (v.mainLoopIteration())
	// {
	// 	auto begin = thrust::device_ptr<float>(fieldCurr);
	// 	auto end = thrust::device_ptr<float>(fieldCurr + sizeX * sizeY);
	// 	thrust::pair<thrust::device_ptr<float>, thrust::device_ptr<float>> minMaxPtrs = thrust::minmax_element(begin, end);
	// 	rm.run({sizeX * sizeY}, kTmpColorizeCustomF32, sizeX * sizeY, fieldCurr, minMaxPtrs.first.get(), minMaxPtrs.second.get(), fieldCol);
	// 	v.renderTexture(0, 0, sizeX, sizeY, fieldCol);
	// 	v.redraw();
    //
	// 	for (int i = 0; i < 100; i++) {
	// 		rm.run(ThreadsLayout(std::array<int, 2>{sizeX, sizeY}), kHeatTransfer, sizeX, sizeY, fieldCurr, fieldNext, 0.01f);
	// 		std::swap(fieldCurr, fieldNext);
    //
	// 	}
	// }
}
