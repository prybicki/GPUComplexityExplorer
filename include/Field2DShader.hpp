#pragma once

#include <Magnum/GL/AbstractShaderProgram.h>
#include <Magnum/DimensionTraits.h>

// This class exists because Magnum::Shaders::FlatGL2D does not support rectangle textures.
struct Field2DShader: public Magnum::GL::AbstractShaderProgram
{
	Field2DShader();
	Field2DShader& drawTexture(Magnum::GL::RectangleTexture& texture, float posX, float posY);
	Field2DShader& setTransformationProjectionMatrix(const Magnum::MatrixTypeFor<2, float>& matrix);
};