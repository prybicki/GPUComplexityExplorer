#include <Field2DShader.hpp>

#include <Magnum/Math/Vector2.h>
#include <Magnum/GL/Attribute.h>
#include <Magnum/GL/Version.h>
#include <Magnum/GL/Mesh.h>
#include <Magnum/GL/Context.h>
#include <Magnum/GL/Shader.h>
#include <Magnum/GL/RectangleTexture.h>
#include <Magnum/Shaders/GenericGL.h>

#include <Magnum/Math/Matrix3.h>

#include <Corrade/Containers/Reference.h>
#include <Corrade/Utility/Resource.h>
#include <fmt/format.h>
#include <ShaderSources.hpp>

using namespace Magnum;

struct Vertex {
	Vector2 position;
	Vector2 textureCoordinates;
};

const ShaderSources& shaderSources = ShaderSources::instance();

Field2DShader::Field2DShader()
{
	bindAttributeLocation(Shaders::GenericGL2D::Position::Location, "position");
	bindAttributeLocation(Shaders::GenericGL2D::TextureCoordinates::Location, "texPosition");

	MAGNUM_ASSERT_GL_VERSION_SUPPORTED(GL::Version::GL330);

	const Utility::Resource shaders {"shaders"};

	GL::Shader frag {GL::Version::GL330, GL::Shader::Type::Fragment};
	GL::Shader vert {GL::Version::GL330, GL::Shader::Type::Vertex};

	frag.addSource(shaders.get("Field2D.frag"));
	vert.addSource(shaders.get("Field2D.vert"));

	CORRADE_INTERNAL_ASSERT_OUTPUT(GL::Shader::compile({vert, frag}));

	this->attachShaders({vert, frag});

	CORRADE_INTERNAL_ASSERT_OUTPUT(link());
}

Field2DShader &Field2DShader::drawTexture(GL::RectangleTexture &texture, float posX, float posY)
{
	auto size = texture.imageSize();
	Vertex ccwQuadStrip[4] = {
			{{posX, posY}, {0.0, 0.0}},
			{{posX, posY + size.y()}, {0.0, static_cast<float>(size.y())}},
			{{posX + size.x(), posY}, {static_cast<float>(size.x()), 0.0}},
			{{posX + size.x(), posY + size.y()}, {static_cast<float>(size.x()), static_cast<float>(size.y())}},
	};

	GL::Buffer vertices(ccwQuadStrip, GL::BufferUsage::StaticDraw);
	GL::Mesh mesh(GL::MeshPrimitive::TriangleStrip);
	mesh.addVertexBuffer(vertices, 0, Shaders::GenericGL2D::Position{}, Shaders::GenericGL2D::TextureCoordinates{});
	mesh.setCount(sizeof(ccwQuadStrip) / sizeof(*ccwQuadStrip));

	setUniform(uniformLocation("textureData"), 0);
	texture.bind(0);
	draw(mesh);
	return *this;
}

Field2DShader& Field2DShader::setTransformationProjectionMatrix(const MatrixTypeFor<2, float> &matrix)
{
	setUniform(uniformLocation("transformationProjectionMatrix"), matrix);
	return *this;
}
