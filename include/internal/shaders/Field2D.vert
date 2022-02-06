layout(location = 0) in vec4 position;
layout(location = 1) in vec2 textureCoordinates;

out vec2 interpolatedTextureCoordinates;
uniform highp mat3 transformationProjectionMatrix;

void main() {
    interpolatedTextureCoordinates = textureCoordinates;

    gl_Position.xywz = vec4((transformationProjectionMatrix * position.xyw), 0.0);
}
