uniform vec3 color = vec3(1.0, 1.0, 1.0);
uniform sampler2DRect textureData;

in vec2 interpolatedTextureCoordinates;

out vec4 fragmentColor;

void main() {
    float c = texture(textureData, interpolatedTextureCoordinates).r;
    fragmentColor = vec4(c, c, c, 1.0);
}
