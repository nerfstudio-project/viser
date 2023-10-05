/** Shaders for Gaussian splatting.
 *
 * These are adapted from Kevin Kwok, with only minor modifications.
 *
 * https://github.com/antimatter15/splat/
 */

export const vertexShaderSource = `
  precision mediump float;
  attribute vec2 position;

  attribute vec3 rgb;
  attribute float opacity;
  attribute vec3 center;
  attribute vec3 covA;
  attribute vec3 covB;

  uniform mat4 projectionMatrix, modelViewMatrix;
  uniform vec2 focal;
  uniform vec2 viewport;

  varying vec3 vRgb;
  varying float vOpacity;
  varying vec2 vPosition;

  mat3 transpose(mat3 m) {
    return mat3(
        m[0][0], m[1][0], m[2][0],
        m[0][1], m[1][1], m[2][1],
        m[0][2], m[1][2], m[2][2]
    );
  }

  void main () {
    // Get center wrt camera. modelViewMatrix is T_cam_world.
    vec4 c_cam = modelViewMatrix * vec4(center, 1);
    vec4 pos2d = projectionMatrix * c_cam;

    // Splat covariance.
    mat3 cov3d = mat3(
        covA.x, covA.y, covA.z, 
        covA.y, covB.x, covB.y,
        covA.z, covB.y, covB.z
    );
    mat3 J = mat3(
        // Note that matrices are column-major.
        focal.x / c_cam.z, 0., 0.0, 
        0., focal.y / c_cam.z, 0.0, 
        -(focal.x * c_cam.x) / (c_cam.z * c_cam.z), -(focal.y * c_cam.y) / (c_cam.z * c_cam.z), 0.
    );
    mat3 A = J * mat3(modelViewMatrix);
    mat3 cov_proj = A * cov3d * transpose(A);
    float diag1 = cov_proj[0][0] + 0.3;
    float offDiag = cov_proj[0][1];
    float diag2 = cov_proj[1][1] + 0.3;

    // Eigendecomposition. This can mostly be derived from characteristic equation, etc.
    float mid = 0.5 * (diag1 + diag2);
    float radius = length(vec2((diag1 - diag2) / 2.0, offDiag));
    float lambda1 = mid + radius;
    float lambda2 = max(mid - radius, 0.1);
    vec2 diagonalVector = normalize(vec2(offDiag, lambda1 - diag1));
    vec2 v1 = min(sqrt(2.0 * lambda1), 1024.0) * diagonalVector;
    vec2 v2 = min(sqrt(2.0 * lambda2), 1024.0) * vec2(diagonalVector.y, -diagonalVector.x);

    vRgb = rgb;
    vOpacity = opacity;
    vPosition = position;

    gl_Position = vec4(
        vec2(pos2d) / pos2d.w
            + position.x * v1 / viewport * 2.0 
            + position.y * v2 / viewport * 2.0, pos2d.z / pos2d.w, 1.);
}
`;

export const fragmentShaderSource = `
  precision mediump float;

  varying vec3 vRgb;
  varying float vOpacity;
  varying vec2 vPosition;

  uniform vec2 viewport;
  uniform vec2 focal;


  void main () {    
    float A = -dot(vPosition, vPosition);
    if (A < -4.0) discard;
    float B = exp(A) * vOpacity;
    gl_FragColor = vec4(vRgb.rgb, B);
  }
`;
