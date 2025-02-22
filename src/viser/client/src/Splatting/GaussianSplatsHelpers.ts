import React from "react";
import * as THREE from "three";
import { create } from "zustand";
import { Object3D } from "three";
import { useThree } from "@react-three/fiber";
import { shaderMaterial } from "@react-three/drei";

const GaussianSplatMaterial = /* @__PURE__ */ shaderMaterial(
  {
    numGaussians: 0,
    focal: 100.0,
    viewport: [640, 480],
    near: 1.0,
    far: 100.0,
    depthTest: true,
    depthWrite: false,
    transparent: true,
    textureBuffer: null,
    shTextureBuffer: null,
    textureT_camera_groups: null,
    transitionInState: 0.0,
  },
  `precision highp usampler2D; // Most important: ints must be 32-bit.
  precision mediump float;

  // Index from the splat sorter.
  attribute uint sortedIndex;

  // Buffers for splat data; each Gaussian gets 4 floats and 4 int32s. We just
  // copy quadjr for this.
  uniform usampler2D textureBuffer;

  // Buffer for spherical harmonics; Each Gaussian gets 24 int32s representing
  // this information (Each coefficient is 16 bits, corr. to 48 coeffs.).
  uniform usampler2D shTextureBuffer;

  // We could also use a uniform to store transforms, but this would be more
  // limiting in terms of the # of groups we can have.
  uniform sampler2D textureT_camera_groups;

  // Various other uniforms...
  uniform uint numGaussians;
  uniform vec2 focal;
  uniform vec2 viewport;
  uniform float near;
  uniform float far;

  // Fade in state between [0, 1].
  uniform float transitionInState;

  out vec4 vRgba;
  out vec2 vPosition;

  // Function to fetch and construct the i-th transform matrix using texelFetch
  mat4 getGroupTransform(uint i) {
    // Calculate the base index for the i-th transform.
    uint baseIndex = i * 3u;

    // Fetch the texels that represent the first 3 rows of the transform. We
    // choose to use row-major here, since it lets us exclude the fourth row of
    // the matrix.
    vec4 row0 = texelFetch(textureT_camera_groups, ivec2(baseIndex + 0u, 0), 0);
    vec4 row1 = texelFetch(textureT_camera_groups, ivec2(baseIndex + 1u, 0), 0);
    vec4 row2 = texelFetch(textureT_camera_groups, ivec2(baseIndex + 2u, 0), 0);

    // Construct the mat4 with the fetched rows.
    mat4 transform = mat4(row0, row1, row2, vec4(0.0, 0.0, 0.0, 1.0));
    return transpose(transform);
  }

  void main () {
    // Get position + scale from float buffer.
    ivec2 texSize = textureSize(textureBuffer, 0);
    uint texStart = sortedIndex << 1u;
    ivec2 texPos0 = ivec2(texStart % uint(texSize.x), texStart / uint(texSize.x));


    // Fetch from textures.
    uvec4 floatBufferData = texelFetch(textureBuffer, texPos0, 0);
    mat4 T_camera_group = getGroupTransform(floatBufferData.w);

    // Any early return will discard the fragment.
    gl_Position = vec4(0.0, 0.0, 2.0, 1.0);

    // Get center wrt camera. modelViewMatrix is T_cam_world.
    vec3 center = uintBitsToFloat(floatBufferData.xyz);
    vec4 c_cam = T_camera_group * vec4(center, 1);
    if (-c_cam.z < near || -c_cam.z > far)
      return;
    vec4 pos2d = projectionMatrix * c_cam;
    float clip = 1.1 * pos2d.w;
    if (pos2d.x < -clip || pos2d.x > clip || pos2d.y < -clip || pos2d.y > clip)
      return;

    // Read covariance terms.
    ivec2 texPos1 = ivec2((texStart + 1u) % uint(texSize.x), (texStart + 1u) / uint(texSize.x));
    uvec4 intBufferData = texelFetch(textureBuffer, texPos1, 0);

    // Get covariance terms from int buffer.
    uint rgbaUint32 = intBufferData.w;
    vec2 triu01 = unpackHalf2x16(intBufferData.x);
    vec2 triu23 = unpackHalf2x16(intBufferData.y);
    vec2 triu45 = unpackHalf2x16(intBufferData.z);

    // Get spherical harmonics terms from int buffer. 48 coefficents per vertex.
    uint shTexStart = sortedIndex * 6u;
    ivec2 shTexSize = textureSize(shTextureBuffer, 0);
    float sh_coeffs_unpacked[48];
    for (int i = 0; i < 6; i++) {
        ivec2 shTexPos = ivec2((shTexStart + uint(i)) % uint(shTexSize.x), (shTexStart + uint(i)) / uint(shTexSize.x));
        uvec4 packedCoeffs = texelFetch(shTextureBuffer, shTexPos, 0);

        // unpack each uint32 directly into two float16 values, we read 4 at a time
        vec2 unpacked;
        unpacked = unpackHalf2x16(packedCoeffs.x);
        sh_coeffs_unpacked[i*8]   = unpacked.x;
        sh_coeffs_unpacked[i*8+1] = unpacked.y;
        
        unpacked = unpackHalf2x16(packedCoeffs.y);
        sh_coeffs_unpacked[i*8+2] = unpacked.x;
        sh_coeffs_unpacked[i*8+3] = unpacked.y;
        
        unpacked = unpackHalf2x16(packedCoeffs.z);
        sh_coeffs_unpacked[i*8+4] = unpacked.x;
        sh_coeffs_unpacked[i*8+5] = unpacked.y;
        
        unpacked = unpackHalf2x16(packedCoeffs.w);
        sh_coeffs_unpacked[i*8+6] = unpacked.x;
        sh_coeffs_unpacked[i*8+7] = unpacked.y;
    }

    // Transition in.
    float startTime = 0.8 * float(sortedIndex) / float(numGaussians);
    float cov_scale = smoothstep(startTime, startTime + 0.2, transitionInState);

    // Do the actual splatting.
    mat3 cov3d = mat3(
        triu01.x, triu01.y, triu23.x,
        triu01.y, triu23.y, triu45.x,
        triu23.x, triu45.x, triu45.y
    );
    mat3 J = mat3(
        // Matrices are column-major.
        focal.x / c_cam.z, 0., 0.0,
        0., focal.y / c_cam.z, 0.0,
        -(focal.x * c_cam.x) / (c_cam.z * c_cam.z), -(focal.y * c_cam.y) / (c_cam.z * c_cam.z), 0.
    );
    mat3 A = J * mat3(T_camera_group);
    mat3 cov_proj = A * cov3d * transpose(A);
    float diag1 = cov_proj[0][0] + 0.3;
    float offDiag = cov_proj[0][1];
    float diag2 = cov_proj[1][1] + 0.3;

    // Eigendecomposition.
    float mid = 0.5 * (diag1 + diag2);
    float radius = length(vec2((diag1 - diag2) / 2.0, offDiag));
    float lambda1 = mid + radius;
    float lambda2 = mid - radius;
    if (lambda2 < 0.0)
      return;
    vec2 diagonalVector = normalize(vec2(offDiag, lambda1 - diag1));
    vec2 v1 = min(sqrt(2.0 * lambda1), 1024.0) * diagonalVector;
    vec2 v2 = min(sqrt(2.0 * lambda2), 1024.0) * vec2(diagonalVector.y, -diagonalVector.x);

    // Calculate the spherical harmonics.
    // According to gsplat implementation, seems that "x" and "y" have opposite direction
    // of conventional SH directions, so square brackets contains the sign of resulting variable
    // multiplications.
    // A comprehensible table of Real SH constants: 
    // https://en.wikipedia.org/wiki/Table_of_spherical_harmonics
    vec3 viewDir = normalize(center - cameraPosition);
    // C0 = 0.5 * sqrt(1.0 / pi)
    const float C0 = 0.28209479177387814;   
    // C1[0] = sqrt(3.0 / (4.0 * pi)) * [-1]
    // C1[1] = sqrt(3.0 / (4.0 * pi)) * [1]
    // C1[2] = sqrt(3.0 / (4.0 * pi)) * [-1]
    const float C1[3] = float[3](
        -0.4886025119029199,
        0.4886025119029199,
        -0.4886025119029199
    );
    // C2[0] = 0.5 * sqrt(15/pi) * [1]
    // C2[1] = 0.5 * sqrt(15/pi) * [-1]
    // C2[2] = 0.25 * sqrt(5/pi) * [1]
    // C2[3] = 0.5 * sqrt(15/pi) * [-1]
    // C2[4] = 0.25 * sqrt(15/pi) * [1]
    const float C2[5] = float[5](
        1.0925484305920792,     
        -1.0925484305920792,    
        0.31539156525252005,    
        -1.0925484305920792,    
        0.5462742152960396      
    );
    // C3[0] = 0.25 * sqrt(35/(2pi)) * [-1]
    // C3[1] = 0.5 * sqrt(105/pi) * [1]
    // C3[2] = 0.25 * sqrt(21/(2pi)) * [-1]
    // C3[3] = 0.25 * sqrt(7/pi) * [1]
    // C3[4] = 0.25 * sqrt(21/(2pi)) * [-1]
    // C3[5] = 0.25 * sqrt(105/(pi)) * [1]
    // C3[6] = 0.25 * sqrt(35/(2pi)) * [-1]
    const float C3[7] = float[7](
        -0.5900435899266435,    
        2.890611442640554,      
        -0.4570457994644658,    
        0.3731763325901154,     
        -0.4570457994644658,    
        1.445305721320277,      
        -0.5900435899266435     
    );

    vec3 sh_coeffs[16];
    for (int i = 0; i < 16; i++) {
        sh_coeffs[i] = vec3(sh_coeffs_unpacked[i*3], sh_coeffs_unpacked[i*3+1], sh_coeffs_unpacked[i*3+2]);
    }

    // View-dependent variables

    float x = viewDir.x;
    float y = viewDir.y;
    float z = viewDir.z;
    float xx = viewDir.x * viewDir.x;
    float yy = viewDir.y * viewDir.y;
    float zz = viewDir.z * viewDir.z;
    float xy = viewDir.x * viewDir.y;
    float yz = viewDir.y * viewDir.z;
    float xz = viewDir.x * viewDir.z;
    
    // 0th degree
    vec3 rgb = C0 * sh_coeffs[0];
    vec3 pointFive = vec3(0.5, 0.5, 0.5);

    // 1st degree
    // From here, variables are included in multiplication with constants
    float pSH1 = C1[0] * y;
    float pSH2 = C1[1] * z;
    float pSH3 = C1[2] * x;
    rgb = rgb + pSH1 * sh_coeffs[1] +
                pSH2 * sh_coeffs[2] + 
                pSH3 * sh_coeffs[3];

    // 2nd degree
    float pSH4 = C2[0] * xy;
    float pSH5 = C2[1] * yz;
    float pSH6 = C2[2] * (3.0 * zz - 1.0);
    float pSH7 = C2[3] * xz;
    float pSH8 = C2[4] * (xx - yy);
    rgb = rgb + pSH4 * sh_coeffs[4] + 
                pSH5 * sh_coeffs[5] + 
                pSH6 * sh_coeffs[6] + 
                pSH7 * sh_coeffs[7] + 
                pSH8 * sh_coeffs[8];

    // 3rd degree
    float pSH9 = C3[0] * y * (3.0 * xx - yy);
    float pSH10 = C3[1] * x * y * z;
    float pSH11 = C3[2] * y * (5.0 * zz - 1.0);
    float pSH12 = C3[3] * z * (5.0 * zz - 3.0);
    float pSH13 = C3[4] * x * (5.0 * zz - 1.0);
    float pSH14 = C3[5] * (xx - yy) * z;
    float pSH15 = C3[6] * x * (xx - 3.0 * yy);
    rgb = rgb + pSH9 * sh_coeffs[9] + 
                pSH10 * sh_coeffs[10] + 
                pSH11 * sh_coeffs[11] + 
                pSH12 * sh_coeffs[12] + 
                pSH13 * sh_coeffs[13] + 
                pSH14 * sh_coeffs[14] + 
                pSH15 * sh_coeffs[15];

    // Finalize the color
    vRgba = vec4(rgb + pointFive, float(rgbaUint32 >> uint(24)) / 255.0);

    // Throw the Gaussian off the screen if it's too close, too far, or too small.
    float weightedDeterminant = vRgba.a * (diag1 * diag2 - offDiag * offDiag);
    if (weightedDeterminant < 0.25)
      return;
    vPosition = position.xy;

    gl_Position = vec4(
        vec2(pos2d) / pos2d.w
            + position.x * v1 / viewport * 2.0
            + position.y * v2 / viewport * 2.0, pos2d.z / pos2d.w, 1.);
  }
`,
  `precision mediump float;

  uniform vec2 viewport;
  uniform vec2 focal;

  in vec4 vRgba;
  in vec2 vPosition;

  void main () {
    float A = -dot(vPosition, vPosition);
    if (A < -4.0) discard;
    float B = exp(A) * vRgba.a;
    if (B < 0.01) discard;  // alphaTest.
    gl_FragColor = vec4(vRgba.rgb, B);
  }`,
);

/**Hook to generate properties for rendering Gaussians via a three.js mesh.*/
export function useGaussianMeshProps(
  gaussianBuffer: Uint32Array,
  combinedSHBuffer: Uint32Array,
  numGroups: number,
) {
  const numGaussians = gaussianBuffer.length / 8;
  const maxTextureSize = useThree((state) => state.gl).capabilities
    .maxTextureSize;

  // Create instanced geometry.
  const geometry = new THREE.InstancedBufferGeometry();
  geometry.instanceCount = numGaussians;
  geometry.setIndex(
    new THREE.BufferAttribute(new Uint32Array([0, 2, 1, 0, 3, 2]), 1),
  );
  geometry.setAttribute(
    "position",
    new THREE.BufferAttribute(
      new Float32Array([-2, -2, 2, -2, 2, 2, -2, 2]),
      2,
    ),
  );

  // Rendering order for Gaussians.
  const sortedIndexAttribute = new THREE.InstancedBufferAttribute(
    new Uint32Array(numGaussians),
    1,
  );
  sortedIndexAttribute.setUsage(THREE.DynamicDrawUsage);
  geometry.setAttribute("sortedIndex", sortedIndexAttribute);

  // Create texture buffers.
  // We store 4 floats and 4 int32s per Gaussian.
  // One "numGaussians" corresponds to 4 32-bit values.
  const textureWidth = Math.min(numGaussians * 2, maxTextureSize);
  const textureHeight = Math.ceil((numGaussians * 2) / textureWidth);
  const bufferPadded = new Uint32Array(textureWidth * textureHeight * 4);
  bufferPadded.set(gaussianBuffer);
  const textureBuffer = new THREE.DataTexture(
    bufferPadded,
    textureWidth,
    textureHeight,
    THREE.RGBAIntegerFormat,
    THREE.UnsignedIntType,
  );
  textureBuffer.internalFormat = "RGBA32UI";
  textureBuffer.needsUpdate = true;

  const rowMajorT_camera_groups = new Float32Array(numGroups * 12);
  const textureT_camera_groups = new THREE.DataTexture(
    rowMajorT_camera_groups,
    (numGroups * 12) / 4,
    1,
    THREE.RGBAFormat,
    THREE.FloatType,
  );
  textureT_camera_groups.internalFormat = "RGBA32F";
  textureT_camera_groups.needsUpdate = true;

  // Values taken from PR https://github.com/nerfstudio-project/viser/pull/286/files
  // WIDTH AND HEIGHT ARE MEASURED IN TEXELS
  // As 48 x float16 = 96 bytes and each texel is 4 uint32s = 16 bytes
  // We can fit 6 spherical harmonics coefficients in a single texel
  const shTextureWidth = Math.min(numGaussians * 6, maxTextureSize);
  const shTextureHeight = Math.ceil((numGaussians * 6) / shTextureWidth);
  const shBufferPadded = new Uint32Array(shTextureWidth * shTextureHeight * 4);
  shBufferPadded.set(combinedSHBuffer);
  const shTextureBuffer = new THREE.DataTexture(
    shBufferPadded,
    shTextureWidth,
    shTextureHeight,
    THREE.RGBAIntegerFormat,
    THREE.UnsignedIntType,
  );
  shTextureBuffer.internalFormat = "RGBA32UI";
  shTextureBuffer.needsUpdate = true;

  const material = new GaussianSplatMaterial({
    // @ts-ignore
    textureBuffer: textureBuffer,
    shTextureBuffer: shTextureBuffer,
    textureT_camera_groups: textureT_camera_groups,
    numGaussians: 0,
    transitionInState: 0.0,
  });

  return {
    geometry,
    material,
    textureBuffer,
    shTextureBuffer,
    sortedIndexAttribute,
    textureT_camera_groups,
    rowMajorT_camera_groups,
  };
}
/**Global splat state.*/
interface SplatState {
  groupBufferFromId: { [id: string]: Uint32Array };
  nodeRefFromId: React.MutableRefObject<{
    [name: string]: undefined | Object3D;
  }>;
  setBuffer: (id: string, buffer: Uint32Array) => void;
  removeBuffer: (id: string) => void;
}

/**Hook for creating global splat state.*/
export function useGaussianSplatStore() {
  const nodeRefFromId = React.useRef({});
  return React.useState(() =>
    create<SplatState>((set) => ({
      groupBufferFromId: {},
      nodeRefFromId: nodeRefFromId,
      setBuffer: (id, buffer) => {
        return set((state) => ({
          groupBufferFromId: { ...state.groupBufferFromId, [id]: buffer },
        }));
      },
      removeBuffer: (id) => {
        return set((state) => {
          // eslint-disable-next-line @typescript-eslint/no-unused-vars
          const { [id]: _, ...buffers } = state.groupBufferFromId;
          return { groupBufferFromId: buffers };
        });
      },
    })),
  )[0];
}

export const GaussianSplatsContext = React.createContext<{
  useGaussianSplatStore: ReturnType<typeof useGaussianSplatStore>;
  updateCamera: React.MutableRefObject<
    | null
    | ((
        camera: THREE.PerspectiveCamera,
        width: number,
        height: number,
        blockingSort: boolean,
      ) => void)
  >;
  meshPropsRef: React.MutableRefObject<ReturnType<
    typeof useGaussianMeshProps
  > | null>;
} | null>(null);
