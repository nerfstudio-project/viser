import React from "react";
import * as THREE from "three";
import SplatSortWorker from "./SplatSortWorker?worker";
import { useFrame, useThree } from "@react-three/fiber";
import { shaderMaterial } from "@react-three/drei";

export const GaussianSplatsContext =
  React.createContext<null | React.MutableRefObject<{
    numSorting: number;
    sortUpdateCallbacks: (() => void)[];
  }>>(null);

export type GaussianBuffers = {
  buffer: Uint32Array;
};

const GaussianSplatMaterial = /* @__PURE__ */ shaderMaterial(
  {
    numGaussians: 0,
    focal: 100.0,
    viewport: [640, 480],
    near: 1.0,
    far: 100.0,
    depthTest: true,
    depthWrite: true,
    transparent: true,
    bufferTexture: null,
    sortSynchronizedModelViewMatrix: new THREE.Matrix4(),
    transitionInState: 0.0,
  },
  `precision highp usampler2D; // Most important: ints must be 32-bit.
  precision mediump float;

  // Index from the splat sorter.
  attribute uint sortedIndex;

  // Buffers for splat data; each Gaussian gets 4 floats and 4 int32s. We just
  // copy quadjr for this.
  uniform usampler2D bufferTexture;

  // Various other uniforms...
  uniform uint numGaussians;
  uniform vec2 focal;
  uniform vec2 viewport;
  uniform float near;
  uniform float far;

  // Depth testing is useful for compositing multiple splat objects, but causes
  // artifacts when closer Gaussians are rendered before further ones.
  // Synchronizing the modelViewMatrix updates used for depth computation with
  // the splat sorter mitigates this for Gaussians within the same object.
  uniform mat4 sortSynchronizedModelViewMatrix;

  // Fade in state between [0, 1].
  uniform float transitionInState;

  out vec4 vRgba;
  out vec2 vPosition;

	float hash2D( vec2 value ) {
		return fract( 1.0e4 * sin( 17.0 * value.x + 0.1 * value.y ) * ( 0.1 + abs( sin( 13.0 * value.y + value.x ) ) ) );
	}
	float hash3D( vec3 value ) {
		return hash2D( vec2( hash2D( value.xy ), value.z ) );
	}

  void main () {
    // Any early return will discard the fragment.
    gl_Position = vec4(0.0, 0.0, 2000.0, 1.0);

    // Get position + scale from float buffer.
    ivec2 texSize = textureSize(bufferTexture, 0);
    ivec2 texPos0 = ivec2((sortedIndex * 2u) % uint(texSize.x), (sortedIndex * 2u) / uint(texSize.x));
    uvec4 floatBufferData = texelFetch(bufferTexture, texPos0, 0);
    vec3 center = uintBitsToFloat(floatBufferData.xyz);

    // Get center wrt camera. modelViewMatrix is T_cam_world.
    vec4 c_cam = modelViewMatrix * vec4(center, 1);
    if (-c_cam.z < near || -c_cam.z > far)
      return;
    vec4 pos2d = projectionMatrix * c_cam;
    float clip = 1.1 * pos2d.w;
    if (pos2d.x < -clip || pos2d.x > clip || pos2d.y < -clip || pos2d.y > clip)
      return;

    vec4 c_camstable = sortSynchronizedModelViewMatrix * vec4(center, 1);
    vec4 stablePos2d = projectionMatrix * c_camstable;

    float perGaussianShift = 1.0 - (float(numGaussians * 2u) - float(sortedIndex)) / float(numGaussians * 2u);
    float cov_scale = max(0.0, transitionInState - perGaussianShift) / (1.0 - perGaussianShift);

    // Get covariance terms from int buffer.
    ivec2 texPos1 = ivec2((sortedIndex * 2u + 1u) % uint(texSize.x), (sortedIndex * 2u + 1u) / uint(texSize.x));
    uvec4 intBufferData = texelFetch(bufferTexture, texPos1, 0);
    uint rgbaUint32 = intBufferData.w;
    vec2 chol01 = unpackHalf2x16(intBufferData.x);
    vec2 chol23 = unpackHalf2x16(intBufferData.y);
    vec2 chol45 = unpackHalf2x16(intBufferData.z);

    // Do the actual splatting.
    mat3 chol = mat3(
        chol01.x, chol01.y, chol23.x,
        0.,       chol23.y, chol45.x,
        0.,       0.,       chol45.y
    );
    mat3 cov3d = chol * transpose(chol) * cov_scale;
    mat3 J = mat3(
        // Matrices are column-major.
        focal.x / c_cam.z, 0., 0.0,
        0., focal.y / c_cam.z, 0.0,
        -(focal.x * c_cam.x) / (c_cam.z * c_cam.z), -(focal.y * c_cam.y) / (c_cam.z * c_cam.z), 0.
    );
    mat3 A = J * mat3(modelViewMatrix);
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

    vRgba = vec4(
      float(rgbaUint32 & uint(0xFF)) / 255.0,
      float((rgbaUint32 >> uint(8)) & uint(0xFF)) / 255.0,
      float((rgbaUint32 >> uint(16)) & uint(0xFF)) / 255.0,
      float(rgbaUint32 >> uint(24)) / 255.0
    );

    // Throw the Gaussian off the screen if it's too close, too far, or too small.
    float weightedDeterminant = vRgba.a * (diag1 * diag2 - offDiag * offDiag);
    if (weightedDeterminant < 0.1)
      return;
    if (weightedDeterminant < 1.0 && hash3D(center) < weightedDeterminant)  // This is not principled. It just makes things faster.
      return;
    vPosition = position.xy;

    gl_Position = vec4(
        vec2(pos2d) / pos2d.w
            + position.x * v1 / viewport * 2.0
            + position.y * v2 / viewport * 2.0, stablePos2d.z / stablePos2d.w, 1.);
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
    if (B < 0.03) discard;  // alphaTest.
    gl_FragColor = vec4(vRgba.rgb, B);
  }`,
);

export default function GaussianSplats({
  buffers,
}: {
  buffers: GaussianBuffers;
}) {
  const [geometry, setGeometry] = React.useState<THREE.BufferGeometry>();
  const [material, setMaterial] = React.useState<THREE.ShaderMaterial>();
  const splatSortWorkerRef = React.useRef<Worker | null>(null);
  const maxTextureSize = useThree((state) => state.gl).capabilities
    .maxTextureSize;
  const initializedTextures = React.useRef<boolean>(false);
  const [sortSynchronizedModelViewMatrix] = React.useState(new THREE.Matrix4());

  const splatContext = React.useContext(GaussianSplatsContext)!.current;

  // We'll use the vanilla three.js API, which for our use case is more
  // flexible than the declarative version (particularly for operations like
  // dynamic updates to buffers and shader uniforms).
  React.useEffect(() => {
    // Create geometry. Each Gaussian will be rendered as a quad.
    const geometry = new THREE.InstancedBufferGeometry();
    const numGaussians = buffers.buffer.length / 8;
    geometry.instanceCount = numGaussians;

    // Quad geometry.
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
    const textureWidth = Math.min(numGaussians * 2, maxTextureSize);
    const textureHeight = Math.ceil((numGaussians * 2) / textureWidth);

    const bufferPadded = new Uint32Array(textureWidth * textureHeight * 4);
    bufferPadded.set(buffers.buffer);

    const bufferTexture = new THREE.DataTexture(
      bufferPadded,
      textureWidth,
      textureHeight,
      THREE.RGBAIntegerFormat,
      THREE.UnsignedIntType,
    );
    bufferTexture.internalFormat = "RGBA32UI";

    const material = new GaussianSplatMaterial({
      // @ts-ignore
      bufferTexture: bufferTexture,
      numGaussians: 0,
      transitionInState: 0.0,
      sortSynchronizedModelViewMatrix: new THREE.Matrix4(),
    });

    // Update component state.
    setGeometry(geometry);
    setMaterial(material);

    // Create sorting worker.
    const sortWorker = new SplatSortWorker();
    sortWorker.onmessage = (e) => {
      sortedIndexAttribute.set(e.data.sortedIndices as Int32Array);
      const synchronizedSortUpdateCallback = () => {
        isSortingRef.current = false;

        // Wait for onmessage  to be triggered for all Gaussians.
        sortedIndexAttribute.needsUpdate = true;
        material.uniforms.sortSynchronizedModelViewMatrix.value.copy(
          sortSynchronizedModelViewMatrix,
        );
        // A simple but reasonably effective heuristic for render ordering.
        //
        // To minimize artifacts:
        // - When there are multiple splat objects, we want to render the closest
        //   ones *last*. This improves the likelihood of correct alpha
        //   compositing and reduces reliance on alpha testing.
        // - We generally want to render other objects like meshes *before*
        //   Gaussians. They're usually opaque.
        meshRef.current!.renderOrder = (-e.data.minDepth as number) + 1000.0;

        // Trigger initial render.
        if (!initializedTextures.current) {
          material.uniforms.numGaussians.value = numGaussians;
          bufferTexture.needsUpdate = true;
          initializedTextures.current = true;
        }
      };

      // Synchronize sort updates across multiple Gaussian splats. This
      // prevents z-fighting.
      splatContext.numSorting -= 1;
      if (splatContext.numSorting === 0) {
        synchronizedSortUpdateCallback();
        console.log(splatContext.sortUpdateCallbacks.length);
        for (const callback of splatContext.sortUpdateCallbacks) {
          callback();
        }
        splatContext.sortUpdateCallbacks.length = 0;
      } else {
        splatContext.sortUpdateCallbacks.push(synchronizedSortUpdateCallback);
      }
    };
    sortWorker.postMessage({
      setBuffer: buffers.buffer,
    });
    splatSortWorkerRef.current = sortWorker;

    // We should always re-send view projection when buffers are replaced.
    prevT_camera_obj.identity();

    return () => {
      bufferTexture.dispose();
      geometry.dispose();
      if (material !== undefined) material.dispose();
      sortWorker.postMessage({ close: true });
    };
  }, [buffers]);

  // Synchronize view projection matrix with sort worker. We pre-allocate some
  // matrices to make life easier for the garbage collector.
  const meshRef = React.useRef<THREE.Mesh>(null);
  const isSortingRef = React.useRef(false);
  const [prevT_camera_obj] = React.useState(new THREE.Matrix4());
  const [T_camera_obj] = React.useState(new THREE.Matrix4());

  useFrame((state, delta) => {
    const mesh = meshRef.current;
    const sortWorker = splatSortWorkerRef.current;
    if (mesh === null || sortWorker === null) return;

    // Update camera parameter uniforms.
    const dpr = state.viewport.dpr;
    const fovY =
      ((state.camera as THREE.PerspectiveCamera).fov * Math.PI) / 180.0;
    const fovX = 2 * Math.atan(Math.tan(fovY / 2) * state.viewport.aspect);
    const fy = (dpr * state.size.height) / (2 * Math.tan(fovY / 2));
    const fx = (dpr * state.size.width) / (2 * Math.tan(fovX / 2));

    if (material === undefined) return;

    const uniforms = material.uniforms;
    uniforms.transitionInState.value = Math.min(
      uniforms.transitionInState.value + delta * 2.0,
      1.0,
    );
    uniforms.focal.value = [fx, fy];
    uniforms.near.value = state.camera.near;
    uniforms.far.value = state.camera.far;
    uniforms.viewport.value = [state.size.width * dpr, state.size.height * dpr];

    // Compute view projection matrix.
    // T_camera_obj = T_cam_world * T_world_obj.
    T_camera_obj.copy(state.camera.matrixWorldInverse).multiply(
      mesh.matrixWorld,
    );

    // If changed, use projection matrix to sort Gaussians.
    if (
      !isSortingRef.current &&
      (prevT_camera_obj === undefined || !T_camera_obj.equals(prevT_camera_obj))
    ) {
      sortSynchronizedModelViewMatrix.copy(T_camera_obj);
      sortWorker.postMessage({ setT_camera_obj: T_camera_obj.elements });
      splatContext.numSorting += 1;
      isSortingRef.current = true;
      prevT_camera_obj.copy(T_camera_obj);
    }
  });

  return <mesh ref={meshRef} geometry={geometry} material={material} />;
}
