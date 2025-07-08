import { Instance, Instances, shaderMaterial } from "@react-three/drei";
import { useFrame, useThree } from "@react-three/fiber";
import { OutlinesIfHovered } from "./OutlinesIfHovered";
import React from "react";
import * as THREE from "three";
import { ImageMessage, PointCloudMessage } from "./WebsocketMessages";
import { BatchedMeshHoverOutlines } from "./mesh/BatchedMeshHoverOutlines";
import { MeshBasicMaterial } from "three";

const originGeom = new THREE.SphereGeometry(1.0);

const PointCloudMaterial = /* @__PURE__ */ shaderMaterial(
  {
    scale: 1.0,
    point_ball_norm: 0.0,
    uniformColor: new THREE.Color(1, 1, 1),
  },
  `
  precision mediump float;

  varying vec3 vPosition;
  varying vec3 vColor; // in the vertex shader
  uniform float scale;
  uniform vec3 uniformColor;

  void main() {
      vPosition = position;
      #ifdef USE_COLOR
      vColor = color;
      #else
      vColor = uniformColor;
      #endif
      vec4 world_pos = modelViewMatrix * vec4(position, 1.0);
      gl_Position = projectionMatrix * world_pos;
      gl_PointSize = (scale / -world_pos.z);
  }
   `,
  `varying vec3 vPosition;
  varying vec3 vColor;
  uniform float point_ball_norm;

  void main() {
      if (point_ball_norm < 1000.0) {
          float r = pow(
              pow(abs(gl_PointCoord.x - 0.5), point_ball_norm)
              + pow(abs(gl_PointCoord.y - 0.5), point_ball_norm),
              1.0 / point_ball_norm);
          if (r > 0.5) discard;
      }
      gl_FragColor = vec4(vColor, 1.0);
  }
   `,
);

export const PointCloud = React.forwardRef<
  THREE.Points,
  PointCloudMessage & { children?: React.ReactNode }
>(function PointCloud({ children, ...message }, ref) {
  const getThreeState = useThree((state) => state.get);

  const props = message.props;

  // Create geometry using useMemo for better performance.
  const geometry = React.useMemo(() => {
    const geometry = new THREE.BufferGeometry();

    if (message.props.precision === "float16") {
      geometry.setAttribute(
        "position",
        new THREE.Float16BufferAttribute(
          new Uint16Array(
            props.points.buffer.slice(
              props.points.byteOffset,
              props.points.byteOffset + props.points.byteLength,
            ),
          ),
          3,
        ),
      );
    } else {
      geometry.setAttribute(
        "position",
        new THREE.Float32BufferAttribute(
          new Float32Array(
            props.points.buffer.slice(
              props.points.byteOffset,
              props.points.byteOffset + props.points.byteLength,
            ),
          ),
          3,
        ),
      );
    }

    // Add color attribute if needed.
    if (props.colors.length > 3) {
      geometry.setAttribute(
        "color",
        new THREE.BufferAttribute(new Uint8Array(props.colors), 3, true),
      );
    } else if (props.colors.length < 3) {
      console.error(`Invalid color buffer length, got ${props.colors.length}`);
    }

    return geometry;
  }, [props.points, props.colors]);

  // Create material using useMemo for better performance.
  const material = React.useMemo(() => {
    const material = new PointCloudMaterial();

    if (props.colors.length > 3) {
      material.vertexColors = true;
    } else {
      material.vertexColors = false;
      material.uniforms.uniformColor.value = new THREE.Color(
        props.colors[0] / 255.0,
        props.colors[1] / 255.0,
        props.colors[2] / 255.0,
      );
    }

    return material;
  }, [props.colors]);

  // Clean up resources when component unmounts.
  React.useEffect(() => {
    return () => {
      geometry.dispose();
      material.dispose();
    };
  }, [geometry, material]);

  // Update material properties with point_ball_norm
  React.useEffect(() => {
    material.uniforms.scale.value = 10.0;
    material.uniforms.point_ball_norm.value = {
      square: Infinity,
      diamond: 1.0,
      circle: 2.0,
      rounded: 3.0,
      sparkle: 0.6,
    }[props.point_shape];
  }, [props.point_shape, material]);

  const rendererSize = new THREE.Vector2();
  useFrame(() => {
    // Match point scale to behavior of THREE.PointsMaterial().
    // point px height / actual height = point meters height / frustum meters height
    // frustum meters height = math.tan(fov / 2.0) * z
    // point px height = (point meters height / math.tan(fov / 2.0) * actual height)  / z
    material.uniforms.scale.value =
      (props.point_size /
        Math.tan(
          (((getThreeState().camera as THREE.PerspectiveCamera).fov / 180.0) *
            Math.PI) /
            2.0,
        )) *
      getThreeState().gl.getSize(rendererSize).height *
      getThreeState().gl.getPixelRatio();
  });
  return (
    <points
      frustumCulled={false}
      ref={ref}
      geometry={geometry}
      material={material}
    >
      {children}
    </points>
  );
});

/** Helper for adding coordinate frames as scene nodes. */
export const CoordinateFrame = React.forwardRef<
  THREE.Group,
  {
    showAxes?: boolean;
    axesLength?: number;
    axesRadius?: number;
    originRadius?: number;
    originColor?: number;
    children?: React.ReactNode;
  }
>(function CoordinateFrame(
  {
    showAxes = true,
    axesLength = 0.5,
    axesRadius = 0.0125,
    originRadius = undefined,
    originColor = 0xecec00,
    children,
  },
  ref,
) {
  originRadius = originRadius ?? axesRadius * 2;
  return (
    <group ref={ref}>
      {showAxes && (
        <>
          <mesh
            geometry={originGeom}
            scale={new THREE.Vector3(originRadius, originRadius, originRadius)}
          >
            <meshBasicMaterial color={originColor} />
            <OutlinesIfHovered />
          </mesh>
          <Instances limit={3}>
            <meshBasicMaterial />
            <cylinderGeometry args={[axesRadius, axesRadius, axesLength, 16]} />
            <Instance
              rotation={new THREE.Euler(0.0, 0.0, (3.0 * Math.PI) / 2.0)}
              position={[0.5 * axesLength, 0.0, 0.0]}
              color={0xcc0000}
            >
              <OutlinesIfHovered />
            </Instance>
            <Instance position={[0.0, 0.5 * axesLength, 0.0]} color={0x00cc00}>
              <OutlinesIfHovered />
            </Instance>
            <Instance
              rotation={new THREE.Euler(Math.PI / 2.0, 0.0, 0.0)}
              position={[0.0, 0.0, 0.5 * axesLength]}
              color={0x0000cc}
            >
              <OutlinesIfHovered />
            </Instance>
          </Instances>
        </>
      )}
      {children}
    </group>
  );
});

/** Helper for adding batched/instanced coordinate frames as scene nodes. */
export const InstancedAxes = React.forwardRef<
  THREE.Group,
  {
    /** Raw bytes containing float32 quaternion values (wxyz) */
    batched_wxyzs: Uint8Array;
    /** Raw bytes containing float32 position values (xyz) */
    batched_positions: Uint8Array;
    /** Raw bytes containing float32 scale values (uniform or per-axis XYZ) */
    batched_scales: Uint8Array | null;
    axes_length?: number;
    axes_radius?: number;
    children?: React.ReactNode;
  }
>(function InstancedAxes(
  {
    batched_wxyzs,
    batched_positions,
    batched_scales,
    axes_length = 0.5,
    axes_radius = 0.0125,
    children,
  },
  ref,
) {
  const axesRef = React.useRef<THREE.InstancedMesh>(null);

  // Create geometry and material using useMemo.
  const cylinderGeom = React.useMemo(
    () => new THREE.CylinderGeometry(axes_radius, axes_radius, axes_length, 16),
    [axes_radius, axes_length],
  );

  const material = React.useMemo(() => new MeshBasicMaterial(), []);

  // Dispose resources when component unmounts.
  React.useEffect(() => {
    return () => {
      cylinderGeom.dispose();
      material.dispose();
    };
  }, [cylinderGeom, material]);

  // Pre-compute transformation matrices for axes using useMemo.
  const axesTransformations = React.useMemo(() => {
    return {
      T_frame_framex: new THREE.Matrix4()
        .makeRotationFromEuler(new THREE.Euler(0.0, 0.0, (3.0 * Math.PI) / 2.0))
        .setPosition(0.5 * axes_length, 0.0, 0.0),
      T_frame_framey: new THREE.Matrix4()
        .makeRotationFromEuler(new THREE.Euler(0.0, 0.0, 0.0))
        .setPosition(0.0, 0.5 * axes_length, 0.0),
      T_frame_framez: new THREE.Matrix4()
        .makeRotationFromEuler(new THREE.Euler(Math.PI / 2.0, 0.0, 0.0))
        .setPosition(0.0, 0.0, 0.5 * axes_length),
      red: new THREE.Color(0xcc0000),
      green: new THREE.Color(0x00cc00),
      blue: new THREE.Color(0x0000cc),
    };
  }, [axes_length]);

  // Update instance matrices and colors.
  React.useEffect(() => {
    if (!axesRef.current) return;

    // Pre-allocate to avoid garbage collector from running during loop.
    const T_world_frame = new THREE.Matrix4();
    const T_world_framex = new THREE.Matrix4();
    const T_world_framey = new THREE.Matrix4();
    const T_world_framez = new THREE.Matrix4();
    const tmpQuat = new THREE.Quaternion();
    const tmpScale = new THREE.Vector3();

    const { T_frame_framex, T_frame_framey, T_frame_framez, red, green, blue } =
      axesTransformations;

    // Create DataViews to read float values directly.
    const positionsView = new DataView(
      batched_positions.buffer,
      batched_positions.byteOffset,
      batched_positions.byteLength,
    );

    const wxyzsView = new DataView(
      batched_wxyzs.buffer,
      batched_wxyzs.byteOffset,
      batched_wxyzs.byteLength,
    );

    const scalesView = batched_scales
      ? new DataView(
          batched_scales.buffer,
          batched_scales.byteOffset,
          batched_scales.byteLength,
        )
      : null;

    // Calculate number of instances.
    const numInstances = batched_wxyzs.byteLength / (4 * 4); // 4 floats, 4 bytes per float

    for (let i = 0; i < numInstances; i++) {
      // Calculate byte offsets for reading float values.
      const posOffset = i * 3 * 4; // 3 floats, 4 bytes per float
      const wxyzOffset = i * 4 * 4; // 4 floats, 4 bytes per float
      const scaleOffset =
        batched_scales &&
        batched_scales.byteLength === (batched_wxyzs.byteLength / 4) * 3
          ? i * 3 * 4 // Per-axis scaling: 3 floats, 4 bytes per float
          : i * 4; // Uniform scaling: 1 float, 4 bytes per float

      // Read scale value if available.
      if (scalesView) {
        // Check if we have per-axis scaling (N,3) or uniform scaling (N,).
        if (batched_scales!.byteLength === (batched_wxyzs.byteLength / 4) * 3) {
          // Per-axis scaling: read 3 floats.
          tmpScale.set(
            scalesView.getFloat32(scaleOffset, true), // x scale
            scalesView.getFloat32(scaleOffset + 4, true), // y scale
            scalesView.getFloat32(scaleOffset + 8, true), // z scale
          );
        } else {
          // Uniform scaling: read 1 float and apply to all axes.
          const scale = scalesView.getFloat32(scaleOffset, true);
          tmpScale.set(scale, scale, scale);
        }
      } else {
        tmpScale.set(1, 1, 1);
      }

      // Set position from DataView.
      T_world_frame.makeRotationFromQuaternion(
        tmpQuat.set(
          wxyzsView.getFloat32(wxyzOffset + 4, true), // x
          wxyzsView.getFloat32(wxyzOffset + 8, true), // y
          wxyzsView.getFloat32(wxyzOffset + 12, true), // z
          wxyzsView.getFloat32(wxyzOffset, true), // w (first value)
        ),
      )
        .scale(tmpScale)
        .setPosition(
          positionsView.getFloat32(posOffset, true), // x
          positionsView.getFloat32(posOffset + 4, true), // y
          positionsView.getFloat32(posOffset + 8, true), // z
        );

      T_world_framex.copy(T_world_frame).multiply(T_frame_framex);
      T_world_framey.copy(T_world_frame).multiply(T_frame_framey);
      T_world_framez.copy(T_world_frame).multiply(T_frame_framez);

      axesRef.current.setMatrixAt(i * 3 + 0, T_world_framex);
      axesRef.current.setMatrixAt(i * 3 + 1, T_world_framey);
      axesRef.current.setMatrixAt(i * 3 + 2, T_world_framez);

      axesRef.current.setColorAt(i * 3 + 0, red);
      axesRef.current.setColorAt(i * 3 + 1, green);
      axesRef.current.setColorAt(i * 3 + 2, blue);
    }
    axesRef.current.instanceMatrix.needsUpdate = true;
    axesRef.current.instanceColor!.needsUpdate = true;
  }, [batched_wxyzs, batched_positions, batched_scales, axesTransformations]);

  // Create cylinder geometries for outlines - one for each axis.
  const outlineCylinderGeom = React.useMemo(
    () => new THREE.CylinderGeometry(axes_radius, axes_radius, axes_length, 16),
    [axes_radius, axes_length],
  );

  // Compute transform matrices for each axis.
  const xAxisTransform = React.useMemo(
    () => ({
      position: new THREE.Vector3(0.5 * axes_length, 0, 0),
      rotation: new THREE.Quaternion().setFromEuler(
        new THREE.Euler(0, 0, (3 * Math.PI) / 2),
      ),
      scale: new THREE.Vector3(1, 1, 1),
    }),
    [axes_length],
  );

  const yAxisTransform = React.useMemo(
    () => ({
      position: new THREE.Vector3(0, 0.5 * axes_length, 0),
      rotation: new THREE.Quaternion().setFromEuler(new THREE.Euler(0, 0, 0)),
      scale: new THREE.Vector3(1, 1, 1),
    }),
    [axes_length],
  );

  const zAxisTransform = React.useMemo(
    () => ({
      position: new THREE.Vector3(0, 0, 0.5 * axes_length),
      rotation: new THREE.Quaternion().setFromEuler(
        new THREE.Euler(Math.PI / 2, 0, 0),
      ),
      scale: new THREE.Vector3(1, 1, 1),
    }),
    [axes_length],
  );

  // Calculate number of instances for args.
  const numInstances = (batched_wxyzs.byteLength / (4 * 4)) * 3; // 4 floats per WXYZ * 4 bytes per float * 3 axes

  return (
    <group ref={ref}>
      <instancedMesh
        ref={axesRef}
        args={[cylinderGeom, material, numInstances]}
      />

      {/* Create hover outlines for each axis */}
      <BatchedMeshHoverOutlines
        geometry={outlineCylinderGeom}
        batched_positions={batched_positions}
        batched_wxyzs={batched_wxyzs}
        batched_scales={batched_scales}
        meshTransform={xAxisTransform}
        computeBatchIndexFromInstanceIndex={(instanceId) =>
          Math.floor(instanceId / 3)
        }
      />

      <BatchedMeshHoverOutlines
        geometry={outlineCylinderGeom}
        batched_positions={batched_positions}
        batched_wxyzs={batched_wxyzs}
        batched_scales={batched_scales}
        meshTransform={yAxisTransform}
        computeBatchIndexFromInstanceIndex={(instanceId) =>
          Math.floor(instanceId / 3)
        }
      />

      <BatchedMeshHoverOutlines
        geometry={outlineCylinderGeom}
        batched_positions={batched_positions}
        batched_wxyzs={batched_wxyzs}
        batched_scales={batched_scales}
        meshTransform={zAxisTransform}
        computeBatchIndexFromInstanceIndex={(instanceId) =>
          Math.floor(instanceId / 3)
        }
      />
      {children}
    </group>
  );
});

export const ViserImage = React.forwardRef<
  THREE.Group,
  ImageMessage & { children?: React.ReactNode }
>(function ViserImage({ children, ...message }, ref) {
  // We can't use useMemo here because TextureLoader.load is asynchronous.
  // And we need to use setState to update the texture after loading.
  const [imageTexture, setImageTexture] = React.useState<THREE.Texture>();

  React.useEffect(() => {
    if (message.props.format !== null && message.props._data !== null) {
      const image_url = URL.createObjectURL(new Blob([message.props._data]), {
        type: "image/" + message.props.format,
      });
      new THREE.TextureLoader().load(image_url, (texture) => {
        setImageTexture(texture);
        URL.revokeObjectURL(image_url);
      });
    }
  }, [message.props.format, message.props._data]);
  return (
    <group ref={ref}>
      <mesh
        rotation={new THREE.Euler(Math.PI, 0.0, 0.0)}
        castShadow={message.props.cast_shadow}
        receiveShadow={message.props.receive_shadow}
      >
        <OutlinesIfHovered />
        <planeGeometry
          attach="geometry"
          args={[message.props.render_width, message.props.render_height]}
        />
        <meshBasicMaterial
          attach="material"
          transparent={true}
          side={THREE.DoubleSide}
          map={imageTexture}
          toneMapped={false}
        />
      </mesh>
      {children}
    </group>
  );
});
