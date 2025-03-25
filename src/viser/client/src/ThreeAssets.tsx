import { Instance, Instances, Line, shaderMaterial } from "@react-three/drei";
import { useFrame, useThree } from "@react-three/fiber";
import { OutlinesIfHovered } from "./OutlinesIfHovered";
import React from "react";
import { HoverableContext } from "./HoverContext";
import * as THREE from "three";
import {
  CameraFrustumMessage,
  ImageMessage,
  PointCloudMessage,
} from "./WebsocketMessages";
import { rgbToInt } from "./mesh/MeshUtils";
import { ShaderMaterial, MeshBasicMaterial } from "three";

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
  uniform vec3 uniformColor;

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

export const PointCloud = React.forwardRef<THREE.Points, PointCloudMessage>(
  function PointCloud(message, ref) {
    const getThreeState = useThree((state) => state.get);

    const props = message.props;

    const [geometry, setGeometry] = React.useState<THREE.BufferGeometry>();
    const [material, setMaterial] = React.useState<ShaderMaterial>();

    React.useEffect(() => {
      const geometry = new THREE.BufferGeometry();

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

      const material = new PointCloudMaterial();
      if (props.colors.length > 3) {
        material.vertexColors = true;
        geometry.setAttribute(
          "color",
          new THREE.BufferAttribute(new Uint8Array(props.colors), 3, true),
        );
      } else {
        if (props.colors.length < 3) {
          console.error(
            `Invalid color buffer length, got ${props.colors.length}`,
          );
        }
        material.vertexColors = false;
        material.uniforms.uniformColor.value = new THREE.Color(
          props.colors[0],
          props.colors[1],
          props.colors[2],
        );
      }
      setGeometry(geometry);
      setMaterial(material);
    }, [props.points, props.colors]);

    React.useEffect(() => {
      return () => {
        if (geometry === undefined) return;
        geometry.dispose();
      };
    }, [geometry]);

    React.useEffect(() => {
      return () => {
        if (material === undefined) return;
        material.dispose();
      };
    }, [material]);

    React.useEffect(() => {
      if (material === undefined) return;
      material.uniforms.scale.value = 10.0;
      material.uniforms.point_ball_norm.value = props.point_ball_norm;
    }, [props.point_ball_norm, material]);

    const rendererSize = new THREE.Vector2();
    useFrame(() => {
      // Match point scale to behavior of THREE.PointsMaterial().
      if (material === undefined) return;
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
    return <points ref={ref} geometry={geometry} material={material} />;
  },
);

/** Helper for adding coordinate frames as scene nodes. */
export const CoordinateFrame = React.forwardRef<
  THREE.Group,
  {
    showAxes?: boolean;
    axesLength?: number;
    axesRadius?: number;
    originRadius?: number;
    originColor?: number;
  }
>(function CoordinateFrame(
  {
    showAxes = true,
    axesLength = 0.5,
    axesRadius = 0.0125,
    originRadius = undefined,
    originColor = 0xecec00,
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
    </group>
  );
});

/** Helper for adding batched/instanced coordinate frames as scene nodes. */
export const InstancedAxes = React.forwardRef<
  THREE.Group,
  {
    batched_wxyzs: Float32Array;
    batched_positions: Float32Array;
    axes_length?: number;
    axes_radius?: number;
  }
>(function InstancedAxes(
  { batched_wxyzs, batched_positions, axes_length = 0.5, axes_radius = 0.0125 },
  ref,
) {
  const axesRef = React.useRef<THREE.InstancedMesh>(null);

  const cylinderGeom = new THREE.CylinderGeometry(
    axes_radius,
    axes_radius,
    axes_length,
    16,
  );
  const material = new MeshBasicMaterial();

  // Dispose when done.
  React.useEffect(() => {
    return () => {
      cylinderGeom.dispose();
      material.dispose();
    };
  });

  // Update instance matrices and colors.
  React.useEffect(() => {
    // Pre-allocate to avoid garbage collector from running during loop.
    const T_world_frame = new THREE.Matrix4();
    const T_world_framex = new THREE.Matrix4();
    const T_world_framey = new THREE.Matrix4();
    const T_world_framez = new THREE.Matrix4();

    const T_frame_framex = new THREE.Matrix4()
      .makeRotationFromEuler(new THREE.Euler(0.0, 0.0, (3.0 * Math.PI) / 2.0))
      .setPosition(0.5 * axes_length, 0.0, 0.0);
    const T_frame_framey = new THREE.Matrix4()
      .makeRotationFromEuler(new THREE.Euler(0.0, 0.0, 0.0))
      .setPosition(0.0, 0.5 * axes_length, 0.0);
    const T_frame_framez = new THREE.Matrix4()
      .makeRotationFromEuler(new THREE.Euler(Math.PI / 2.0, 0.0, 0.0))
      .setPosition(0.0, 0.0, 0.5 * axes_length);

    const tmpQuat = new THREE.Quaternion();

    const red = new THREE.Color(0xcc0000);
    const green = new THREE.Color(0x00cc00);
    const blue = new THREE.Color(0x0000cc);

    for (let i = 0; i < batched_wxyzs.length / 4; i++) {
      T_world_frame.makeRotationFromQuaternion(
        tmpQuat.set(
          batched_wxyzs[i * 4 + 1],
          batched_wxyzs[i * 4 + 2],
          batched_wxyzs[i * 4 + 3],
          batched_wxyzs[i * 4 + 0],
        ),
      ).setPosition(
        batched_positions[i * 3 + 0],
        batched_positions[i * 3 + 1],
        batched_positions[i * 3 + 2],
      );
      T_world_framex.copy(T_world_frame).multiply(T_frame_framex);
      T_world_framey.copy(T_world_frame).multiply(T_frame_framey);
      T_world_framez.copy(T_world_frame).multiply(T_frame_framez);

      axesRef.current!.setMatrixAt(i * 3 + 0, T_world_framex);
      axesRef.current!.setMatrixAt(i * 3 + 1, T_world_framey);
      axesRef.current!.setMatrixAt(i * 3 + 2, T_world_framez);

      axesRef.current!.setColorAt(i * 3 + 0, red);
      axesRef.current!.setColorAt(i * 3 + 1, green);
      axesRef.current!.setColorAt(i * 3 + 2, blue);
    }
    axesRef.current!.instanceMatrix.needsUpdate = true;
    axesRef.current!.instanceColor!.needsUpdate = true;
  }, [batched_wxyzs, batched_positions]);

  return (
    <group ref={ref}>
      <instancedMesh
        ref={axesRef}
        args={[cylinderGeom, material, (batched_wxyzs.length / 4) * 3]}
      >
        <OutlinesIfHovered />
      </instancedMesh>
    </group>
  );
});

export const ViserImage = React.forwardRef<THREE.Group, ImageMessage>(
  function ViserImage(message, ref) {
    const [imageTexture, setImageTexture] = React.useState<THREE.Texture>();

    React.useEffect(() => {
      if (message.props.media_type !== null && message.props._data !== null) {
        const image_url = URL.createObjectURL(new Blob([message.props._data]));
        new THREE.TextureLoader().load(image_url, (texture) => {
          setImageTexture(texture);
          URL.revokeObjectURL(image_url);
        });
      }
    }, [message.props.media_type, message.props._data]);
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
      </group>
    );
  },
);

/** Helper for visualizing camera frustums. */
export const CameraFrustum = React.forwardRef<
  THREE.Group,
  CameraFrustumMessage
>(function CameraFrustum(message, ref) {
  const [imageTexture, setImageTexture] = React.useState<THREE.Texture>();

  React.useEffect(() => {
    if (
      message.props.image_media_type !== null &&
      message.props._image_data !== null
    ) {
      const image_url = URL.createObjectURL(
        new Blob([message.props._image_data]),
      );
      new THREE.TextureLoader().load(image_url, (texture) => {
        setImageTexture(texture);
        URL.revokeObjectURL(image_url);
      });
    } else {
      setImageTexture(undefined);
    }
  }, [message.props.image_media_type, message.props._image_data]);

  let y = Math.tan(message.props.fov / 2.0);
  let x = y * message.props.aspect;
  let z = 1.0;

  const volumeScale = Math.cbrt((x * y * z) / 3.0);
  x /= volumeScale;
  y /= volumeScale;
  z /= volumeScale;
  x *= message.props.scale;
  y *= message.props.scale;
  z *= message.props.scale;

  const hoveredRef = React.useContext(HoverableContext);
  const [isHovered, setIsHovered] = React.useState(false);

  useFrame(() => {
    if (hoveredRef !== null && hoveredRef.current !== isHovered) {
      setIsHovered(hoveredRef.current);
    }
  });

  const frustumPoints: [number, number, number][] = [
    // Rectangle.
    [-1, -1, 1],
    [1, -1, 1],
    [1, -1, 1],
    [1, 1, 1],
    [1, 1, 1],
    [-1, 1, 1],
    [-1, 1, 1],
    [-1, -1, 1],
    // Lines to origin.
    [-1, -1, 1],
    [0, 0, 0],
    [0, 0, 0],
    [1, -1, 1],
    // Lines to origin.
    [-1, 1, 1],
    [0, 0, 0],
    [0, 0, 0],
    [1, 1, 1],
    // Up direction indicator.
    // Don't overlap with the image if the image is present.
    [0.0, -1.2, 1.0],
    imageTexture === undefined ? [0.0, -0.9, 1.0] : [0.0, -1.0, 1.0],
  ].map((xyz) => [xyz[0] * x, xyz[1] * y, xyz[2] * z]);

  return (
    <group ref={ref}>
      <Line
        points={frustumPoints}
        color={isHovered ? 0xfbff00 : rgbToInt(message.props.color)}
        lineWidth={
          isHovered ? 1.5 * message.props.line_width : message.props.line_width
        }
        segments
      />
      {imageTexture && (
        <mesh
          // 0.999999 is to avoid z-fighting with the frustum lines.
          position={[0.0, 0.0, z * 0.999999]}
          rotation={new THREE.Euler(Math.PI, 0.0, 0.0)}
          castShadow={message.props.cast_shadow}
          receiveShadow={message.props.receive_shadow}
        >
          <planeGeometry
            attach="geometry"
            args={[message.props.aspect * y * 2, y * 2]}
          />
          <meshBasicMaterial
            attach="material"
            transparent={true}
            side={THREE.DoubleSide}
            map={imageTexture}
            toneMapped={false}
          />
        </mesh>
      )}
    </group>
  );
});
