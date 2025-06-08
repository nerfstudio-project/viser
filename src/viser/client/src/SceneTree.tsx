import {
  CatmullRomLine,
  CubicBezierLine,
  Grid,
  PivotControls,
} from "@react-three/drei";
import { ContextBridge, useContextBridge } from "its-fine";
import { useFrame } from "@react-three/fiber";
import React from "react";
import * as THREE from "three";

import { ViewerContext, ViewerContextContents } from "./ViewerContext";
import {
  makeThrottledMessageSender,
  useThrottledMessageSender,
} from "./WebsocketUtils";
import { Html } from "@react-three/drei";
import { useSceneTreeState } from "./SceneTreeState";
import { rayToViserCoords } from "./WorldTransformUtils";
import { HoverableContext, HoverState } from "./HoverContext";

/** Turn a click event to normalized OpenCV coordinate (NDC) vector.
 * Normalizes click coordinates to be between (0, 0) as upper-left corner,
 * and (1, 1) as lower-right corner, with (0.5, 0.5) being the center of the screen.
 * Uses offsetX/Y, and clientWidth/Height to get the coordinates.
 */
function opencvXyFromPointerXy(
  viewer: ViewerContextContents,
  xy: [number, number],
): THREE.Vector2 {
  const mouseVector = new THREE.Vector2();
  mouseVector.x = (xy[0] + 0.5) / viewer.mutable.current.canvas!.clientWidth;
  mouseVector.y = (xy[1] + 0.5) / viewer.mutable.current.canvas!.clientHeight;
  return mouseVector;
}
import {
  CameraFrustum,
  CoordinateFrame,
  InstancedAxes,
  PointCloud,
  ViserImage,
} from "./ThreeAssets";
import { SceneNodeMessage } from "./WebsocketMessages";
import { SplatObject } from "./Splatting/GaussianSplats";
import { Paper } from "@mantine/core";
import GeneratedGuiContainer from "./ControlPanel/Generated";
import { Line } from "./Line";
import { shadowArgs } from "./ShadowArgs";
import { CsmDirectionalLight } from "./CsmDirectionalLight";
import { BasicMesh } from "./mesh/BasicMesh";
import { BoxMesh } from "./mesh/BoxMesh";
import { IcosphereMesh } from "./mesh/IcosphereMesh";
import { SkinnedMesh } from "./mesh/SkinnedMesh";
import { BatchedMesh } from "./mesh/BatchedMesh";
import { SingleGlbAsset } from "./mesh/SingleGlbAsset";
import { BatchedGlbAsset } from "./mesh/BatchedGlbAsset";

function rgbToInt(rgb: [number, number, number]): number {
  return (rgb[0] << 16) | (rgb[1] << 8) | rgb[2];
}

/** Type corresponding to a zustand-style useSceneTree hook. */
export type UseSceneTree = ReturnType<typeof useSceneTreeState>;

/** Component for updating attributes of a scene node. */
function SceneNodeLabel(props: { name: string }) {
  const viewer = React.useContext(ViewerContext)!;
  const labelVisible = viewer.useSceneTree(
    (state) => state.labelVisibleFromName[props.name],
  );
  return labelVisible ? (
    <Html>
      <span
        style={{
          backgroundColor: "rgba(240, 240, 240, 0.9)",
          borderRadius: "0.2rem",
          userSelect: "none",
          padding: "0.1em 0.2em",
        }}
      >
        {props.name}
      </span>
    </Html>
  ) : null;
}

export type MakeObject = (
  ref: React.Ref<any>,
  children: React.ReactNode,
) => React.ReactNode;

function createObjectFactory(
  message: SceneNodeMessage | undefined,
  viewer: ViewerContextContents,
  ContextBridge: ContextBridge,
): {
  makeObject: MakeObject;
  unmountWhenInvisible?: boolean;
  computeClickInstanceIndexFromInstanceId?: (
    instanceId: number | undefined,
  ) => number | null;
} {
  if (message === undefined) return { makeObject: () => null };

  switch (message.type) {
    // Add a coordinate frame.
    case "FrameMessage": {
      return {
        makeObject: (ref, children) => (
          <CoordinateFrame
            ref={ref}
            showAxes={message.props.show_axes}
            axesLength={message.props.axes_length}
            axesRadius={message.props.axes_radius}
            originRadius={message.props.origin_radius}
            originColor={rgbToInt(message.props.origin_color)}
          >
            {children}
          </CoordinateFrame>
        ),
      };
    }

    // Add axes to visualize.
    case "BatchedAxesMessage": {
      return {
        makeObject: (ref, children) => (
          <InstancedAxes
            ref={ref}
            batched_wxyzs={message.props.batched_wxyzs}
            batched_positions={message.props.batched_positions}
            batched_scales={message.props.batched_scales}
            axes_length={message.props.axes_length}
            axes_radius={message.props.axes_radius}
          >
            {children}
          </InstancedAxes>
        ),
        // Compute click instance index from instance ID. Each visualized
        // frame has 1 instance for each of 3 line segments.
        computeClickInstanceIndexFromInstanceId: (instanceId) =>
          Math.floor(instanceId! / 3),
      };
    }

    case "GridMessage": {
      const gridQuaternion = new THREE.Quaternion().setFromEuler(
        // There's redundancy here when we set the side to
        // THREE.DoubleSide, where xy and yx should be the same.
        //
        // But it makes sense to keep this parameterization because
        // specifying planes by xy seems more natural than the normal
        // direction (z, +z, or -z), and it opens the possibility of
        // rendering only FrontSide or BackSide grids in the future.
        //
        // If we add support for FrontSide or BackSide, we should
        // double-check that the normal directions from each of these
        // rotations match the right-hand rule!
        message.props.plane == "xz"
          ? new THREE.Euler(0.0, 0.0, 0.0)
          : message.props.plane == "xy"
            ? new THREE.Euler(Math.PI / 2.0, 0.0, 0.0)
            : message.props.plane == "yx"
              ? new THREE.Euler(0.0, Math.PI / 2.0, Math.PI / 2.0)
              : message.props.plane == "yz"
                ? new THREE.Euler(0.0, 0.0, Math.PI / 2.0)
                : message.props.plane == "zx"
                  ? new THREE.Euler(0.0, Math.PI / 2.0, 0.0)
                  : //message.props.plane == "zy"
                    new THREE.Euler(-Math.PI / 2.0, 0.0, -Math.PI / 2.0),
      );

      // When rotations are identity: plane is XY, while grid is XZ.
      const planeQuaternion = new THREE.Quaternion()
        .setFromEuler(new THREE.Euler(-Math.PI / 2, 0.0, 0.0))
        .premultiply(gridQuaternion);

      let shadowPlane;
      if (message.props.shadow_opacity > 0.0) {
        shadowPlane = (
          <mesh
            receiveShadow
            position={[0.0, 0.0, -0.01]}
            quaternion={planeQuaternion}
          >
            <planeGeometry args={[message.props.width, message.props.height]} />
            <shadowMaterial
              opacity={message.props.shadow_opacity}
              color={0x000000}
              depthWrite={false}
            />
          </mesh>
        );
      } else {
        // when opacity = 0.0, no shadowPlane for performance
        shadowPlane = null;
      }
      return {
        makeObject: (ref, children) => (
          <group ref={ref}>
            <Grid
              args={[
                message.props.width,
                message.props.height,
                message.props.width_segments,
                message.props.height_segments,
              ]}
              side={THREE.DoubleSide}
              cellColor={rgbToInt(message.props.cell_color)}
              cellThickness={message.props.cell_thickness}
              cellSize={message.props.cell_size}
              sectionColor={rgbToInt(message.props.section_color)}
              sectionThickness={message.props.section_thickness}
              sectionSize={message.props.section_size}
              quaternion={gridQuaternion}
            />
            {shadowPlane}
            {children}
          </group>
        ),
      };
    }

    // Add a point cloud.
    case "PointCloudMessage": {
      return {
        makeObject: (ref, children) => (
          <PointCloud ref={ref} {...message}>
            {children}
          </PointCloud>
        ),
      };
    }

    // Add mesh
    case "SkinnedMeshMessage": {
      return {
        makeObject: (ref, children) => (
          <SkinnedMesh ref={ref} {...message}>
            {children}
          </SkinnedMesh>
        ),
      };
    }
    case "MeshMessage": {
      return {
        makeObject: (ref, children) => (
          <BasicMesh ref={ref} {...message}>
            {children}
          </BasicMesh>
        ),
      };
    }
    case "BoxMessage": {
      return {
        makeObject: (ref, children) => (
          <BoxMesh ref={ref} {...message}>
            {children}
          </BoxMesh>
        ),
      };
    }
    case "IcosphereMessage": {
      return {
        makeObject: (ref, children) => (
          <IcosphereMesh ref={ref} {...message}>
            {children}
          </IcosphereMesh>
        ),
      };
    }
    case "BatchedMeshesMessage": {
      return {
        makeObject: (ref, children) => (
          <BatchedMesh ref={ref} {...message}>
            {children}
          </BatchedMesh>
        ),
        computeClickInstanceIndexFromInstanceId:
          message.type === "BatchedMeshesMessage"
            ? (instanceId) => instanceId!
            : undefined,
      };
    }
    // Add a camera frustum.
    case "CameraFrustumMessage": {
      return {
        makeObject: (ref, children) => (
          <CameraFrustum ref={ref} {...message}>
            {children}
          </CameraFrustum>
        ),
      };
    }
    case "TransformControlsMessage": {
      const name = message.name;
      const sendDragMessage = makeThrottledMessageSender(viewer, 50);
      return {
        makeObject: (ref, children) => (
          <PivotControls
            ref={ref}
            scale={message.props.scale}
            lineWidth={message.props.line_width}
            fixed={message.props.fixed}
            activeAxes={message.props.active_axes}
            disableAxes={message.props.disable_axes}
            disableSliders={message.props.disable_sliders}
            disableRotations={message.props.disable_rotations}
            disableScaling={true}
            translationLimits={message.props.translation_limits}
            rotationLimits={message.props.rotation_limits}
            depthTest={message.props.depth_test}
            opacity={message.props.opacity}
            onDrag={(l) => {
              const attrs = viewer.mutable.current.nodeAttributesFromName;
              if (attrs[message.name] === undefined) {
                attrs[message.name] = {};
              }

              const wxyz = new THREE.Quaternion();
              wxyz.setFromRotationMatrix(l);
              const position = new THREE.Vector3().setFromMatrixPosition(l);

              const nodeAttributes = attrs[message.name]!;
              nodeAttributes.wxyz = [wxyz.w, wxyz.x, wxyz.y, wxyz.z];
              nodeAttributes.position = position.toArray();
              sendDragMessage({
                type: "TransformControlsUpdateMessage",
                name: name,
                wxyz: nodeAttributes.wxyz,
                position: nodeAttributes.position,
              });
            }}
          >
            {children}
          </PivotControls>
        ),
        unmountWhenInvisible: true,
      };
    }
    // Add a 2D label.
    case "LabelMessage": {
      return {
        makeObject: (ref, children) => (
          // We wrap with <group /> because Html doesn't implement THREE.Object3D.
          <group ref={ref}>
            <Html>
              <div
                style={{
                  width: "10em",
                  fontSize: "0.8em",
                  transform: "translateX(0.1em) translateY(0.5em)",
                }}
              >
                <span
                  style={{
                    background: "#fff",
                    border: "1px solid #777",
                    borderRadius: "0.2em",
                    color: "#333",
                    padding: "0.2em",
                  }}
                >
                  {message.props.text}
                </span>
              </div>
            </Html>
            {children}
          </group>
        ),
        unmountWhenInvisible: true,
      };
    }
    case "Gui3DMessage": {
      return {
        makeObject: (ref, children) => {
          // We wrap with <group /> because Html doesn't implement
          // THREE.Object3D.
          return (
            <group ref={ref}>
              <Html>
                <ContextBridge>
                  <Paper
                    style={{
                      width: "18em",
                      fontSize: "0.875em",
                      marginLeft: "0.5em",
                      marginTop: "0.5em",
                    }}
                    shadow="0 0 0.8em 0 rgba(0,0,0,0.1)"
                    pb="0.25em"
                    onPointerDown={(evt) => {
                      evt.stopPropagation();
                    }}
                  >
                    <GeneratedGuiContainer
                      containerUuid={message.props.container_uuid}
                    />
                  </Paper>
                </ContextBridge>
              </Html>
              {children}
            </group>
          );
        },
        unmountWhenInvisible: true,
      };
    }
    // Add an image.
    case "ImageMessage": {
      return {
        makeObject: (ref, children) => (
          <ViserImage ref={ref} {...message}>
            {children}
          </ViserImage>
        ),
      };
    }
    // Add a glTF/GLB asset.
    case "GlbMessage": {
      return {
        makeObject: (ref, children) => (
          <SingleGlbAsset ref={ref} {...message}>
            {children}
          </SingleGlbAsset>
        ),
      };
    }
    case "BatchedGlbMessage": {
      return {
        makeObject: (ref, children) => (
          <BatchedGlbAsset ref={ref} {...message}>
            {children}
          </BatchedGlbAsset>
        ),
        computeClickInstanceIndexFromInstanceId: (instanceId) => instanceId!,
      };
    }
    case "LineSegmentsMessage": {
      return {
        makeObject: (ref, children) => {
          // The array conversion here isn't very efficient. We go from buffer
          // => TypeArray => Javascript Array, then back to buffers in drei's
          // <Line /> abstraction.
          const pointsArray = new Float32Array(
            message.props.points.buffer.slice(
              message.props.points.byteOffset,
              message.props.points.byteOffset + message.props.points.byteLength,
            ),
          );
          const colorArray = new Uint8Array(
            message.props.colors.buffer.slice(
              message.props.colors.byteOffset,
              message.props.colors.byteOffset + message.props.colors.byteLength,
            ),
          );
          return (
            <group ref={ref}>
              <Line
                points={pointsArray}
                lineWidth={message.props.line_width}
                vertexColors={colorArray}
                segments={true}
              />
              {children}
            </group>
          );
        },
      };
    }
    case "CatmullRomSplineMessage": {
      return {
        makeObject: (ref, children) => {
          return (
            <group ref={ref}>
              <CatmullRomLine
                points={message.props.positions}
                closed={message.props.closed}
                curveType={message.props.curve_type}
                tension={message.props.tension}
                lineWidth={message.props.line_width}
                color={rgbToInt(message.props.color)}
                // Sketchy cast needed due to https://github.com/pmndrs/drei/issues/1476.
                segments={(message.props.segments ?? undefined) as undefined}
              />
              {children}
            </group>
          );
        },
      };
    }
    case "CubicBezierSplineMessage": {
      return {
        makeObject: (ref, children) => (
          <group ref={ref}>
            {[...Array(message.props.positions.length - 1).keys()].map((i) => (
              <CubicBezierLine
                key={i}
                start={message.props.positions[i]}
                end={message.props.positions[i + 1]}
                midA={message.props.control_points[2 * i]}
                midB={message.props.control_points[2 * i + 1]}
                lineWidth={message.props.line_width}
                color={rgbToInt(message.props.color)}
                // Sketchy cast needed due to https://github.com/pmndrs/drei/issues/1476.
                segments={(message.props.segments ?? undefined) as undefined}
              ></CubicBezierLine>
            ))}
            {children}
          </group>
        ),
      };
    }
    case "GaussianSplatsMessage": {
      return {
        makeObject: (ref, children) => (
          <SplatObject
            ref={ref}
            buffer={
              new Uint32Array(
                message.props.buffer.buffer.slice(
                  message.props.buffer.byteOffset,
                  message.props.buffer.byteOffset +
                    message.props.buffer.byteLength,
                ),
              )
            }
          >
            {children}
          </SplatObject>
        ),
      };
    }

    // Add a directional light
    case "DirectionalLightMessage": {
      return {
        makeObject: (ref, children) => (
          <group ref={ref}>
            <CsmDirectionalLight
              lightIntensity={message.props.intensity}
              color={rgbToInt(message.props.color)}
              castShadow={message.props.cast_shadow}
            />
            {children}
          </group>
        ),
        // CsmDirectionalLight is not influenced by visibility, since the
        // lights it adds are portaled to the scene root.
        unmountWhenInvisible: true,
      };
    }

    // Add an ambient light
    // Cannot cast shadows
    case "AmbientLightMessage": {
      return {
        makeObject: (ref, children) => (
          <ambientLight
            ref={ref}
            intensity={message.props.intensity}
            color={rgbToInt(message.props.color)}
          >
            {children}
          </ambientLight>
        ),
      };
    }

    // Add a hemisphere light
    // Cannot cast shadows
    case "HemisphereLightMessage": {
      return {
        makeObject: (ref, children) => (
          <hemisphereLight
            ref={ref}
            intensity={message.props.intensity}
            color={rgbToInt(message.props.sky_color)}
            groundColor={rgbToInt(message.props.ground_color)}
          >
            {children}
          </hemisphereLight>
        ),
      };
    }

    // Add a point light
    case "PointLightMessage": {
      return {
        makeObject: (ref, children) => (
          <pointLight
            ref={ref}
            intensity={message.props.intensity}
            color={rgbToInt(message.props.color)}
            distance={message.props.distance}
            decay={message.props.decay}
            castShadow={message.props.cast_shadow}
            {...shadowArgs}
          >
            {children}
          </pointLight>
        ),
      };
    }
    // Add a rectangular area light
    // Cannot cast shadows
    case "RectAreaLightMessage": {
      return {
        makeObject: (ref, children) => (
          <rectAreaLight
            ref={ref}
            intensity={message.props.intensity}
            color={rgbToInt(message.props.color)}
            width={message.props.width}
            height={message.props.height}
          >
            {children}
          </rectAreaLight>
        ),
      };
    }

    // Add a spot light
    case "SpotLightMessage": {
      return {
        makeObject: (ref, children) => (
          <spotLight
            ref={ref}
            intensity={message.props.intensity}
            color={rgbToInt(message.props.color)}
            distance={message.props.distance}
            angle={message.props.angle}
            penumbra={message.props.penumbra}
            decay={message.props.decay}
            castShadow={message.props.cast_shadow}
            {...shadowArgs}
          >
            {children}
          </spotLight>
        ),
      };
    }
    default: {
      console.log("Received message did not match any known types:", message);
      return { makeObject: () => null };
    }
  }
}

export function SceneNodeThreeObject(props: {
  name: string;
  parent: THREE.Object3D | null;
}) {
  const viewer = React.useContext(ViewerContext)!;
  const message = viewer.useSceneTree(
    (state) => state.nodeFromName[props.name]?.message,
  );
  const ContextBridge = useContextBridge();

  const {
    makeObject,
    unmountWhenInvisible,
    computeClickInstanceIndexFromInstanceId,
  } = React.useMemo(
    () => createObjectFactory(message, viewer, ContextBridge),
    [message, viewer, ContextBridge],
  );

  const [unmount, setUnmount] = React.useState(false);
  const clickable =
    viewer.useSceneTree((state) => state.nodeFromName[props.name]?.clickable) ??
    false;
  const objRef = React.useRef<THREE.Object3D | null>(null);

  // Update global registry of node objects.
  // This is used for updating bone transforms in skinned meshes.
  const viewerMutable = viewer.mutable.current;
  React.useEffect(() => {
    if (objRef.current !== null)
      viewerMutable.nodeRefFromName[props.name] = objRef.current;
  });

  // Get children from scene tree state
  const childrenNames = viewer.useSceneTree(
    (state) => state.nodeFromName[props.name]?.children,
  );

  // Create object + children.
  //
  // For not-fully-understood reasons, wrapping makeObject with useMemo() fixes
  // stability issues (eg breaking runtime errors) associated with
  // PivotControls.
  const objNode = React.useMemo(() => {
    if (makeObject === undefined) return null;

    // Pose will need to be updated.
    const attrs = viewerMutable.nodeAttributesFromName;
    if (!(props.name in attrs)) {
      attrs[props.name] = {};
    }
    attrs[props.name]!.poseUpdateState = "needsUpdate";

    // Create children components
    const children = (
      <>
        {childrenNames &&
          childrenNames.map((child_id) => (
            <SceneNodeThreeObject
              key={child_id}
              name={child_id}
              parent={objRef.current}
            />
          ))}
        <SceneNodeLabel name={props.name} />
      </>
    );

    return makeObject(objRef, children);
  }, [makeObject, childrenNames]);

  // Helper for transient visibility checks. Checks the .visible attribute of
  // both this object and ancestors.
  //
  // This is used for (1) suppressing click events and (2) unmounting when
  // unmountWhenInvisible is true. The latter is used for <Html /> components.
  function isDisplayed(): boolean {
    // We avoid checking obj.visible because obj may be unmounted when
    // unmountWhenInvisible=true.
    const attrs = viewerMutable.nodeAttributesFromName[props.name];
    const visibility =
      (attrs?.overrideVisibility === undefined
        ? attrs?.visibility
        : attrs.overrideVisibility) ?? true;
    if (visibility === false) return false;
    if (props.parent === null) return true;

    // Check visibility of parents + ancestors.
    let visible = props.parent.visible;
    if (visible) {
      props.parent.traverseAncestors((ancestor) => {
        visible = visible && ancestor.visible;
      });
    }
    return visible;
  }

  // Pose needs to be updated whenever component is remounted.
  React.useEffect(() => {
    const attrs = viewerMutable.nodeAttributesFromName[props.name];
    if (attrs !== undefined) attrs.poseUpdateState = "needsUpdate";
  });

  // Update attributes on a per-frame basis. Currently does redundant work,
  // although this shouldn't be a bottleneck.
  useFrame(
    () => {
      const attrs = viewerMutable.nodeAttributesFromName[props.name];

      // Unmount when invisible.
      // Examples: <Html /> components, PivotControls.
      //
      // This is a workaround for situations where just setting `visible` doesn't
      // work (like <Html />), or to prevent invisible elements from being
      // interacted with (<PivotControls />).
      //
      // https://github.com/pmndrs/drei/issues/1323
      if (unmountWhenInvisible) {
        const displayed = isDisplayed();
        if (displayed && unmount) {
          if (objRef.current !== null) objRef.current.visible = false;
          setUnmount(false);
        }
        if (!displayed && !unmount) {
          setUnmount(true);
        }
      }

      if (objRef.current === null) return;
      if (attrs === undefined) return;

      const visibility =
        (attrs?.overrideVisibility === undefined
          ? attrs?.visibility
          : attrs.overrideVisibility) ?? true;
      objRef.current.visible = visibility;

      if (attrs.poseUpdateState == "needsUpdate") {
        attrs.poseUpdateState = "updated";
        const wxyz = attrs.wxyz ?? [1, 0, 0, 0];
        objRef.current.quaternion.set(wxyz[1], wxyz[2], wxyz[3], wxyz[0]);
        const position = attrs.position ?? [0, 0, 0];
        objRef.current.position.set(position[0], position[1], position[2]);

        // Update matrices if necessary. This is necessary for PivotControls.
        if (!objRef.current.matrixAutoUpdate) objRef.current.updateMatrix();
        if (!objRef.current.matrixWorldAutoUpdate)
          objRef.current.updateMatrixWorld();
      }
    },
    // Other useFrame hooks may depend on transforms + visibility. So it's best
    // to call this hook early.
    //
    // However, it's also important that this is *higher* than the priority for
    // the MessageHandler's useFrame. This is to make sure that transforms are
    // updated in the same frame that they are set.
    -1000,
  );

  // Clicking logic.
  const sendClicksThrottled = useThrottledMessageSender(50);

  // Track hover state.
  const hoveredRef = React.useRef<HoverState>({
    isHovered: false,
    instanceId: null,
    clickable: false,
  });
  hoveredRef.current.clickable = clickable;

  // Handle case where clickable is toggled to false while still hovered.
  if (!clickable && hoveredRef.current.isHovered) {
    hoveredRef.current.isHovered = false;
    viewerMutable.hoveredElementsCount--;
    if (viewerMutable.hoveredElementsCount === 0) {
      document.body.style.cursor = "auto";
    }
  }

  const dragInfo = React.useRef({
    dragging: false,
    startClientX: 0,
    startClientY: 0,
  });

  if (objNode === undefined || unmount) {
    return null;
  } else {
    return (
      <>
        <group
          // Instead of using onClick, we use onPointerDown/Move/Up to check mouse drag,
          // and only send a click if the mouse hasn't moved between the down and up events.
          //  - onPointerDown resets the click state (dragged = false)
          //  - onPointerMove, if triggered, sets dragged = true
          //  - onPointerUp, if triggered, sends a click if dragged = false.
          // Note: It would be cool to have dragged actions too...
          onPointerDown={
            !clickable
              ? undefined
              : (e) => {
                  if (!isDisplayed()) return;
                  e.stopPropagation();
                  const state = dragInfo.current;
                  const canvasBbox =
                    viewerMutable.canvas!.getBoundingClientRect();
                  state.startClientX = e.clientX - canvasBbox.left;
                  state.startClientY = e.clientY - canvasBbox.top;
                  state.dragging = false;
                }
          }
          onPointerMove={
            !clickable
              ? undefined
              : (e) => {
                  if (!isDisplayed()) return;
                  e.stopPropagation();
                  const state = dragInfo.current;
                  const canvasBbox =
                    viewerMutable.canvas!.getBoundingClientRect();
                  const deltaX =
                    e.clientX - canvasBbox.left - state.startClientX;
                  const deltaY =
                    e.clientY - canvasBbox.top - state.startClientY;
                  // Minimum motion.
                  if (Math.abs(deltaX) <= 3 && Math.abs(deltaY) <= 3) return;
                  state.dragging = true;
                }
          }
          onPointerUp={
            !clickable
              ? undefined
              : (e) => {
                  if (!isDisplayed()) return;
                  e.stopPropagation();
                  const state = dragInfo.current;
                  if (state.dragging) return;
                  // Convert ray to viser coordinates.
                  const ray = rayToViserCoords(viewer, e.ray);

                  // Send OpenCV image coordinates to the server (normalized).
                  const canvasBbox =
                    viewerMutable.canvas!.getBoundingClientRect();
                  const mouseVectorOpenCV = opencvXyFromPointerXy(viewer, [
                    e.clientX - canvasBbox.left,
                    e.clientY - canvasBbox.top,
                  ]);

                  sendClicksThrottled({
                    type: "SceneNodeClickMessage",
                    name: props.name,
                    instance_index:
                      computeClickInstanceIndexFromInstanceId === undefined
                        ? null
                        : computeClickInstanceIndexFromInstanceId(e.instanceId),
                    // Note that the threejs up is +Y, but we expose a +Z up.
                    ray_origin: [ray.origin.x, ray.origin.y, ray.origin.z],
                    ray_direction: [
                      ray.direction.x,
                      ray.direction.y,
                      ray.direction.z,
                    ],
                    screen_pos: [mouseVectorOpenCV.x, mouseVectorOpenCV.y],
                  });
                }
          }
          onPointerOver={
            !clickable
              ? undefined
              : (e) => {
                  if (!isDisplayed()) return;
                  e.stopPropagation();

                  // Update hover state
                  hoveredRef.current.isHovered = true;
                  // Store the instanceId in the hover ref
                  hoveredRef.current.instanceId = e.instanceId ?? null;

                  // Increment global hover count and update cursor
                  viewerMutable.hoveredElementsCount++;
                  if (viewerMutable.hoveredElementsCount === 1) {
                    document.body.style.cursor = "pointer";
                  }
                }
          }
          onPointerOut={
            !clickable
              ? undefined
              : () => {
                  if (!isDisplayed()) return;

                  // Update hover state
                  hoveredRef.current.isHovered = false;
                  // Clear the instanceId when no longer hovering
                  hoveredRef.current.instanceId = null;

                  // Decrement global hover count and update cursor if needed
                  viewerMutable.hoveredElementsCount--;
                  if (viewerMutable.hoveredElementsCount === 0) {
                    document.body.style.cursor = "auto";
                  }
                }
          }
        >
          <HoverableContext.Provider value={hoveredRef}>
            {objNode}
          </HoverableContext.Provider>
        </group>
      </>
    );
  }
}
