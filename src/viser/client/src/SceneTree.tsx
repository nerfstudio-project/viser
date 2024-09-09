import {
  CatmullRomLine,
  CubicBezierLine,
  Grid,
  PivotControls,
  useCursor,
} from "@react-three/drei";
import { useContextBridge } from "its-fine";
import { createPortal, useFrame } from "@react-three/fiber";
import React from "react";
import * as THREE from "three";

import { ViewerContext } from "./App";
import {
  makeThrottledMessageSender,
  useThrottledMessageSender,
} from "./WebsocketFunctions";
import { Html } from "@react-three/drei";
import { useSceneTreeState } from "./SceneTreeState";
import { ErrorBoundary } from "react-error-boundary";
import { rayToViserCoords } from "./WorldTransformUtils";
import {
  CameraFrustum,
  CoordinateFrame,
  GlbAsset,
  HoverableContext,
  InstancedAxes,
  PointCloud,
  ViserImage,
  ViserMesh,
} from "./ThreeAssets";
import { opencvXyFromPointerXy } from "./ClickUtils";
import { SceneNodeMessage } from "./WebsocketMessages";
import { SplatObject } from "./Splatting/GaussianSplats";
import { Paper } from "@mantine/core";
import GeneratedGuiContainer from "./ControlPanel/Generated";

/** Type corresponding to a zustand-style useSceneTree hook. */
export type UseSceneTree = ReturnType<typeof useSceneTreeState>;

function SceneNodeThreeChildren(props: {
  name: string;
  parent: THREE.Object3D;
}) {
  const viewer = React.useContext(ViewerContext)!;

  const [children, setChildren] = React.useState<string[]>(
    viewer.useSceneTree.getState().nodeFromName[props.name]?.children ?? [],
  );

  React.useEffect(() => {
    let updateQueued = false;
    return viewer.useSceneTree.subscribe((state) => {
      // Do nothing if an update is already queued.
      if (updateQueued) return;

      // Do nothing if children haven't changed.
      const newChildren = state.nodeFromName[props.name]?.children;
      if (
        newChildren === undefined ||
        newChildren === children || // Note that this won't check for elementwise equality!
        (newChildren.length === 0 && children.length == 0)
      )
        return;

      // Queue a (throttled) children update.
      updateQueued = true;
      setTimeout(
        () => {
          updateQueued = false;
          const newChildren =
            viewer.useSceneTree.getState().nodeFromName[props.name]!.children!;
          setChildren(newChildren);
        },
        // Throttle more when we have a lot of children...
        newChildren.length <= 16 ? 10 : newChildren.length <= 128 ? 50 : 200,
      );
    });
  }, []);

  // Create a group of children inside of the parent object.
  return createPortal(
    <group>
      {children &&
        children.map((child_id) => (
          <SceneNodeThreeObject
            key={child_id}
            name={child_id}
            parent={props.parent}
          />
        ))}
      <SceneNodeLabel name={props.name} />
    </group>,
    props.parent,
  );
}

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

export type MakeObject = (ref: React.Ref<any>) => React.ReactNode;

function useObjectFactory(message: SceneNodeMessage | undefined): {
  makeObject: MakeObject;
  unmountWhenInvisible?: boolean;
  computeClickInstanceIndexFromInstanceId?: (
    instanceId: number | undefined,
  ) => number | null;
} {
  const viewer = React.useContext(ViewerContext)!;
  const ContextBridge = useContextBridge();

  if (message === undefined) return { makeObject: () => null };

  switch (message.type) {
    // Add a coordinate frame.
    case "FrameMessage": {
      return {
        makeObject: (ref) => (
          <CoordinateFrame
            ref={ref}
            showAxes={message.show_axes}
            axesLength={message.axes_length}
            axesRadius={message.axes_radius}
            originRadius={message.origin_radius}
          />
        ),
      };
    }

    // Add axes to visualize.
    case "BatchedAxesMessage": {
      return {
        makeObject: (ref) => (
          // Minor naming discrepancy: I think "batched" will be clearer to
          // folks on the Python side, but instanced is somewhat more
          // precise.
          <InstancedAxes
            ref={ref}
            wxyzsBatched={
              new Float32Array(
                message.wxyzs_batched.buffer.slice(
                  message.wxyzs_batched.byteOffset,
                  message.wxyzs_batched.byteOffset +
                    message.wxyzs_batched.byteLength,
                ),
              )
            }
            positionsBatched={
              new Float32Array(
                message.positions_batched.buffer.slice(
                  message.positions_batched.byteOffset,
                  message.positions_batched.byteOffset +
                    message.positions_batched.byteLength,
                ),
              )
            }
            axes_length={message.axes_length}
            axes_radius={message.axes_radius}
          />
        ),
        // Compute click instance index from instance ID. Each visualized
        // frame has 1 instance for each of 3 line segments.
        computeClickInstanceIndexFromInstanceId: (instanceId) =>
          Math.floor(instanceId! / 3),
      };
    }

    case "GridMessage": {
      return {
        makeObject: (ref) => (
          <group ref={ref}>
            <Grid
              args={[
                message.width,
                message.height,
                message.width_segments,
                message.height_segments,
              ]}
              side={THREE.DoubleSide}
              cellColor={message.cell_color}
              cellThickness={message.cell_thickness}
              cellSize={message.cell_size}
              sectionColor={message.section_color}
              sectionThickness={message.section_thickness}
              sectionSize={message.section_size}
              rotation={
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
                message.plane == "xz"
                  ? new THREE.Euler(0.0, 0.0, 0.0)
                  : message.plane == "xy"
                    ? new THREE.Euler(Math.PI / 2.0, 0.0, 0.0)
                    : message.plane == "yx"
                      ? new THREE.Euler(0.0, Math.PI / 2.0, Math.PI / 2.0)
                      : message.plane == "yz"
                        ? new THREE.Euler(0.0, 0.0, Math.PI / 2.0)
                        : message.plane == "zx"
                          ? new THREE.Euler(0.0, Math.PI / 2.0, 0.0)
                          : message.plane == "zy"
                            ? new THREE.Euler(
                                -Math.PI / 2.0,
                                0.0,
                                -Math.PI / 2.0,
                              )
                            : undefined
              }
            />
          </group>
        ),
      };
    }

    // Add a point cloud.
    case "PointCloudMessage": {
      return {
        makeObject: (ref) => (
          <PointCloud
            ref={ref}
            pointSize={message.point_size}
            pointBallNorm={message.point_ball_norm}
            points={
              new Float32Array(
                message.points.buffer.slice(
                  message.points.byteOffset,
                  message.points.byteOffset + message.points.byteLength,
                ),
              )
            }
            colors={new Float32Array(message.colors).map((val) => val / 255.0)}
          />
        ),
      };
    }

    // Add mesh
    case "SkinnedMeshMessage":
    case "MeshMessage": {
      return { makeObject: (ref) => <ViserMesh ref={ref} {...message} /> };
    }
    // Add a camera frustum.
    case "CameraFrustumMessage": {
      return {
        makeObject: (ref) => (
          <CameraFrustum
            ref={ref}
            fov={message.fov}
            aspect={message.aspect}
            scale={message.scale}
            color={message.color}
            imageBinary={message.image_binary}
            imageMediaType={message.image_media_type}
          />
        ),
      };
    }
    case "TransformControlsMessage": {
      const name = message.name;
      const sendDragMessage = makeThrottledMessageSender(viewer, 50);
      return {
        makeObject: (ref) => (
          <group onClick={(e) => e.stopPropagation()}>
            <PivotControls
              ref={ref}
              scale={message.scale}
              lineWidth={message.line_width}
              fixed={message.fixed}
              autoTransform={message.auto_transform}
              activeAxes={message.active_axes}
              disableAxes={message.disable_axes}
              disableSliders={message.disable_sliders}
              disableRotations={message.disable_rotations}
              disableScaling={true}
              translationLimits={message.translation_limits}
              rotationLimits={message.rotation_limits}
              depthTest={message.depth_test}
              opacity={message.opacity}
              onDrag={(l) => {
                const attrs = viewer.nodeAttributesFromName.current;
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
            />
          </group>
        ),
        unmountWhenInvisible: true,
      };
    }
    // Add a 2D label.
    case "LabelMessage": {
      return {
        makeObject: (ref) => (
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
                  {message.text}
                </span>
              </div>
            </Html>
          </group>
        ),
        unmountWhenInvisible: true,
      };
    }
    case "Gui3DMessage": {
      return {
        makeObject: (ref) => {
          // We wrap with <group /> because Html doesn't implement
          // THREE.Object3D. The initial position is intended to be
          // off-screen; it will be overwritten with the actual position
          // after the component is mounted.
          return (
            <group ref={ref} position={new THREE.Vector3(1e8, 1e8, 1e8)}>
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
                    <GeneratedGuiContainer containerId={message.container_id} />
                  </Paper>
                </ContextBridge>
              </Html>
            </group>
          );
        },
        unmountWhenInvisible: true,
      };
    }
    // Add an image.
    case "ImageMessage": {
      return {
        makeObject: (ref) => <ViserImage ref={ref} {...message} />,
      };
    }
    // Add a glTF/GLB asset.
    case "GlbMessage": {
      return {
        makeObject: (ref) => (
          <GlbAsset
            ref={ref}
            glb_data={new Uint8Array(message.glb_data)}
            scale={message.scale}
          />
        ),
      };
    }
    case "CatmullRomSplineMessage": {
      return {
        makeObject: (ref) => {
          return (
            <group ref={ref}>
              <CatmullRomLine
                points={message.positions}
                closed={message.closed}
                curveType={message.curve_type}
                tension={message.tension}
                lineWidth={message.line_width}
                color={message.color}
                // Sketchy cast needed due to https://github.com/pmndrs/drei/issues/1476.
                segments={(message.segments ?? undefined) as undefined}
              />
            </group>
          );
        },
      };
    }
    case "CubicBezierSplineMessage": {
      return {
        makeObject: (ref) => (
          <group ref={ref}>
            {[...Array(message.positions.length - 1).keys()].map((i) => (
              <CubicBezierLine
                key={i}
                start={message.positions[i]}
                end={message.positions[i + 1]}
                midA={message.control_points[2 * i]}
                midB={message.control_points[2 * i + 1]}
                lineWidth={message.line_width}
                color={message.color}
                // Sketchy cast needed due to https://github.com/pmndrs/drei/issues/1476.
                segments={(message.segments ?? undefined) as undefined}
              ></CubicBezierLine>
            ))}
          </group>
        ),
      };
    }
    case "GaussianSplatsMessage": {
      return {
        makeObject: (ref) => (
          <SplatObject
            ref={ref}
            buffer={
              new Uint32Array(
                message.buffer.buffer.slice(
                  message.buffer.byteOffset,
                  message.buffer.byteOffset + message.buffer.byteLength,
                ),
              )
            }
          />
        ),
      };
    }
  }
}

/** Component containing the three.js object and children for a particular scene node. */
// export function SceneNodeThreeObject(props: {
//   name: string;
//   parent: THREE.Object3D | null;
// }) {
//   const viewer = React.useContext(ViewerContext)!;
//   const message = viewer.useSceneTree(
//     (state) => state.nodeFromName[props.name].message,
//   );
//   // const clickable = viewer.useSceneTree(
//   //   (state) => state.nodeFromName[props.name].clickable,
//   // );
//   //
//
//   const unmountTypes: (typeof message.type)[] = [
//     "TransformControlsMessage",
//     "LabelMessage",
//     "Gui3DMessage",
//   ];
//   const unmountWhenInvisible = message.type in unmountTypes;
//   return null;
// }

export function SceneNodeThreeObject(props: {
  name: string;
  parent: THREE.Object3D | null;
}) {
  const viewer = React.useContext(ViewerContext)!;
  const message = viewer.useSceneTree(
    (state) => state.nodeFromName[props.name]?.message,
  );
  const {
    makeObject,
    unmountWhenInvisible,
    computeClickInstanceIndexFromInstanceId,
  } = useObjectFactory(message);

  const [unmount, setUnmount] = React.useState(false);
  const clickable =
    viewer.useSceneTree((state) => state.nodeFromName[props.name]?.clickable) ??
    false;
  const [obj, setRef] = React.useState<THREE.Object3D | null>(null);

  // Update global registry of node objects.
  // This is used for updating bone transforms in skinned meshes.
  React.useEffect(() => {
    if (obj !== null) viewer.nodeRefFromName.current[props.name] = obj;
  }, [obj]);

  // Create object + children.
  //
  // For not-fully-understood reasons, wrapping makeObject with useMemo() fixes
  // stability issues (eg breaking runtime errors) associated with
  // PivotControls.
  const objNode = React.useMemo(() => {
    if (makeObject === undefined) return null;

    // Pose will need to be updated.
    const attrs = viewer.nodeAttributesFromName.current;
    if (!(props.name in attrs)) {
      attrs[props.name] = {};
    }
    attrs[props.name]!.poseUpdateState = "needsUpdate";

    return makeObject(setRef);
  }, [makeObject]);
  const children =
    obj === null ? null : (
      <SceneNodeThreeChildren name={props.name} parent={obj} />
    );

  // Helper for transient visibility checks. Checks the .visible attribute of
  // both this object and ancestors.
  //
  // This is used for (1) suppressing click events and (2) unmounting when
  // unmountWhenInvisible is true. The latter is used for <Html /> components.
  function isDisplayed() {
    // We avoid checking obj.visible because obj may be unmounted when
    // unmountWhenInvisible=true.
    const attrs = viewer.nodeAttributesFromName.current[props.name];
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
    const attrs = viewer.nodeAttributesFromName.current[props.name];
    if (attrs !== undefined) attrs.poseUpdateState = "needsUpdate";
  });

  // Update attributes on a per-frame basis. Currently does redundant work,
  // although this shouldn't be a bottleneck.
  useFrame(
    () => {
      const attrs = viewer.nodeAttributesFromName.current[props.name];

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
          if (obj !== null) obj.visible = false;
          setUnmount(false);
        }
        if (!displayed && !unmount) {
          setUnmount(true);
        }
      }

      if (obj === null) return;
      if (attrs === undefined) return;

      const visibility =
        (attrs?.overrideVisibility === undefined
          ? attrs?.visibility
          : attrs.overrideVisibility) ?? true;
      obj.visible = visibility;

      if (attrs.poseUpdateState == "needsUpdate") {
        attrs.poseUpdateState = "updated";
        const wxyz = attrs.wxyz;
        if (wxyz !== undefined) {
          obj.quaternion.set(wxyz[1], wxyz[2], wxyz[3], wxyz[0]);
        }
        const position = attrs.position;
        if (position !== undefined) {
          obj.position.set(position[0], position[1], position[2]);
        }

        // Update matrices if necessary. This is necessary for PivotControls.
        if (!obj.matrixAutoUpdate) obj.updateMatrix();
        if (!obj.matrixWorldAutoUpdate) obj.updateMatrixWorld();
      }
    },
    // Other useFrame hooks may depend on transforms + visibility. So it's best
    // to call this hook early.
    -10000,
  );

  // Clicking logic.
  const sendClicksThrottled = useThrottledMessageSender(50);
  const [hovered, setHovered] = React.useState(false);
  useCursor(hovered);
  const hoveredRef = React.useRef(false);
  if (!clickable && hovered) setHovered(false);

  const dragInfo = React.useRef({
    dragging: false,
    startClientX: 0,
    startClientY: 0,
  });

  if (objNode === undefined || unmount) {
    return <>{children}</>;
  } else if (clickable) {
    return (
      <>
        <ErrorBoundary
          fallbackRender={() => {
            // This sometimes (but very rarely) catches a race condition when
            // we remove scene nodes. I would guess it's related to portaling,
            // but the issue is unnoticeable with ErrorBoundary in-place so not
            // debugging further for now...
            console.error(
              "There was an error rendering a scene node object:",
              objNode,
            );
            return null;
          }}
        >
          <group
            // Instead of using onClick, we use onPointerDown/Move/Up to check mouse drag,
            // and only send a click if the mouse hasn't moved between the down and up events.
            //  - onPointerDown resets the click state (dragged = false)
            //  - onPointerMove, if triggered, sets dragged = true
            //  - onPointerUp, if triggered, sends a click if dragged = false.
            // Note: It would be cool to have dragged actions too...
            onPointerDown={(e) => {
              if (!isDisplayed()) return;
              e.stopPropagation();
              const state = dragInfo.current;
              const canvasBbox =
                viewer.canvasRef.current!.getBoundingClientRect();
              state.startClientX = e.clientX - canvasBbox.left;
              state.startClientY = e.clientY - canvasBbox.top;
              state.dragging = false;
            }}
            onPointerMove={(e) => {
              if (!isDisplayed()) return;
              e.stopPropagation();
              const state = dragInfo.current;
              const canvasBbox =
                viewer.canvasRef.current!.getBoundingClientRect();
              const deltaX = e.clientX - canvasBbox.left - state.startClientX;
              const deltaY = e.clientY - canvasBbox.top - state.startClientY;
              // Minimum motion.
              if (Math.abs(deltaX) <= 3 && Math.abs(deltaY) <= 3) return;
              state.dragging = true;
            }}
            onPointerUp={(e) => {
              if (!isDisplayed()) return;
              e.stopPropagation();
              const state = dragInfo.current;
              if (state.dragging) return;
              // Convert ray to viser coordinates.
              const ray = rayToViserCoords(viewer, e.ray);

              // Send OpenCV image coordinates to the server (normalized).
              const canvasBbox =
                viewer.canvasRef.current!.getBoundingClientRect();
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
            }}
            onPointerOver={(e) => {
              if (!isDisplayed()) return;
              e.stopPropagation();
              setHovered(true);
              hoveredRef.current = true;
            }}
            onPointerOut={() => {
              if (!isDisplayed()) return;
              setHovered(false);
              hoveredRef.current = false;
            }}
          >
            <HoverableContext.Provider value={hoveredRef}>
              {objNode}
            </HoverableContext.Provider>
          </group>
          {children}
        </ErrorBoundary>
      </>
    );
  } else {
    return (
      <>
        {/* This <group /> does nothing, but switching between clickable vs not
        causes strange transform behavior without it. */}
        <group>{objNode}</group>
        {children}
      </>
    );
  }
}
