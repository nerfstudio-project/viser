import * as React from 'react';
import * as THREE from 'three';
import { Text, Button, Slider, Table, Flex, ActionIcon, Divider, Switch, Input, NumberInput, Group } from "@mantine/core";
import { MultiSlider } from "./MultiSlider";
import { IconArrowsHorizontal, IconArrowsVertical, IconCamera, IconCameraPlus, IconChevronDown, IconChevronUp, IconClockHour5, IconDownload, IconKeyframes, IconPlayerPauseFilled, IconPlayerPlayFilled, IconPlayerTrackNextFilled, IconPlayerTrackPrevFilled, IconTrash, IconUpload } from '@tabler/icons-react';
import { ViewerContext, ViewerContextContents } from "../App";
import { SceneNode } from '../SceneTree';
import { CameraFrustum, CoordinateFrame } from '../ThreeAssets';
import { getR_threeworld_world } from '../WorldTransformUtils';
import { get_curve_object_from_cameras, get_transform_matrix } from './curve';
import { MeshLineGeometry, MeshLineMaterial } from 'meshline';
import { MantineReactTable, useMantineReactTable } from 'mantine-react-table';


export function pickFile(onFilePicked: (file: File) => void): void {
  const inputElemenet = document.createElement('input');
  inputElemenet.style.display = 'none';
  inputElemenet.type = 'file';

  inputElemenet.addEventListener('change', () => {
    if (inputElemenet.files) {
      onFilePicked(inputElemenet.files[0]);
    }
  });

  const teardown = () => {
    document.body.removeEventListener('focus', teardown, true);
    setTimeout(() => {
        document.body.removeChild(inputElemenet);
    }, 1000);
  }
  document.body.addEventListener('focus', teardown, true);

  document.body.appendChild(inputElemenet);
  inputElemenet.click();
}


export interface CameraTrajectoryPanelProps {
  visible: boolean
}

export interface Camera {
  time: number
  name: string
  fov: number
  position: [number, number, number]
  wxyz: [number, number, number, number]
}

function getCameraHash({ fov, position, wxyz }: { fov: number, position: [number, number, number], wxyz: [number, number, number, number] }) {
  const data = [fov, ...position, ...wxyz];
  const seed = 0;
  let h1 = 0xdeadbeef ^ seed, h2 = 0x41c6ce57 ^ seed;
  for(let i = 0, ch; i < data.length; i++) {
    h1 = Math.imul(h1 ^ data[i], 2654435761);
    h2 = Math.imul(h2 ^ data[i], 1597334677);
  }
  h1  = Math.imul(h1 ^ (h1 >>> 16), 2246822507);
  h1 ^= Math.imul(h2 ^ (h2 >>> 13), 3266489909);
  h2  = Math.imul(h2 ^ (h2 >>> 16), 2246822507);
  h2 ^= Math.imul(h1 ^ (h1 >>> 13), 3266489909);
  return 4294967296 * (2097151 & h2) + (h1 >>> 0);
}

function mapNumberToAlphabet(number: number) : string {
  let out = ""
  const n = 10 + 'z'.charCodeAt(0) - 'a'.charCodeAt(0) + 1;
  while (number > 0) {
    const current = number % n;
    out += (current < 10) ? (current).toString() : String.fromCharCode('a'.charCodeAt(0) + current - 10);
    number = Math.floor(number / n);
  }
  return out;
}

function rescaleCameras(cameras: Camera[]) : Camera[] {
  if (cameras.length === 0) return [];
  if (cameras.length == 1) return [{...cameras[0], time: 0}];
  const min = Math.min(...cameras.map(x => x.time));
  const max = Math.max(...cameras.map(x => x.time));
  return cameras.map(x => ({...x, time: (x.time - min) / (max - min)}));
}


function getPoseFromCamera(viewer: ViewerContextContents) {
  const three_camera = viewer.cameraRef.current!;
  const R_threecam_cam = new THREE.Quaternion().setFromEuler(
    new THREE.Euler(Math.PI, 0.0, 0.0),
  );
  const R_world_threeworld = getR_threeworld_world(viewer).invert();
  const R_world_camera = R_world_threeworld.clone()
    .multiply(three_camera.quaternion);
    //.multiply(R_threecam_cam);

  return {
    wxyz: [
      R_world_camera.w,
      R_world_camera.x,
      R_world_camera.y,
      R_world_camera.z,
    ] as [number, number, number, number],
    position: three_camera.position
      .clone()
      .applyQuaternion(R_world_threeworld)
      .toArray(),
    aspect: three_camera.aspect,
    fov: (three_camera.fov * Math.PI) / 180.0,
  };
}


export default function CameraTrajectoryPanel(props: CameraTrajectoryPanelProps) {
  if (!props.visible) {
    return null;
  }

  const viewer = React.useContext(ViewerContext)!;
  const R_threeworld_world = getR_threeworld_world(viewer);
  const removeSceneNode = viewer.useSceneTree((state) => state.removeSceneNode);
  const addSceneNode = viewer.useSceneTree((state) => state.addSceneNode);
  const nodeFromName = viewer.useSceneTree((state) => state.nodeFromName);
  const isRenderMode = viewer.useGui((state) => state.isRenderMode);
  
  // Get default FOV from camera
  const [isCycle, setIsCycle] = React.useState<boolean>(false);
  const [isPlaying, setIsPlaying] = React.useState<boolean>(false);
  const [fps, setFps] = React.useState<number>(24);
  const [smoothness, setSmoothness] = React.useState<number>(0.5);
  const [cameras, setCameras] = React.useState<Camera[]>([]);
  const [fov, setFov] = React.useState<number>(viewer.cameraRef.current?.fov ?? 75.0);
  const [renderWidth, setRenderWidth] = React.useState<number>(1920);
  const [renderHeight, setRenderHeight] = React.useState<number>(1080);
  const [playerTime, setPlayerTime] = React.useState<number>(0.);
  const aspect = renderWidth / renderHeight;

  const baseTreeName = "CameraTrajectory"
  const ensureThreeRootExists = () => {
    if (!(baseTreeName in nodeFromName)) {
      addSceneNode(
        new SceneNode<THREE.Group>(baseTreeName, (ref) => (
          <CoordinateFrame ref={ref} show_axes={false} />
        )) as SceneNode<any>,
      );
    }
    if (!(`${baseTreeName}/PlayerCamera` in nodeFromName)) {
      addSceneNode(
        new SceneNode<THREE.Group>(
            `${baseTreeName}/PlayerCamera`,
            (ref) => (
              <CameraFrustum
                ref={ref}
                fov={fov}
                aspect={aspect}
                scale={0.4}
                color={0xff00ff00}
              />
            ),
        ) as SceneNode<any>);
    }
    const attr = viewer.nodeAttributesFromName.current;
    if (attr[`${baseTreeName}/PlayerCamera`] === undefined) attr[`${baseTreeName}/PlayerCamera`] = {};
    if (attr[baseTreeName] === undefined) attr[baseTreeName] = {};
    attr[`${baseTreeName}/PlayerCamera`]!.visibility = false;
    attr[`${baseTreeName}/PlayerCamera`]!.renderModeVisibility = false;
    attr[baseTreeName]!.renderModeVisibility = false;
  }
  React.useEffect(() => {
    ensureThreeRootExists();
    return () => {
      const attr = viewer.nodeAttributesFromName.current;
      `${baseTreeName}/PlayerCamera` in nodeFromName && removeSceneNode(`${baseTreeName}/PlayerCamera`);
      baseTreeName in nodeFromName && removeSceneNode(baseTreeName);
      `${baseTreeName}/PlayerCamera` in attr && delete attr[`${baseTreeName}/PlayerCamera`];
      baseTreeName in attr && delete attr[baseTreeName];
    }
  }, []);

  const curveObject = React.useMemo(() => cameras.length > 1 ? get_curve_object_from_cameras(
    cameras.map(({fov, wxyz, position, time}: Camera) => ({
      time,
      fov,
      position: new THREE.Vector3(...position),
      quaternion: new THREE.Quaternion(wxyz[1], wxyz[2], wxyz[3], wxyz[0]),
    })), isCycle, smoothness) : null, [cameras, isCycle, smoothness]);

  // Update cameras and trajectory
  React.useEffect(() => {
    ensureThreeRootExists();
    // Update trajectory
    if (!(baseTreeName in nodeFromName)) return;
    const children = nodeFromName[baseTreeName]!.children;
    const enabledCameras = new Set(cameras.map(x => `${baseTreeName}/Camera.${x.name}`));
    children.filter((c) => !enabledCameras.has(c)).forEach((c) => {
      removeSceneNode(c);
      const attr = viewer.nodeAttributesFromName.current;
      if (attr[c] !== undefined) delete attr[c];
    });
    cameras.forEach((camera, index) => {
      const nodeName = `${baseTreeName}/Camera.${camera.name}`;
      if (!(nodeName in nodeFromName)) {
        const node: SceneNode<any> = new SceneNode<THREE.Group>(
            `${baseTreeName}/Camera.${camera.name}`,
            (ref) => (
              <CameraFrustum
                ref={ref}
                fov={camera.fov}
                aspect={aspect}
                scale={0.2}
                color={0xff888888}
              />
            ),
        );
        addSceneNode(node);
      }
      const attr = viewer.nodeAttributesFromName.current;
      if (attr[nodeName] === undefined) attr[nodeName] = {};
      attr[nodeName]!.wxyz = camera.wxyz;
      attr[nodeName]!.position = camera.position;
      attr[nodeName]!.renderModeVisibility = false;
    });
  }, [cameras, aspect, smoothness]);


  // Render camera path
  React.useEffect(() => {
    ensureThreeRootExists();
    const nodeName = `${baseTreeName}/Trajectory`;
    if (curveObject !== null) {
      const num_points = fps * seconds;
      const points = curveObject.curve_positions.getPoints(num_points);
      const resolution = new THREE.Vector2( window.innerWidth, window.innerHeight );
      const geometry = new MeshLineGeometry();
      geometry.setPoints(points);
      const material = new MeshLineMaterial({ 
        lineWidth: 0.03, 
        color: 0xff5024,
        resolution,
      });
      addSceneNode(new SceneNode<THREE.Mesh>(nodeName, (ref) => {
        return <mesh ref={ref} geometry={geometry} material={material} />
      }, () => {
        geometry.dispose();
        material.dispose();
      }) as SceneNode<any>);

      const attr = viewer.nodeAttributesFromName.current;
      if (attr[nodeName] === undefined) attr[nodeName] = {};
      attr[nodeName]!.renderModeVisibility = false;
    } else if (nodeName in nodeFromName) {
      removeSceneNode(nodeName);
    }
  }, [curveObject, fps, isRenderMode]);

  React.useEffect(() => {
    ensureThreeRootExists();
    // set the camera
    if (curveObject !== null) {
      if (isRenderMode) {
        const point = getKeyframePoint(playerTime);
        const position = curveObject.curve_positions.getPoint(point).applyQuaternion(R_threeworld_world);
        const lookat = curveObject.curve_lookats.getPoint(point).applyQuaternion(R_threeworld_world);
        const up = curveObject.curve_ups.getPoint(point).multiplyScalar(-1).applyQuaternion(R_threeworld_world);
        const fov = curveObject.curve_fovs.getPoint(point).z;

        // const cameraControls = viewer.cameraControlRef.current!;
        const threeCamera = viewer.cameraRef.current!;

        threeCamera.position.set(...position.toArray());
        threeCamera.up.set(...up.toArray());
        threeCamera.lookAt(...lookat.toArray());
        // cameraControls.updateCameraUp();
        // cameraControls.setLookAt(...position.toArray(), ...lookat.toArray(), false);
        // const target = position.clone().add(lookat);
        // NOTE: lookat is being ignored when calling setLookAt
        // cameraControls.setTarget(...target.toArray(), false);
        threeCamera.setFocalLength(
          (0.5 * threeCamera.getFilmHeight()) / Math.tan(fov / 2.0),
        );
        // cameraControls.update(1.);
      } else {
        const point = getKeyframePoint(playerTime);
        const position = curveObject.curve_positions.getPoint(point);
        const lookat = curveObject.curve_lookats.getPoint(point);
        const up = curveObject.curve_ups.getPoint(point);

        const mat = get_transform_matrix(position, lookat, up);
        const quaternion = new THREE.Quaternion().setFromRotationMatrix(mat);

        addSceneNode(
          new SceneNode<THREE.Group>(
              `${baseTreeName}/PlayerCamera`,
              (ref) => (
                <CameraFrustum
                  ref={ref}
                  fov={fov}
                  aspect={aspect}
                  scale={0.4}
                  color={0xff00ff00}
                />
              ),
          ) as SceneNode<any>);
        const attr = viewer.nodeAttributesFromName.current;
        if (attr[`${baseTreeName}/PlayerCamera`] === undefined) attr[`${baseTreeName}/PlayerCamera`] = {};
        attr[`${baseTreeName}/PlayerCamera`]!.visibility = true;
        attr[`${baseTreeName}/PlayerCamera`]!.wxyz = [quaternion.w, quaternion.x, quaternion.y, quaternion.z];
        attr[`${baseTreeName}/PlayerCamera`]!.position = position.toArray();
        attr[`${baseTreeName}/PlayerCamera`]!.renderModeVisibility = false;
      }
    } else {
      const attr = viewer.nodeAttributesFromName.current;
      if (attr[`${baseTreeName}/PlayerCamera`] !== undefined)
        attr[`${baseTreeName}/PlayerCamera`]!.visibility = false;
    }
  }, [curveObject, fps, isRenderMode, playerTime]);

  const [seconds, setSeconds] = React.useState(4);

  React.useEffect(() => {
    if (isPlaying && cameras.length > 1) {
      const interval = setInterval(() => {
        setPlayerTime((prev) => {
          let out = Math.min(1., prev + 1 / (fps * seconds))
          if (out >= 1) {
            setIsPlaying(false);
            out = 0;
          }
          return out;
        });
      }, 1000 / fps);
      return () => clearInterval(interval);
    }
  }, [isPlaying, seconds, fps]);


  const addCamera = () => {
    setCameras((cameras) => {
      const { position, wxyz } = getPoseFromCamera(viewer);
      const hash = getCameraHash({ fov, position, wxyz });
      let name = `${mapNumberToAlphabet(hash).slice(0, 6)}`;
      const nameNumber = cameras.filter(x => x.name.startsWith(name)).length;
      if (nameNumber > 0) {
        name += `-${nameNumber+1}`;
      }
      if (cameras.length >= 2) {
        const mult = 1 - 1/cameras.length;
        return [...cameras.map(x => ({...x, time: x.time * mult})), { 
          time: 1,
          name,
          position,
          wxyz,
          fov,
        }];
      } else {
        return [...cameras, { 
          time: cameras.length === 0 ? 0 : 1,
          name,
          position,
          wxyz,
          fov,
        }];
      }
    });
  }


  const displayRenderTime = false;


  const getKeyframePoint = (progress: number) => {
    const times = [];
    const ratio = (cameras.length - 1) / cameras.length;
    cameras.forEach((camera) => {
      const time = camera.time;
      times.push(isCycle ? time * ratio : time);
    });

    if (isCycle) {
      times.push(1.0);
    }

    let new_point = 0.0;
    if (progress <= times[0]) {
      new_point = 0.0;
    } else if (progress >= times[times.length - 1]) {
      new_point = 1.0;
    } else {
      let i = 0;
      while (
        i < times.length - 1 &&
        !(progress >= times[i] && progress < times[i + 1])
      ) {
        i += 1;
      }
      const percentage = (progress - times[i]) / (times[i + 1] - times[i]);
      new_point = (i + percentage) / (times.length - 1);
    }
    return new_point;
  };


  const loadCameraPath = (camera_path_object: any) => {
    const new_camera_list = [];

    setRenderHeight(camera_path_object.render_height);
    setRenderWidth(camera_path_object.render_width);
    if (camera_path_object.camera_type !== "perspective") {
      // TODO: handle this better!
      throw new Error("Unsupported camera type: " + camera_path_object.camera_type);
    }

    setFps(camera_path_object.fps);
    setSeconds(camera_path_object.seconds);

    setSmoothness(camera_path_object.smoothness_value);
    setIsCycle(camera_path_object.is_cycle);

    for (let i = 0; i < camera_path_object.keyframes.length; i += 1) {
      const keyframe = camera_path_object.keyframes[i];

      // properties
      const properties = new Map(JSON.parse(keyframe.properties));
      const mat = new THREE.Matrix4();
      mat.fromArray(JSON.parse(keyframe.matrix));
      const position = new THREE.Vector3();
      const quaternion = new THREE.Quaternion();
      const scale = new THREE.Vector3();
      mat.decompose(position, quaternion, scale);

      // aspect = keyframe.aspect;
      const camera: Camera = {
        position: position.toArray(),
        wxyz: [quaternion.w, quaternion.x, quaternion.y, quaternion.z],
        name: properties.get("NAME") as string,
        time: properties.get("TIME") as number, 
        fov: keyframe.fov,
      };
      new_camera_list.push(camera);
    }
    setCameras(new_camera_list);
  }


  const getCameraPath = () => {
    const num_points = Math.round(fps * seconds);
    const camera_path = [];

    for (let i = 0; i < num_points; i += 1) {
      const pt = getKeyframePoint(i / num_points);

      const position = curveObject!.curve_positions.getPoint(pt);
      const lookat = curveObject!.curve_lookats.getPoint(pt);
      const up = curveObject!.curve_ups.getPoint(pt);
      const fov = curveObject!.curve_fovs.getPoint(pt).z;

      const mat = get_transform_matrix(position, lookat, up);

      if (displayRenderTime) {
        const renderTime = curveObject!.curve_render_times.getPoint(pt).z;
        camera_path.push({
          camera_to_world: mat.transpose().elements, // convert from col-major to row-major matrix
          fov,
          aspect: aspect,
          render_time: Math.max(Math.min(renderTime, 1.0), 0.0), // clamp time values to [0, 1]
        });
      } else {
        camera_path.push({
          camera_to_world: mat.transpose().elements, // convert from col-major to row-major matrix
          fov,
          aspect: aspect,
        });
      }
    }

    const keyframes = [];
    for (let i = 0; i < cameras.length; i += 1) {
      const camera = cameras[i];

      const up = new THREE.Vector3(0, 1, 0); // y is up in local space
      const lookat = new THREE.Vector3(0, 0, 1); // z is forward in local space
      const wxyz = camera.wxyz
      const quaternion = new THREE.Quaternion(wxyz[1], wxyz[2], wxyz[3], wxyz[0]);

      up.applyQuaternion(quaternion);
      lookat.applyQuaternion(quaternion);

      const matrix = get_transform_matrix(new THREE.Vector3(...camera.position), lookat, up);
      keyframes.push({
        matrix: JSON.stringify(matrix.toArray()),
        fov: camera.fov,
        aspect: aspect,
        properties: JSON.stringify([
          ["FOV", camera.fov],
          ["NAME", camera.name],
          ["TIME", camera.time],
        ]),
      });
    }

    // const myData
    const camera_path_object = {
      format: "nerfstudio-viewer",
      keyframes,
      camera_type: "perspective",
      render_height: renderHeight,
      render_width: renderWidth,
      camera_path,
      fps,
      seconds,
      smoothness_value: smoothness,
      is_cycle: isCycle,
      crop: null,
    };
    return camera_path_object;
  }


  const exportCameraPath = () => {
    const cameraPath = getCameraPath();

    // create file in browser
    const json = JSON.stringify(cameraPath, null, 2);
    const blob = new Blob([json], { type: 'application/json' });
    const href = URL.createObjectURL(blob);

    // create "a" HTLM element with href to file
    const link = document.createElement('a');
    link.href = href;

    const filename = 'camera_path.json';
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    // clean up "a" element & remove ObjectURL
    document.body.removeChild(link);
    URL.revokeObjectURL(href);
  }

  const uploadCameraPath = () => {
    pickFile((file) => {
      const fr = new FileReader();
      fr.onload = (res) => {
        const camera_path_object = JSON.parse(res.target!.result! as string);
        loadCameraPath(camera_path_object);
      };
      fr.readAsText(file);
    });
  }

  const marks = [];
  for (let i = 0; i <= 1; i += 0.25) {
    marks.push({ value: i, label: `${(seconds * i).toFixed(1).toString()}s` });
  }

  // const table = useMantineReactTable({
  //   data: cameras,
  //   enableBottomToolbar: false,
  //   enableTopToolbar: false,
  //   enableSorting: false,
  //   enableColumnActions: false,
  //   enableDensityToggle: false,
  //   enableRowSelection: false,
  //   enableRowOrdering: true,
  //   enableHiding: false,
  //   enableColumnFilters: false,
  //   enablePagination: false,
  //   mantineTableProps: {
  //     verticalSpacing: 2,
  //   },
  //   enableRowActions: true,
  //   mantinePaperProps: { shadow: undefined },
  //   mantineTableContainerProps: { sx: { maxHeight: "30em" } },
  //   mantinePaginationProps: {
  //     showRowsPerPage: false,
  //   },
  //   displayColumnDefOptions: {
  //     'mrt-row-drag': {
  //       header: "",
  //       minSize: 0,
  //       size: 10,
  //       maxSize: 10,
  //     },
  //     'mrt-row-actions': {
  //       header: "",
  //       minSize: 0,
  //       size: 10,
  //       maxSize: 10,
  //     },
  //   },
  //   renderRowActions: (row) => {
  //     return <ActionIcon onClick={() => {
  //       const cameraId = row.row.original.name;
  //       setCameras(rescaleCameras(cameras.filter(x => x.name !== cameraId)));
  //     }}>
  //       <IconTrash />
  //     </ActionIcon>
  //   },
  //   // mantineTableBodyRowProps={({ row }) => ({
  //   //   onPointerOver: () => {
  //   //     setLabelVisibility(row.getValue("name"), true);
  //   //   },
  //   //   onPointerOut: () => {
  //   //     setLabelVisibility(row.getValue("name"), false);/
  //   //   },
  //   //   ...(row.subRows === undefined || row.subRows.length === 0
  //   //     ? {}
  //   //     : {
  //   //         onClick: () => {
  //   //           row.toggleExpanded();
  //   //         },
  //   //         sx: {
  //   //           cursor: "pointer",
  //   //         },
  //   //       }),
  //   // })}
  //   mantineRowDragHandleProps: ({table}) => ({
  //     onDragEnd: () => {
  //       const { draggingRow, hoveredRow } = table.getState();
  //       if (hoveredRow && draggingRow) {
  //         const clone = [...cameras];
  //         clone.splice(hoveredRow.index, 0, clone.splice(draggingRow.index, 1)[0]);
  //       }
  //     },
  //   }),
  //   columns: [{
  //     header: "Time",
  //     minSize: 10,
  //     size: 30,
  //     accessorFn(originalRow) {
  //       return (originalRow.time * seconds).toFixed(2).toString() + "s";
  //     },
  //   }, {
  //     size: 30,
  //     minSize: 10,
  //     header: "Camera",
  //     accessorKey: "name",
  //   }],
  //   initialState: {
  //     density: "xs",
  //   }
  // });

  return (<>
    <Flex gap="xs" mb="md">
      <Button
        mt="xs"
        fullWidth
        leftIcon={<IconDownload size="1rem" />}
        style={{ height: "1.875rem" }}
        onClick={exportCameraPath}>Export</Button>
      <Button
        mt="xs"
        fullWidth
        leftIcon={<IconUpload size="1rem" />}
        style={{ height: "1.875rem" }}
        onClick={uploadCameraPath}>Import</Button>
    </Flex>
    <Group grow>
      <NumberInput
        id="fps"
        label="FPS"
        value={fps}
        precision={0}
        min={1}
        max={undefined}
        icon={<IconKeyframes size="1rem" />}
        step={1}
        size="xs"
        onChange={(newValue) => newValue !== "" && setFps(newValue)}
        styles={{
          input: {
            minHeight: "1.625rem",
            height: "1.625rem",
          },
        }}
        stepHoldDelay={500}
        stepHoldInterval={(t) => Math.max(1000 / t ** 2, 25)}
      />
      <NumberInput
        id="fov"
        label="FOV"
        value={fov}
        precision={2}
        min={0.01}
        max={undefined}
        icon={<IconCamera size="1rem" />}
        step={1.0}
        size="xs"
        onChange={(newValue) => newValue !== "" && setFov(newValue)}
        styles={{
          input: {
            minHeight: "1.625rem",
            height: "1.625rem",
          },
        }}
        stepHoldDelay={500}
        stepHoldInterval={(t) => Math.max(1000 / t ** 2, 25)}
      />
    </Group>
    <Group grow>
      <NumberInput
        id="width"
        label="Width"
        description="Width in px."
        value={renderWidth}
        precision={0}
        min={1}
        max={undefined}
        icon={<IconArrowsHorizontal size="1rem" />}
        step={100}
        size="xs"
        onChange={(newValue) => newValue !== "" && setRenderWidth(newValue)}
        styles={{
          input: {
            minHeight: "1.625rem",
            height: "1.625rem",
          },
        }}
        stepHoldDelay={500}
        stepHoldInterval={(t) => Math.max(1000 / t ** 2, 25)}
      />
      <NumberInput
        id="height"
        label="Height"
        description="Height in px."
        value={renderHeight}
        precision={0}
        min={1}
        max={undefined}
        icon={<IconArrowsVertical size="1rem" />}
        step={100}
        size="xs"
        onChange={(newValue) => newValue !== "" && setRenderHeight(newValue)}
        styles={{
          input: {
            minHeight: "1.625rem",
            height: "1.625rem",
          },
        }}
        stepHoldDelay={500}
        stepHoldInterval={(t) => Math.max(1000 / t ** 2, 25)}
      />
    </Group>
    <NumberInput
      id="duration"
      label="Duration"
      description="Duration of the camera trajectory in seconds."
      value={seconds}
      precision={2}
      min={0.3}
      max={undefined}
      icon={<IconClockHour5 size="1rem" />}
      step={1.00}
      size="xs"
      onChange={(newValue) => newValue !== "" && setSeconds(newValue)}
      styles={{
        input: {
          minHeight: "1.625rem",
          height: "1.625rem",
        },
      }}
      stepHoldDelay={500}
      stepHoldInterval={(t) => Math.max(1000 / t ** 2, 25)}
    />
    <Text size="xs" mt="sm">Smoothness</Text>
    <Slider
      mb="xl"
      pb="md"
      value={smoothness}
      onChange={setSmoothness}
      min={0.0}
      max={1.0}
      step={0.01}
      marks={[
        { value: 0.0, label: "0.0" },
        { value: 0.5, label: "0.5" },
        { value: 1.0, label: "1.0" },
      ]}
    />
    <Divider mt="xs" />
    <Text size="sm" mt="sm">Timeline</Text>
    <MultiSlider 
      mb="xl" 
      pb="xl"
      fixedEndpoints={true} 
      max={1.0} 
      step={0.01} 
      minRange={0.02} 
      label={(x) => `${(seconds*x).toFixed(2)}s`}
      value={cameras.map(x=>x.time)} 
      onChange={(value) => {
       setCameras(cameras.map((camera, index) => ({
         ...camera,
         time: value[index],
       })));
      }}
      onChangeEnd={(value) => {
       setCameras(cameras.map((camera, index) => ({
         ...camera,
         time: value[index],
       })));
      }}
      marks={marks} />
    <Button  
        fullWidth
        mt="xl"
        mb="md"
        leftIcon={<IconCameraPlus size="1rem" />}
        style={{ height: "1.875rem" }}
        onClick={addCamera}>Add camera</Button>
    <Table mb="lg">
      <thead>
        <tr>
          <th></th>
          <th>Time</th>
          <th>Camera</th>
        </tr>
      </thead>
      <tbody>
        {cameras.map((camera, index) => {
          return (
            <tr key={index}>
              <td>
                <Flex>
                  <ActionIcon onClick={() => {
                      setCameras(rescaleCameras([...cameras.slice(0, index), ...cameras.slice(index + 1)]));
                    }}>
                    <IconTrash />
                  </ActionIcon>
                  <ActionIcon disabled={index == 0} onClick={() => {
                      const clone = [...cameras];
                      const tmp = cameras[index];
                      clone[index] = {
                        ...cameras[index - 1],
                        time: cameras[index].time
                      };
                      clone[index - 1] = {
                        ...tmp,
                        time: cameras[index - 1].time
                      }
                      setCameras(clone);
                    }}>
                    <IconChevronUp />
                  </ActionIcon>
                  <ActionIcon disabled={index == cameras.length - 1} onClick={() => {
                      const clone = [...cameras];
                      const tmp = cameras[index];
                      clone[index] = {
                        ...cameras[index + 1],
                        time: cameras[index].time
                      };
                      clone[index + 1] = {
                        ...tmp,
                        time: cameras[index + 1].time
                      }
                      setCameras(clone);
                    }}>
                    <IconChevronDown />
                  </ActionIcon>
                </Flex>
              </td>
              <td>{(seconds * camera.time).toFixed(2).toString()}s</td>
              <td>{camera.name}</td>
            </tr>
          )
        })}
      </tbody>
    </Table>

    <Divider mt="xs" />
    <Text size="sm" mt="sm">Player</Text>
    <Slider mt="xs" label="Time" value={playerTime} onChange={setPlayerTime} min={0} max={1} step={0.001} />
    <Flex style={{ alignContent: "center", justifyContent: "center" }}>
      <ActionIcon aria-label="prev" size="lg" onClick={() => setPlayerTime(Math.max(0, ...cameras.map(x => x.time).filter(x => x < playerTime)))}>
        <IconPlayerTrackPrevFilled />
      </ActionIcon>
      {isPlaying ? <ActionIcon aria-label="pause" size="lg" onClick={() => setIsPlaying(false)}>
        <IconPlayerPauseFilled />
      </ActionIcon> : <ActionIcon aria-label="play" size="lg" onClick={() => setIsPlaying(true)}>
        <IconPlayerPlayFilled />
      </ActionIcon>}
      <ActionIcon aria-label="next" size="lg" onClick={() => setPlayerTime(Math.min(1, ...cameras.map(x => x.time).filter(x => x > playerTime)))}>
        <IconPlayerTrackNextFilled />
      </ActionIcon>
    </Flex>
    <Switch
      mt="xs"
      radius="sm"
      label="Render mode"
      checked={isRenderMode}
      onChange={(event) => {
        viewer.setIsRenderMode(event.currentTarget.checked);
      }}
      size="sm"
    />
  </>)
}

CameraTrajectoryPanel.defaultProps = {
  visible: true,
};