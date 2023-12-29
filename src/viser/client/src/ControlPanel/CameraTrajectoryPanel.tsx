import * as React from 'react';
import * as THREE from 'three';
//import {
//  IconEdit,
//  IconEye,
//  IconTrash,
//  IconLayoutNavbarExpand,
//} from '@tabler/icons-react';

// import { MeshLine, MeshLineMaterial } from 'meshline';
// import { useContext, useEffect } from 'react';
// import { useDispatch, useSelector } from 'react-redux';

// import { CSS2DObject } from 'three/examples/jsm/renderers/CSS2DRenderer';
import { Text, Button, Slider, Table, Flex, ActionIcon, Divider, Switch, Input, NumberInput, Group } from "@mantine/core";
import { MultiSlider } from "./MultiSlider";
import { IconArrowsHorizontal, IconArrowsVertical, IconCamera, IconCameraPlus, IconChevronDown, IconChevronUp, IconClockHour5, IconDownload, IconKeyframes, IconPlayerPauseFilled, IconPlayerPlay, IconPlayerPlayFilled, IconPlayerTrackNextFilled, IconPlayerTrackPrevFilled, IconTrash, IconUpload } from '@tabler/icons-react';
import { ViewerContext, ViewerContextContents } from "../App";
import { SceneNode } from '../SceneTree';
import { CameraFrustum, CameraFrustumProps, CoordinateFrame } from '../ThreeAssets';
import { getR_threeworld_world } from '../WorldTransformUtils';
import { get_curve_object_from_cameras, get_transform_matrix } from './curve';
import { MeshLineGeometry, MeshLineMaterial, raycast } from 'meshline';
import { MantineReactTable, useMantineReactTable } from 'mantine-react-table';
import { render } from '@react-three/fiber';


// const FOV_LABELS = {
//   FOV: '°',
//   MM: 'mm',
// };

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
    .multiply(three_camera.quaternion)
    .multiply(R_threecam_cam);

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

  const baseTreeName = "CameraTrajectory"
  React.useEffect(() => {
    if (!(baseTreeName in nodeFromName)) {
      addSceneNode(
        new SceneNode<THREE.Group>(baseTreeName, (ref) => (
          <CoordinateFrame ref={ref} show_axes={false} />
        )) as SceneNode<any>,
      );
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
      attr[`${baseTreeName}/PlayerCamera`]!.visibility = false;
      return () => {
        removeSceneNode(`${baseTreeName}/PlayerCamera`);
        removeSceneNode(baseTreeName);
      }
    }
  }, []);

  const [isCycle, setIsCycle] = React.useState<boolean>(false);
  const [isPlaying, setIsPlaying] = React.useState<boolean>(false);
  const [fps, setFps] = React.useState<number>(30);
  const [smoothness, setSmoothness] = React.useState<number>(0.5);
  const [cameras, setCameras] = React.useState<Camera[]>([]);
  const [fov, setFov] = React.useState<number>(1.);
  const [renderWidth, setRenderWidth] = React.useState<number>(1920);
  const [renderHeight, setRenderHeight] = React.useState<number>(1080);
  const [playerTime, setPlayerTime] = React.useState<number>(0.);
  const aspect = renderWidth / renderHeight;

  const curveObject = React.useMemo(() => cameras.length > 1 ? get_curve_object_from_cameras(
    cameras.map(({fov, wxyz, position, time}: Camera) => ({
      time,
      fov,
      position: new THREE.Vector3(...position),
      quaternion: new THREE.Quaternion(wxyz[1], wxyz[2], wxyz[3], wxyz[0]),
    })), isCycle, smoothness) : null, [cameras, isCycle, smoothness]);

  // Update cameras and trajectory
  React.useEffect(() => {
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
    });
  }, [cameras, aspect, smoothness]);


  // Render camera path
  React.useEffect(() => {
    if (!(baseTreeName in nodeFromName)) return;
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

    } else if (nodeName in nodeFromName) {
      removeSceneNode(nodeName);
    }
  }, [curveObject, fps, isRenderMode]);

  React.useEffect(() => {
    // set the camera
    if (curveObject !== null) {
      if (isRenderMode) {
        const point = getKeyframePoint(playerTime);
        const position = curveObject.curve_positions.getPoint(point).applyQuaternion(R_threeworld_world);
        const lookat = curveObject.curve_lookats.getPoint(point).applyQuaternion(R_threeworld_world);
        const up = curveObject.curve_ups.getPoint(point).multiplyScalar(-1).applyQuaternion(R_threeworld_world);
        const fov = curveObject.curve_fovs.getPoint(point).z;

        const cameraControls = viewer.cameraControlRef.current!;
        const threeCamera = viewer.cameraRef.current!;

        threeCamera.up.set(...up.toArray());
        cameraControls.updateCameraUp();
        cameraControls.setLookAt(...position.toArray(), ...lookat.toArray(), false);
        const target = position.clone().add(lookat);
        // NOTE: lookat is being ignore when calling setLookAt
        cameraControls.setTarget(...target.toArray(), false);
        threeCamera.setFocalLength(
          (0.5 * threeCamera.getFilmHeight()) / Math.tan(fov / 2.0),
        );
        cameraControls.update(1.);
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
    const { position, wxyz } = getPoseFromCamera(viewer);
    const hash = getCameraHash({ fov, position, wxyz });
    const name = `${mapNumberToAlphabet(hash).slice(0, 6)}`;

    if (cameras.length >= 2) {
      const mult = 1 - 1/cameras.length;
      setCameras([...cameras.map(x => ({...x, time: x.time * mult})), { 
        time: 1,
        name,
        position,
        wxyz,
        fov,
      }]);
    } else {
      setCameras([...cameras, { 
        time: cameras.length === 0 ? 0 : 1,
        name,
        position,
        wxyz,
        fov,
      }]);
    }
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
        viewer.useGui.setState({ isRenderMode: event.currentTarget.checked });
      }}
      size="sm"
    />
  </>)
}

CameraTrajectoryPanel.defaultProps = {
  visible: true,
};



//function set_camera_position(camera, matrix) {
//  const mat = new THREE.Matrix4();
//  mat.fromArray(matrix.elements);
//  mat.decompose(camera.position, camera.quaternion, camera.scale);
//}
//
//function CameraList(props) {
//  const throttled_time_message_sender = props.throttled_time_message_sender;
//  const sceneTree = props.sceneTree;
//  const cameras = props.cameras;
//  const camera_main = props.camera_main;
//  const transform_controls = props.transform_controls;
//  const setCameras = props.setCameras;
//  const swapCameras = props.swapCameras;
//  const fovLabel = props.fovLabel;
//  const setFovLabel = props.setFovLabel;
//  const cameraProperties = props.cameraProperties;
//  const setCameraProperties = props.setCameraProperties;
//  const isAnimated = props.isAnimated;
//  // eslint-disable-next-line no-unused-vars
//  const slider_value = props.slider_value;
//  const set_slider_value = props.set_slider_value;
//
//  const [expanded, setExpanded] = React.useState(null);
//
//  const camera_type = useSelector((state) => state.renderingState.camera_type);
//
//  const handleChange =
//    (cameraUUID: string) =>
//    (event: React.SyntheticEvent, isExpanded: boolean) => {
//      setExpanded(isExpanded ? cameraUUID : false);
//    };
//
//  const set_transform_controls = (index) => {
//    // camera helper object so grab the camera inside
//    const camera = sceneTree.find_object_no_create([
//      'Camera Path',
//      'Cameras',
//      index.toString(),
//      'Camera',
//    ]);
//    if (camera !== null) {
//      const viewer_buttons = document.getElementsByClassName(
//        'ViewerWindow-buttons',
//      )[0];
//      if (camera === transform_controls.object) {
//        // double click to remove controls from object
//        transform_controls.detach();
//        viewer_buttons.style.display = 'none';
//      } else {
//        transform_controls.detach();
//        transform_controls.attach(camera);
//        viewer_buttons.style.display = 'block';
//      }
//    }
//  };
//
//  const reset_slider_render_on_change = () => {
//    // set slider and render camera back to 0
//    const slider_min = 0;
//    const camera_render = sceneTree.find_object_no_create([
//      'Cameras',
//      'Render Camera',
//    ]);
//    const camera_render_helper = sceneTree.find_object_no_create([
//      'Cameras',
//      'Render Camera',
//      'Helper',
//    ]);
//    if (cameras.length >= 1) {
//      let first_camera = sceneTree.find_object_no_create([
//        'Camera Path',
//        'Cameras',
//        0,
//        'Camera',
//      ]);
//      if (first_camera.type !== 'PerspectiveCamera' && cameras.length > 1) {
//        first_camera = sceneTree.find_object_no_create([
//          'Camera Path',
//          'Cameras',
//          1,
//          'Camera',
//        ]);
//      }
//      set_camera_position(camera_render, first_camera.matrix);
//      camera_render_helper.set_visibility(true);
//      camera_render.fov = first_camera.fov;
//      camera_render.renderTime = first_camera.renderTime;
//    }
//    set_slider_value(slider_min);
//  };
//
//  const delete_camera = (index: number) => {
//    const camera_render_helper = sceneTree.find_object_no_create([
//      'Cameras',
//      'Render Camera',
//      'Helper',
//    ]);
//    console.log('TODO: deleting camera: ', index);
//    sceneTree.delete(['Camera Path', 'Cameras', index.toString(), 'Camera']);
//    sceneTree.delete([
//      'Camera Path',
//      'Cameras',
//      index.toString(),
//      'Camera Helper',
//    ]);
//
//    setCameras([...cameras.slice(0, index), ...cameras.slice(index + 1)]);
//    // detach and hide transform controls
//    transform_controls.detach();
//    const viewer_buttons = document.getElementsByClassName(
//      'ViewerWindow-buttons',
//    )[0];
//    viewer_buttons.style.display = 'none';
//    if (cameras.length < 1) {
//      camera_render_helper.set_visibility(false);
//    }
//    reset_slider_render_on_change();
//  };
//
//  const cameraList = cameras.map((camera, index) => {
//    return (
//      <Accordion
//        className="CameraList-row"
//        key={camera.uuid}
//        expanded={expanded === camera.uuid}
//      >
//        <Accordion.Control
//          chevron={<ExpandMore sx={{ color: '#eeeeee' }} />}
//          aria-controls="panel1bh-content"
//          id="panel1bh-header"
//        >
//          <Stack spacing={0}>
//            <Button
//              size="small"
//              onClick={(e) => {
//                swapCameras(index, index - 1);
//                e.stopPropagation();
//              }}
//              style={{
//                maxWidth: '20px',
//                maxHeight: '20px',
//                minWidth: '20px',
//                minHeight: '20px',
//              }}
//              disabled={index === 0}
//            >
//              <KeyboardArrowUp />
//            </Button>
//            <Button
//              size="small"
//              onClick={(e) => {
//                swapCameras(index, index + 1);
//                e.stopPropagation();
//              }}
//              style={{
//                maxWidth: '20px',
//                maxHeight: '20px',
//                minWidth: '20px',
//                minHeight: '20px',
//              }}
//              disabled={index === cameras.length - 1}
//            >
//              <KeyboardArrowDown />
//            </Button>
//          </Stack>
//          <Button size="small" sx={{ ml: '3px' }}>
//            <TextField
//              id="standard-basic"
//              value={camera.properties.get('NAME')}
//              variant="standard"
//              onClick={(e) => e.stopPropagation()}
//              onChange={(e) => {
//                const cameraProps = new Map(cameraProperties);
//                cameraProps.get(camera.uuid).set('NAME', e.target.value);
//                setCameraProperties(cameraProps);
//              }}
//              sx={{
//                alignItems: 'center',
//                alignContent: 'center',
//              }}
//            />
//          </Button>
//          <Button
//            size="small"
//            onClick={(e) => {
//              e.stopPropagation();
//              set_transform_controls(index);
//            }}
//          >
//            <IconEdit />
//          </Button>
//          <Stack spacing={0} direction="row" justifyContent="end">
//            <Button
//              size="small"
//              onClick={(e) => {
//                e.stopPropagation();
//                set_camera_position(camera_main, camera.matrix);
//                camera_main.fov = camera.fov;
//                camera_main.renderTime = camera.renderTime;
//                set_slider_value(camera.properties.get('TIME'));
//              }}
//            >
//              <IconEye />
//            </Button>
//            <Button size="small" onClick={() => delete_camera(index)}>
//              <IconTrash />
//            </Button>
//          </Stack>
//        </Accordion.Control>
//        <Accordion.Panel>
//          {isAnimated('FOV') && camera_type !== 'equirectangular' && (
//            <span>HI</span>
//          )}
//          {!isAnimated('FOV') && !isAnimated('RenderTime') && (
//            <p style={{ fontSize: 'smaller', color: '#999999' }}>
//              Animated camera properties will show up here!
//            </p>
//          )}
//        </Accordion.Panel>
//      </Accordion>
//    );
//  });
//  return <div>{cameraList}</div>;
//}
//
//export default function CameraPanel(props) {
//  // unpack relevant information
//  const sceneTree = props.sceneTree;
//  const camera_main = sceneTree.find_object_no_create([
//    'Cameras',
//    'Main Camera',
//  ]);
//  const camera_render = sceneTree.find_object_no_create([
//    'Cameras',
//    'Render Camera',
//  ]);
//  const camera_render_helper = sceneTree.find_object_no_create([
//    'Cameras',
//    'Render Camera',
//    'Helper',
//  ]);
//  const transform_controls = sceneTree.find_object_no_create([
//    'Transform Controls',
//  ]);
//
//  // redux store state
//  const DEFAULT_FOV = 50;
//  const DEFAULT_RENDER_TIME = 0.0;
//
//  interface Camera {
//    time: number
//    name: string
//  };
//
//  // react state
//  const [cameras, setCameras] = React.useState<Camera[]>([]);
//  // Mapping of camera id to each camera's properties
//  const [cameraProperties, setCameraProperties] = React.useState(new Map());
//  const [slider_value, set_slider_value] = React.useState(0);
//  const [smoothness_value, set_smoothness_value] = React.useState(0.5);
//  const [is_playing, setIsPlaying] = React.useState(false);
//  const [is_cycle, setIsCycle] = React.useState(false);
//  const [seconds, setSeconds] = React.useState(4);
//  const [fps, setFps] = React.useState(24);
//  const [render_modal_open, setRenderModalOpen] = React.useState(false);
//  const [load_path_modal_open, setLoadPathModalOpen] = React.useState(false);
//  const [animate, setAnimate] = React.useState(new Set());
//  const [globalFov, setGlobalFov] = React.useState(DEFAULT_FOV);
//  const [globalRenderTime, setGlobalRenderTime] =
//    React.useState(DEFAULT_RENDER_TIME);
//
//  const scene_state = sceneTree.get_scene_state();
//
//  // Template for sharing state between Vanilla JS Three.js and React components
//  // eslint-disable-next-line no-unused-vars
//  const [mouseInScene, setMouseInScene] = React.useState<boolean>(false);
//  React.useEffect(() => {
//    scene_state.addCallback(
//      (value: boolean) => setMouseInScene(value),
//      'mouse_in_scene',
//    );
//    // eslint-disable-next-line react-hooks/exhaustive-deps
//  }, []);
//
//  // ui state
//  const [fovLabel, setFovLabel] = React.useState(FOV_LABELS.FOV);
//
//  // nonlinear render option
//  const slider_min = 0;
//  const slider_max = 1;
//
//  // animation constants
//  const total_num_steps = seconds * fps;
//  const step_size = slider_max / total_num_steps;
//
//  const reset_slider_render_on_add = (new_camera_list) => {
//    // set slider and render camera back to 0
//    if (new_camera_list.length >= 1) {
//      set_camera_position(camera_render, new_camera_list[0].matrix);
//      setFieldOfView(new_camera_list[0].fov);
//      set_slider_value(slider_min);
//    }
//  };
//
//  const add_camera = () => {
//    const camera_main_copy = camera_main.clone();
//    camera_main_copy.aspect = 1.0;
//    camera_main_copy.fov = globalFov;
//    camera_main_copy.renderTime = globalRenderTime;
//    const new_camera_properties = new Map();
//    camera_main_copy.properties = new_camera_properties;
//    new_camera_properties.set('FOV', globalFov);
//    new_camera_properties.set('NAME', `Camera ${cameras.length}`);
//    // TIME VALUES ARE 0-1
//    if (cameras.length === 0) {
//      new_camera_properties.set('TIME', 0.0);
//    } else {
//      new_camera_properties.set('TIME', 1.0);
//    }
//
//    const ratio = (cameras.length - 1) / cameras.length;
//
//    const new_properties = new Map(cameraProperties);
//    new_properties.forEach((properties) => {
//      properties.set('TIME', properties.get('TIME') * ratio);
//    });
//    new_properties.set(camera_main_copy.uuid, new_camera_properties);
//    setCameraProperties(new_properties);
//
//    const new_camera_list = cameras.concat(camera_main_copy);
//    setCameras(new_camera_list);
//    reset_slider_render_on_add(new_camera_list);
//  };
//
//  const setCameraProperty = (property, value, index) => {
//    const activeCamera = cameras[index];
//    const activeProperties = new Map(activeCamera.properties);
//    activeProperties.set(property, value);
//    const newProperties = new Map(cameraProperties);
//    newProperties.set(activeCamera.uuid, activeProperties);
//    activeCamera.properties = activeProperties;
//    setCameraProperties(newProperties);
//  };
//
//  const swapCameras = (index: number, new_index: number) => {
//    if (
//      Math.min(index, new_index) < 0 ||
//      Math.max(index, new_index) >= cameras.length
//    )
//      return;
//
//    const swapCameraTime = cameras[index].time;
//    cameras[index].time = cameras[new_index].time;
//    cameras[new_index].time = swapCameraTime;
//
//    const new_cameras = [
//      ...cameras.slice(0, index),
//      ...cameras.slice(index + 1),
//    ];
//    setCameras([
//      ...new_cameras.slice(0, new_index),
//      cameras[index],
//      ...new_cameras.slice(new_index),
//    ]);
//
//    // reset_slider_render_on_change();
//  };
//
//  // force a rerender if the cameras are dragged around
//  let update_cameras_interval = null;
//  // eslint-disable-next-line no-unused-vars
//  transform_controls.addEventListener('mouseDown', (event) => {
//    // prevent multiple loops
//    if (update_cameras_interval === null) {
//      // hardcoded for 100 ms per update
//      update_cameras_interval = setInterval(() => {}, 100);
//    }
//  });
//  // eslint-disable-next-line no-unused-vars
//  transform_controls.addEventListener('mouseUp', (event) => {
//    if (update_cameras_interval !== null) {
//      clearInterval(update_cameras_interval);
//      update_cameras_interval = null;
//      setCameras(cameras);
//    }
//  });
//
//  // draw cameras and curve to the scene
//  useEffect(() => {
//    // draw the cameras
//
//    const labels = Array.from(document.getElementsByClassName('label'));
//    labels.forEach((label) => {
//      label.remove();
//    });
//
//    sceneTree.delete(['Camera Path', 'Cameras']); // delete old cameras, which is important
//    if (cameras.length < 1) {
//      dispatch({
//        type: 'write',
//        path: 'renderingState/camera_choice',
//        data: 'Main Camera',
//      });
//      camera_render_helper.set_visibility(false);
//    } else {
//      camera_render_helper.set_visibility(true);
//    }
//    for (let i = 0; i < cameras.length; i += 1) {
//      const camera = cameras[i];
//      // camera.aspect = render_width / render_height;
//      const camera_helper = new CameraHelper(camera, 0x393e46);
//
//      const labelDiv = document.createElement('div');
//      labelDiv.className = 'label';
//      labelDiv.textContent = camera.name;
//      labelDiv.style.color = 'black';
//      labelDiv.style.backgroundColor = 'rgba(255, 255, 255, 0.61)';
//      labelDiv.style.backdropFilter = 'blur(5px)';
//      labelDiv.style.padding = '6px';
//      labelDiv.style.borderRadius = '6px';
//      labelDiv.style.visibility = 'visible';
//      const camera_label = new CSS2DObject(labelDiv);
//      camera_label.name = 'CAMERA_LABEL';
//      camera_label.position.set(0, -0.1, -0.1);
//      camera_helper.add(camera_label);
//      camera_label.layers.set(0);
//
//      // camera
//      sceneTree.set_object_from_path(
//        ['Camera Path', 'Cameras', i.toString(), 'Camera'],
//        camera,
//      );
//      // camera helper
//      sceneTree.set_object_from_path(
//        ['Camera Path', 'Cameras', i.toString(), 'Camera Helper'],
//        camera_helper,
//      );
//    }
//    // eslint-disable-next-line react-hooks/exhaustive-deps
//  }, [cameras, cameraProperties, render_width, render_height]);
//
//  // update the camera curve
//  const curve_object = get_curve_object_from_cameras(
//    cameras,
//    is_cycle,
//    smoothness_value,
//  );
//
//  const getKeyframePoint = (progress: Number) => {
//    const times = [];
//    const ratio = (cameras.length - 1) / cameras.length;
//    cameras.forEach((camera) => {
//      const time = camera.properties.get('TIME');
//      times.push(is_cycle ? time * ratio : time);
//    });
//
//    if (is_cycle) {
//      times.push(1.0);
//    }
//
//    let new_point = 0.0;
//    if (progress <= times[0]) {
//      new_point = 0.0;
//    } else if (progress >= times[times.length - 1]) {
//      new_point = 1.0;
//    } else {
//      let i = 0;
//      while (
//        i < times.length - 1 &&
//        !(progress >= times[i] && progress < times[i + 1])
//      ) {
//        i += 1;
//      }
//      const percentage = (progress - times[i]) / (times[i + 1] - times[i]);
//      new_point = (i + percentage) / (times.length - 1);
//    }
//    return new_point;
//  };
//
//  if (cameras.length > 1) {
//    const num_points = fps * seconds;
//    const points = curve_object.curve_positions.getPoints(num_points);
//    const geometry = new THREE.BufferGeometry().setFromPoints(points);
//    const spline = new MeshLine();
//    spline.setGeometry(geometry);
//    const material = new MeshLineMaterial({ lineWidth: 0.01, color: 0xff5024 });
//    const spline_mesh = new THREE.Mesh(spline.geometry, material);
//    sceneTree.set_object_from_path(['Camera Path', 'Curve'], spline_mesh);
//
//    // set the camera
//
//    const point = getKeyframePoint(slider_value);
//    let position = null;
//    let lookat = null;
//    let up = null;
//    let fov = null;
//    position = curve_object.curve_positions.getPoint(point);
//    lookat = curve_object.curve_lookats.getPoint(point);
//    up = curve_object.curve_ups.getPoint(point);
//    fov = curve_object.curve_fovs.getPoint(point).z;
//
//    const mat = get_transform_matrix(position, lookat, up);
//    set_camera_position(camera_render, mat);
//    setFieldOfView(fov);
//  } else {
//    sceneTree.delete(['Camera Path', 'Curve']);
//  }
//
//  const values = [];
//  cameras.forEach((camera) => {
//    const time = camera.properties.get('TIME');
//    const ratio = (cameras.length - 1) / cameras.length;
//    values.push(is_cycle ? time * ratio : time);
//  });
//
//  if (is_cycle && cameras.length !== 0) {
//    values.push(1.0);
//  }
//
//  const handleKeyframeSlider = (
//    newValue: number | number[],
//    activeThumb: number,
//  ) => {
//    if (activeThumb === cameras.length) return;
//    const ratio = (cameras.length - 1) / cameras.length;
//    const val = newValue[activeThumb];
//    setCameraProperty(
//      'TIME',
//      is_cycle ? Math.min(val / ratio, 1.0) : val,
//      activeThumb,
//    );
//  };
//
//  // when the slider changes, update the main camera position
//  useEffect(() => {
//    if (cameras.length > 1) {
//      const point = getKeyframePoint(slider_value);
//      let position = null;
//      let lookat = null;
//      let up = null;
//      let fov = null;
//      let render_time = null;
//      position = curve_object.curve_positions.getPoint(point);
//      lookat = curve_object.curve_lookats.getPoint(point);
//      up = curve_object.curve_ups.getPoint(point);
//      fov = curve_object.curve_fovs.getPoint(point).z;
//      render_time = curve_object.curve_render_times.getPoint(point).z;
//      render_time = Math.max(Math.min(render_time, 1.0), 0.0); // clamp time values to [0, 1]
//      const mat = get_transform_matrix(position, lookat, up);
//      set_camera_position(camera_render, mat);
//      setFieldOfView(fov);
//      setGlobalFov(fov);
//      setGlobalRenderTime(render_time);
//    }
//    // eslint-disable-next-line react-hooks/exhaustive-deps
//  }, [slider_value, render_height, render_width]);
//
//  // call this function whenever slider state changes
//  useEffect(() => {
//    if (is_playing && cameras.length > 1) {
//      const interval = setInterval(() => {
//        set_slider_value((prev) => prev + step_size);
//      }, 1000 / fps);
//      return () => clearInterval(interval);
//    }
//    return () => {};
//    // eslint-disable-next-line react-hooks/exhaustive-deps
//  }, [is_playing]);
//
//  // make sure to pause if the slider reaches the end
//  useEffect(() => {
//    if (slider_value >= slider_max) {
//      set_slider_value(slider_max);
//      setIsPlaying(false);
//    }
//  }, [slider_value]);
//
//  const get_camera_path = () => {
//    // NOTE: currently assuming these are ints
//    const num_points = fps * seconds;
//    const camera_path = [];
//
//    for (let i = 0; i < num_points; i += 1) {
//      const pt = getKeyframePoint(i / num_points);
//
//      const position = curve_object.curve_positions.getPoint(pt);
//      const lookat = curve_object.curve_lookats.getPoint(pt);
//      const up = curve_object.curve_ups.getPoint(pt);
//      const fov = curve_object.curve_fovs.getPoint(pt).z;
//
//      const mat = get_transform_matrix(position, lookat, up);
//
//      if (display_render_time) {
//        const renderTime = curve_object.curve_render_times.getPoint(pt).z;
//        camera_path.push({
//          camera_to_world: mat.transpose().elements, // convert from col-major to row-major matrix
//          fov,
//          aspect: camera_render.aspect,
//          render_time: Math.max(Math.min(renderTime, 1.0), 0.0), // clamp time values to [0, 1]
//        });
//      } else {
//        camera_path.push({
//          camera_to_world: mat.transpose().elements, // convert from col-major to row-major matrix
//          fov,
//          aspect: camera_render.aspect,
//        });
//      }
//    }
//
//    const keyframes = [];
//    for (let i = 0; i < cameras.length; i += 1) {
//      const camera = cameras[i];
//      keyframes.push({
//        matrix: JSON.stringify(camera.matrix.toArray()),
//        fov: camera.fov,
//        aspect: camera_render.aspect,
//        properties: JSON.stringify(Array.from(camera.properties.entries())),
//      });
//    }
//
//    // const myData
//    const camera_path_object = {
//      keyframes,
//      camera_type,
//      render_height,
//      render_width,
//      camera_path,
//      fps,
//      seconds,
//      smoothness_value,
//      is_cycle,
//      crop: null,
//    };
//    return camera_path_object;
//  };
//
//  const export_camera_path = () => {
//    // export the camera path
//    // inspired by:
//    // https://stackoverflow.com/questions/55613438/reactwrite-to-json-file-or-export-download-no-server
//
//    sendWebsocketMessage(viser_websocket, { type: 'SaveCheckpointMessage' });
//
//    const camera_path_object = get_camera_path();
//    console.log(camera_render.toJSON());
//
//    // create file in browser
//    const json = JSON.stringify(camera_path_object, null, 2);
//    const blob = new Blob([json], { type: 'application/json' });
//    const href = URL.createObjectURL(blob);
//
//    // create "a" HTLM element with href to file
//    const link = document.createElement('a');
//    link.href = href;
//
//    const filename = 'camera_path.json';
//    link.download = filename;
//    document.body.appendChild(link);
//    link.click();
//    // clean up "a" element & remove ObjectURL
//    document.body.removeChild(link);
//    URL.revokeObjectURL(href);
//  };
//
//  const load_camera_path = (camera_path_object) => {
//    const new_camera_list = [];
//    const new_properties = new Map(cameraProperties);
//
//    setRenderHeight(camera_path_object.render_height);
//    setRenderWidth(camera_path_object.render_width);
//    setCameraType(camera_path_object.camera_type);
//
//    setFps(camera_path_object.fps);
//    setSeconds(camera_path_object.seconds);
//
//    set_smoothness_value(camera_path_object.smoothness_value);
//    setIsCycle(camera_path_object.is_cycle);
//
//    for (let i = 0; i < camera_path_object.keyframes.length; i += 1) {
//      const keyframe = camera_path_object.keyframes[i];
//      const camera = new THREE.PerspectiveCamera(
//        keyframe.fov,
//        keyframe.aspect,
//        0.1,
//        1000,
//      );
//
//      // properties
//      camera.properties = new Map(JSON.parse(keyframe.properties));
//      new_properties.set(camera.uuid, camera.properties);
//
//      const mat = new THREE.Matrix4();
//      mat.fromArray(JSON.parse(keyframe.matrix));
//      set_camera_position(camera, mat);
//      new_camera_list.push(camera);
//    }
//
//    setCameraProperties(new_properties);
//    setCameras(new_camera_list);
//    reset_slider_render_on_add(new_camera_list);
//
//    if ('crop' in camera_path_object && camera_path_object.crop !== null) {
//      const bg_color = camera_path_object.crop.crop_bg_color;
//      sendWebsocketMessage(viser_websocket, {
//        type: 'CropParamsMessage',
//        crop_enabled: true,
//        crop_bg_color: [bg_color.r, bg_color.g, bg_color.b],
//        crop_center: camera_path_object.crop.crop_center,
//        crop_scale: camera_path_object.crop.crop_scale,
//      });
//    }
//  };
//
//  const uploadCameraPath = (e) => {
//    const fileUpload = e.target.files[0];
//
//    const fr = new FileReader();
//    fr.onload = (res) => {
//      const camera_path_object = JSON.parse(res.target.result);
//      load_camera_path(camera_path_object);
//    };
//
//    fr.readAsText(fileUpload);
//  };
//
//  const open_render_modal = () => {
//    setRenderModalOpen(true);
//
//    const camera_path_object = get_camera_path();
//
//    sendWebsocketMessage(viser_websocket, {
//      type: 'CameraPathPayloadMessage',
//      camera_path_filename: export_path,
//      camera_path: camera_path_object,
//    });
//    sendWebsocketMessage(viser_websocket, { type: 'SaveCheckpointMessage' });
//  };
//
//  const open_load_path_modal = () => {
//    sendWebsocketMessage(viser_websocket, { type: 'CameraPathOptionsRequest' });
//    setLoadPathModalOpen(true);
//  };
//
//  const isAnimated = (property) => animate.has(property);
//
//  const toggleAnimate = (property) => {
//    const new_animate = new Set(animate);
//    if (animate.has(property)) {
//      new_animate.delete(property);
//      setAnimate(new_animate);
//    } else {
//      new_animate.add(property);
//      setAnimate(new_animate);
//    }
//  };
//
//  const setAllCameraFOV = (val) => {
//    if (fovLabel === FOV_LABELS.FOV) {
//      for (let i = 0; i < cameras.length; i += 1) {
//        cameras[i].fov = val;
//      }
//    } else {
//      for (let i = 0; i < cameras.length; i += 1) {
//        cameras[i].setFocalLength(val / cameras[i].aspect);
//      }
//    }
//  };
//
//  const setAllCameraRenderTime = (val) => {
//    for (let i = 0; i < cameras.length; i += 1) {
//      cameras[i].renderTime = val;
//    }
//  };
//
//  return (
//    <div className="CameraPanel">
//      <div>
//        <div className="CameraPanel-path-row">
//          <LoadPathModal
//            open={load_path_modal_open}
//            setOpen={setLoadPathModalOpen}
//            pathUploadFunction={uploadCameraPath}
//            loadCameraPathFunction={load_camera_path}
//          />
//          <Button
//            size="small"
//            className="CameraPanel-top-button"
//            component="label"
//            variant="outlined"
//            startIcon={<FileUploadOutlinedIcon />}
//            onClick={open_load_path_modal}
//          >
//            Load Path
//          </Button>
//        </div>
//        <div className="CameraPanel-path-row">
//          <Button
//            size="small"
//            className="CameraPanel-top-button"
//            variant="outlined"
//            startIcon={<FileDownloadOutlinedIcon />}
//            onClick={export_camera_path}
//            disabled={cameras.length === 0}
//          >
//            Export Path
//          </Button>
//        </div>
//        <br />
//        <RenderModal open={render_modal_open} setOpen={setRenderModalOpen} />
//        <Button
//          className="CameraPanel-render-button"
//          variant="outlined"
//          size="small"
//          startIcon={<VideoCameraBackIcon />}
//          onClick={open_render_modal}
//          disabled={cameras.length === 0}
//        >
//          Render
//        </Button>
//      </div>
//      <div className="CameraPanel-props">
//        <LevaPanel
//          store={cameraPropsStore}
//          className="Leva-panel"
//          theme={LevaTheme}
//          titleBar={false}
//          fill
//          flat
//        />
//        <LevaStoreProvider store={cameraPropsStore}>
//          <CameraPropPanel
//            seconds={seconds}
//            set_seconds={setSeconds}
//            fps={fps}
//            set_fps={setFps}
//          />
//        </LevaStoreProvider>
//      </div>
//      {display_render_time && (
//        <div className="CameraList-row-animation-properties">
//          <Tooltip title="Animate Render Time for Each Camera">
//            <Button
//              value="animateRenderTime"
//              selected={isAnimated('RenderTime')}
//              onClick={() => {
//                toggleAnimate('RenderTime');
//              }}
//              style={{
//                maxWidth: '20px',
//                maxHeight: '20px',
//                minWidth: '20px',
//                minHeight: '20px',
//                position: 'relative',
//                top: '22px',
//              }}
//              sx={{
//                mt: 1,
//              }}
//            >
//              <Animation
//                style={{
//                  color: isAnimated('RenderTime') ? '#24B6FF' : '#EBEBEB',
//                  maxWidth: '20px',
//                  maxHeight: '20px',
//                  minWidth: '20px',
//                  minHeight: '20px',
//                }}
//              />
//            </Button>
//          </Tooltip>
//          <RenderTimeSelector
//            throttled_time_message_sender={throttled_time_message_sender}
//            disabled={false}
//            isGlobal
//            camera={camera_main}
//            dispatch={dispatch}
//            globalRenderTime={globalRenderTime}
//            setGlobalRenderTime={setGlobalRenderTime}
//            applyAll={!isAnimated('RenderTime')}
//            setAllCameraRenderTime={setAllCameraRenderTime}
//            changeMain
//          />
//        </div>
//      )}
//      {camera_type !== 'equirectangular' && (
//        <div className="CameraList-row-animation-properties">
//          <Tooltip title="Animate FOV for Each Camera">
//            <Button
//              value="animatefov"
//              selected={isAnimated('FOV')}
//              onClick={() => {
//                toggleAnimate('FOV');
//              }}
//              style={{
//                maxWidth: '20px',
//                maxHeight: '20px',
//                minWidth: '20px',
//                minHeight: '20px',
//                position: 'relative',
//                top: '22px',
//              }}
//              sx={{
//                mt: 1,
//              }}
//            >
//              <Animation
//                style={{
//                  color: isAnimated('FOV') ? '#24B6FF' : '#EBEBEB',
//                  maxWidth: '20px',
//                  maxHeight: '20px',
//                  minWidth: '20px',
//                  minHeight: '20px',
//                }}
//              />
//            </Button>
//          </Tooltip>
//          <FovSelector
//            fovLabel={fovLabel}
//            setFovLabel={setFovLabel}
//            camera={camera_main}
//            cameras={cameras}
//            dispatch={dispatch}
//            disabled={isAnimated('FOV')}
//            applyAll={!isAnimated('FOV')}
//            isGlobal
//            globalFov={globalFov}
//            setGlobalFov={setGlobalFov}
//            setAllCameraFOV={setAllCameraFOV}
//            changeMain
//          />
//        </div>
//      )}
//      <div>
//        <div className="CameraPanel-row">
//          <Button
//            size="small"
//            variant="outlined"
//            startIcon={<AddAPhotoIcon />}
//            onClick={add_camera}
//          >
//            Add Camera
//          </Button>
//        </div>
//        <div className="CameraPanel-row">
//          <Tooltip
//            className="curve-button"
//            title="Toggle looping camera spline"
//          >
//            {!is_cycle ? (
//              <Button
//                size="small"
//                variant="outlined"
//                onClick={() => {
//                  setIsCycle(true);
//                }}
//              >
//                <GestureOutlined />
//              </Button>
//            ) : (
//              <Button
//                size="small"
//                variant="outlined"
//                onClick={() => {
//                  setIsCycle(false);
//                }}
//              >
//                <AllInclusiveOutlined />
//              </Button>
//            )}
//          </Tooltip>
//        </div>
//        <div className="CameraPanel-row">
//          <Tooltip title="Reset Keyframe Timing">
//            <Button
//              size="small"
//              variant="outlined"
//              onClick={() => {
//                const new_properties = new Map(cameraProperties);
//                cameras.forEach((camera, i) => {
//                  const uuid = camera.uuid;
//                  const new_time = i / (cameras.length - 1);
//                  const current_cam_properties = new_properties.get(uuid);
//                  current_cam_properties.set('TIME', new_time);
//                });
//                setCameraProperties(new_properties);
//              }}
//            >
//              <ClearAll />
//            </Button>
//          </Tooltip>
//        </div>
//      </div>
//      <div
//        className="CameraPanel-slider-container"
//        style={{ marginTop: '5px' }}
//      >
//        <Stack spacing={2} direction="row" sx={{ mb: 1 }} alignItems="center">
//          <p style={{ fontSize: 'smaller', color: '#999999' }}>Smoothness</p>
//          <ChangeHistory />
//          <Slider
//            value={smoothness_value}
//            step={step_size}
//            valueLabelFormat={smoothness_value.toFixed(2)}
//            min={0}
//            max={1}
//            onChange={(value: number) => {
//              set_smoothness_value(value);
//            }}
//          />
//          <RadioButtonUnchecked />
//        </Stack>
//      </div>
//      <div className="CameraPanel-slider-container">
//        <b style={{ fontSize: 'smaller', color: '#999999', textAlign: 'left' }}>
//          Camera Keyframes
//        </b>
//        <Slider
//          value={values}
//          step={step_size}
//          valueLabelDisplay="auto"
//          valueLabelFormat={(value, i) => {
//            if (cameras.length === 0) {
//              return '';
//            }
//            if (i === cameras.length && is_cycle) {
//              return `${cameras[0].properties.get('NAME')} @ ${parseFloat(
//                (value * seconds).toFixed(2),
//              )}s`;
//            }
//            return `${cameras[i].properties.get('NAME')} @ ${parseFloat(
//              (value * seconds).toFixed(2),
//            )}s`;
//          }}
//          marks={marks}
//          min={slider_min}
//          max={slider_max}
//          disabled={cameras.length < 2}
//          track={false}
//          onChange={handleKeyframeSlider}
//          sx={{
//            '& .MuiSlider-thumb': {
//              borderRadius: '6px',
//              width: `${24.0 / Math.max(Math.sqrt(cameras.length), 2)}px`,
//            },
//          }}
//          disableSwap
//        />
//        <b style={{ fontSize: 'smaller', color: '#999999', textAlign: 'left' }}>
//          Playback
//        </b>
//        <Slider
//          value={slider_value}
//          step={step_size}
//          valueLabelDisplay={is_playing ? 'on' : 'off'}
//          valueLabelFormat={`${(Math.min(slider_value, 1.0) * seconds).toFixed(
//            2,
//          )}s`}
//          marks={marks}
//          min={slider_min}
//          max={slider_max}
//          disabled={cameras.length < 2}
//          onChange={(event, value) => {
//            set_slider_value(value);
//          }}
//        />
//      </div>
//      <div className="CameraPanel-slider-button-container">
//        <Button
//          size="small"
//          variant="outlined"
//          onClick={() => {
//            setIsPlaying(false);
//            set_slider_value(slider_min);
//          }}
//        >
//          <FirstPage />
//        </Button>
//        <Button
//          size="small"
//          variant="outlined"
//          onClick={() =>
//            set_slider_value(Math.max(0.0, slider_value - step_size))
//          }
//        >
//          <ArrowBackIosNew />
//        </Button>
//        {/* eslint-disable-next-line no-nested-ternary */}
//        {!is_playing && slider_max === slider_value ? (
//          <Button
//            size="small"
//            variant="outlined"
//            onClick={() => {
//              set_slider_value(slider_min);
//            }}
//          >
//            <Replay />
//          </Button>
//        ) : !is_playing ? (
//          <Button
//            size="small"
//            variant="outlined"
//            onClick={() => {
//              if (cameras.length > 1) {
//                setIsPlaying(true);
//              }
//            }}
//          >
//            <PlayArrow />
//          </Button>
//        ) : (
//          <Button
//            size="small"
//            variant="outlined"
//            onClick={() => {
//              setIsPlaying(false);
//            }}
//          >
//            <Pause />
//          </Button>
//        )}
//        <Button
//          size="small"
//          variant="outlined"
//          onClick={() =>
//            set_slider_value(Math.min(slider_max, slider_value + step_size))
//          }
//        >
//          <ArrowForwardIos />
//        </Button>
//        <Button
//          size="small"
//          variant="outlined"
//          onClick={() => set_slider_value(slider_max)}
//        >
//          <LastPage />
//        </Button>
//      </div>
//      <div className="CameraList-container">
//        <CameraList
//          throttled_time_message_sender={throttled_time_message_sender}
//          sceneTree={sceneTree}
//          transform_controls={transform_controls}
//          camera_main={camera_render}
//          cameras={cameras}
//          setCameras={setCameras}
//          swapCameras={swapCameras}
//          cameraProperties={cameraProperties}
//          setCameraProperties={setCameraProperties}
//          fovLabel={fovLabel}
//          setFovLabel={setFovLabel}
//          isAnimated={isAnimated}
//          dispatch={dispatch}
//          slider_value={slider_value}
//          set_slider_value={set_slider_value}
//        />
//      </div>
//    </div>
//  );
//}