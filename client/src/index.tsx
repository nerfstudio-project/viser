import styled from "@emotion/styled";
import { OrbitControls, Stats } from "@react-three/drei";
import { Canvas } from "@react-three/fiber";
import React from "react";
import { createRoot } from "react-dom/client";

import ControlPanel from "./ControlPanel";
import LabelRenderer from "./LabelRenderer";
import { SceneNodeThreeObject, useSceneTreeState } from "./SceneTree";

import "./index.css";

import Box from "@mui/material/Box";

function Root() {
  // Layout and styles.
  const Wrapper = styled(Box)`
    width: 100%;
    height: 100%;
    position: relative;
  `;

  const Viewport = styled(Canvas)`
    position: relative;
    z-index: 0;

    background-color: #fff;

    width: 100%;
    height: 100%;
  `;

  // Our 2D label renderer needs access to the div used for rendering.
  const wrapperRef = React.useRef<HTMLDivElement>(null);

  // Declare the scene tree state. This returns a zustand store/hook, which we
  // can pass to any children that need state access.
  const useSceneTree = useSceneTreeState();

  return (
    <Wrapper ref={wrapperRef}>
      <ControlPanel useSceneTree={useSceneTree} />
      <Viewport>
        <Stats showPanel={0} className="stats" />
        <LabelRenderer wrapperRef={wrapperRef} />
        <OrbitControls
          minDistance={0.5}
          maxDistance={200.0}
          enableDamping={true}
          dampingFactor={0.1}
        />
        <SceneNodeThreeObject id={0} useSceneTree={useSceneTree} />
        <gridHelper args={[10.0, 10]} />
      </Viewport>
    </Wrapper>
  );
}

createRoot(document.getElementById("root") as HTMLElement).render(<Root />);
