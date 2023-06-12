import { TreeItem } from "@mui/lab";
import { Box, IconButton } from "@mui/material";
import { VisibilityOffRounded, VisibilityRounded } from "@mui/icons-material";
import React from "react";
// import { CSS2DObject } from "three/examples/jsm/renderers/CSS2DRenderer";
import { UseSceneTree } from "../SceneTree";
// import { ViewerContext } from "..";

/** Control panel component for listing children of a scene node. */
function SceneNodeUIChildren(props: {
  name: string;
  useSceneTree: UseSceneTree;
}) {
  const children = props.useSceneTree(
    (state) => state.nodeFromName[props.name]?.children
  );
  if (children === undefined) return <></>;
  return (
    <>
      {children.map((child_id) => {
        return (
          <SceneNodeUI
            name={child_id}
            useSceneTree={props.useSceneTree}
            key={child_id}
          />
        );
      })}
    </>
  );
}

/** Control panel component for showing a particular scene node. */
export function SceneNodeUI(props: {
  name: string;
  useSceneTree: UseSceneTree;
}) {
  const sceneNode = props.useSceneTree(
    (state) => state.nodeFromName[props.name]
  );

  if (sceneNode === undefined) return <></>;

  // const { objFromSceneNodeNameRef } = React.useContext(ViewerContext)!;
  const visible = props.useSceneTree(
    (state) => state.attributesFromName[props.name]?.visibility
  );

  const setVisibility = props.useSceneTree((state) => state.setVisibility);
  const ToggleVisibilityIcon = visible
    ? VisibilityRounded
    : VisibilityOffRounded;

  const itemRef = React.useRef<HTMLElement>(null);

  // const labelRef = React.useRef<CSS2DObject>();

  // React.useEffect(() => {
  //   const threeObj = objFromSceneNodeNameRef.current[props.name];
  //   if (!threeObj) return;

  //   const labelDiv = document.createElement("div");
  //   labelDiv.style.cssText = `
  //     font-size: 0.7em;
  //     background-color: rgba(255, 255, 255, 0.9);
  //     padding: 0.5em;
  //     border-radius: 0.5em;
  //     color: #333;
  //   `;
  //   labelDiv.textContent = sceneNode.name;

  //   // TODO: the <Html /> component from @react-three/drei should be used instead of
  //   // vanilla threejs.
  //   const label = new CSS2DObject(labelDiv);
  //   labelRef.current = label;

  //   if (itemRef.current && itemRef.current.matches(":hover")) {
  //     threeObj.add(label);
  //   }
  //   return () => {
  //     threeObj.remove(label);
  //   };
  // }, [sceneNode.name, visible]);

  // // Flag for indicating when we're dragging across hide/show icons. Makes it
  // // easier to toggle visibility for many scene nodes at once.
  // const suppressMouseLeave = React.useRef(false);

  // const mouseEnter = (event: React.MouseEvent) => {
  //   // On hover, add an object label to the scene.
  //   const threeObj = objFromSceneNodeNameRef.current[props.name];
  //   threeObj && labelRef.current && threeObj.add(labelRef.current);
  //   event.stopPropagation();
  //   if (event.buttons !== 0) {
  //     suppressMouseLeave.current = true;
  //     setVisibility(props.name, !visible);
  //   }
  // };
  // const mouseLeave = (event: React.MouseEvent) => {
  //   // Remove the object label.
  //   const threeObj = objFromSceneNodeNameRef.current[props.name];
  //   threeObj && labelRef.current && threeObj.remove(labelRef.current);
  //   if (suppressMouseLeave.current) {
  //     suppressMouseLeave.current = false;
  //     return;
  //   }
  //   if (event.buttons !== 0) {
  //     setVisibility(props.name, !visible);
  //   }
  // };

  const hideShowIcon = (
    <IconButton
      onClick={(event) => {
        event.stopPropagation();
        setVisibility(props.name, !visible);
      }}
      // onMouseEnter={mouseEnter}
      // onMouseLeave={mouseLeave}
    >
      <ToggleVisibilityIcon />
    </IconButton>
  );
  const label = (
    <Box 
      component="div" 
      // onMouseEnter={mouseEnter} 
      // onMouseLeave={mouseLeave}
    >
      {sceneNode.name === "" ? "/" : sceneNode.name}
    </Box>
  );

  return (
    <TreeItem
      nodeId={"node_" + props.name.toString()}
      sx={{
        opacity: visible ? 1.0 : 0.5,
      }}
      ref={itemRef}
      icon={hideShowIcon}
      label={label}
    >
      <SceneNodeUIChildren
        name={props.name}
        useSceneTree={props.useSceneTree}
      />
    </TreeItem>
  );
}
