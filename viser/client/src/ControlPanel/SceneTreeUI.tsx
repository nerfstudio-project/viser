import { TreeItem } from "@mui/lab";
import { Box, IconButton } from "@mui/material";
import { VisibilityOffRounded, VisibilityRounded } from "@mui/icons-material";
import React from "react";
import { UseSceneTree } from "../SceneTree";

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

  const visible = props.useSceneTree(
    (state) => state.attributesFromName[props.name]?.visibility
  );

  const setVisibility = props.useSceneTree((state) => state.setVisibility);
  const ToggleVisibilityIcon = visible
    ? VisibilityRounded
    : VisibilityOffRounded;


  const itemRef = React.useRef<HTMLElement>(null);

  const { setLabelVisibility } = props.useSceneTree();
  const mouseEnter = () => {
    setLabelVisibility(sceneNode.name, true);
  }
  const mouseLeave = () => {
    setLabelVisibility(sceneNode.name, false);
  }

  const hideShowIcon = (
    <IconButton
      onClick={(event) => {
        event.stopPropagation();
        setVisibility(props.name, !visible);
      }}
    >
      <ToggleVisibilityIcon />
    </IconButton>
  );
  const label = (
    <Box 
      component="div" 
    >
      {sceneNode.name === "" ? "/" : sceneNode.name}
    </Box>
  );

  return (
    <div
      onMouseEnter={mouseEnter}
      onMouseLeave={mouseLeave}
    >
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
    </div>
  );
}
