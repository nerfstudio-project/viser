import { TreeItem, TreeView } from "@mui/lab";
import Tabs from "@mui/material/Tabs";
import Box from "@mui/material/Box";
import React, { MutableRefObject, RefObject } from "react";
import styled from "@emotion/styled";
import Tab from "@mui/material/Tab";
import { IconButton } from "@mui/material";
import { UseSceneTree, NodeIdType } from "./SceneTree";
import { Visibility, VisibilityOff } from "@mui/icons-material";
import { CSS2DObject } from "three/examples/jsm/renderers/CSS2DRenderer";
import useWebsocketInterface from "./WebsocketInterface";

interface ControlPanelProps {
  wrapperRef: RefObject<HTMLDivElement>;
  websocketRef: MutableRefObject<WebSocket | null>;
  useSceneTree: UseSceneTree;
}

/** Root component for control panel. Parents the websocket interface and a set
 * of control tabs. */
export default function ControlPanel(props: ControlPanelProps) {
  const [hidden, setHidden] = React.useState(false);

  const ControlPanelWrapper = styled(Box)`
    box-sizing: content-box;
    position: absolute;
    width: 20em;
    z-index: 1;
    top: 0;
    right: 0;
    background-color: rgba(255, 255, 255, 0.9);
  `;

  const ControlPanelHandle = styled(Box)`
    height: 3em;
    cursor: pointer;
  `;

  const connected = useWebsocketInterface(
    props.useSceneTree,
    props.websocketRef
  );
  // const [parent, setParent] = React.useState<HTMLDivElement>();
  //
  // useEffect(() => {
  //   props.wrapperRef.current && setParent(props.wrapperRef.current);
  // });

  return (
    <ControlPanelWrapper
      sx={{
        ...(hidden
          ? { height: "3em", overflowY: "clip" }
          : { height: "100%", overflowY: "scroll" }),
        borderLeftWidth: "1px",
        borderLeftStyle: "solid",
        borderLeftColor: "divider",
      }}
    >
      <ControlPanelHandle
        sx={{ backgroundColor: connected ? "lightgreen" : "lightpink" }}
        onClick={() => {
          setHidden(!hidden);
        }}
      />
      <ControlPanelContents tab_labels={["Scene"]}>
        <TreeView
          sx={{
            paddingRight: "1em",
          }}
        >
          <SceneNodeUIChildren id={0} useSceneTree={props.useSceneTree} />
        </TreeView>
      </ControlPanelContents>
    </ControlPanelWrapper>
  );
}

interface ControlPanelTabContentsProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

/** One tab in the control panel. */
function ControlPanelTabContents(props: ControlPanelTabContentsProps) {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`simple-tabpanel-${index}`}
      aria-labelledby={`simple-tab-${index}`}
      {...other}
    >
      {children}
    </div>
  );
}

interface ControlPanelContentsProps {
  children?: React.ReactNode;
  tab_labels: string[];
}

/** Wrapper for tabulated control panel interface. */
function ControlPanelContents(props: ControlPanelContentsProps) {
  const [tabState, setTabState] = React.useState(0);
  const handleChange = (_event: React.SyntheticEvent, newValue: number) => {
    setTabState(newValue);
  };
  const arrayChildren = React.Children.toArray(props.children);

  // Our wrapper box needs a component prop set for type inference; the
  // typescript compiler will complain without it.
  return (
    <Box
      component="div"
      sx={{
        paddingLeft: "1em",
        paddingRight: "1em",
      }}
    >
      <Tabs
        value={tabState}
        onChange={handleChange}
        sx={{
          borderBottom: 1,
          borderColor: "divider",
        }}
      >
        {props.tab_labels.map((value, index) => {
          return <Tab label={value} key={index} />;
        })}
      </Tabs>

      {arrayChildren.map((child, index) => (
        <ControlPanelTabContents value={tabState} index={index} key={index}>
          {child}
        </ControlPanelTabContents>
      ))}
    </Box>
  );
}

interface SceneNodeUIChildrenProp {
  id: NodeIdType;
  useSceneTree: UseSceneTree;
}

/** Control panel component for listing children of a scene node. */
export function SceneNodeUIChildren(props: SceneNodeUIChildrenProp) {
  const children = props.useSceneTree(
    (state) => state.nodeFromId[props.id].children
  );
  return (
    <>
      {children.map((child_id) => {
        return (
          <SceneNodeUI
            id={child_id}
            useSceneTree={props.useSceneTree}
            key={child_id}
          />
        );
      })}
    </>
  );
}

interface SceneNodeUIProp {
  id: NodeIdType;
  useSceneTree: UseSceneTree;
}

/** Control panel component for showing a particular scene node. */
export function SceneNodeUI(props: SceneNodeUIProp) {
  const sceneNode = props.useSceneTree((state) => state.nodeFromId[props.id]);
  const threeObj = props.useSceneTree((state) => state.objFromId[props.id]);
  const [visible, setVisible] = React.useState(true);
  const ToggleVisibilityIcon = visible ? Visibility : VisibilityOff;

  const labelDiv = document.createElement("div");
  labelDiv.style.cssText = `
    font-size: 0.7em;
    background-color: rgba(255, 255, 255, 0.9);
    padding: 0.5em;
    border-radius: 0.5em;
    color: #333;
  `;
  labelDiv.textContent = sceneNode.name;
  const label = new CSS2DObject(labelDiv);
  const itemRef = React.useRef<HTMLElement>(null);

  React.useEffect(() => {
    if (threeObj === undefined) return;
    if (threeObj === null) return;
    if (itemRef.current!.matches(":hover")) {
      threeObj.add(label);
    }
    threeObj.visible = visible;
    return () => {
      threeObj.remove(label);
    };
  });

  // Flag for indicating when we're dragging across hide/show icons. Makes it
  // easier to toggle visibility for many scene nodes at once.
  const suppressMouseLeave = React.useRef(false);

  // TODO: it would be nice to get rid of the graphical UI for hiding nodes,
  // and instead just add the ability to filter scene nodes via regex. (similar
  // to Tensorboard)

  const hideShowIcon = (
    <IconButton
      onClick={(event) => {
        threeObj.visible = !threeObj.visible;
        setVisible(threeObj.visible);
        event.stopPropagation();
      }}
    >
      <ToggleVisibilityIcon />
    </IconButton>
  );

  return (
    <TreeItem
      nodeId={"node_" + props.id.toString()}
      style={{ opacity: visible ? 1.0 : 0.5 }}
      ref={itemRef}
      icon={hideShowIcon}
      onMouseEnter={(event) => {
        // On hover, add an object label to the scene.
        threeObj.add(label);
        event.stopPropagation();
        if (event.buttons !== 0) {
          threeObj.visible = !threeObj.visible;
          setVisible(threeObj.visible);
          suppressMouseLeave.current = true;
        }
      }}
      onMouseLeave={(event) => {
        // Remove the object label.
        threeObj.remove(label);
        if (suppressMouseLeave.current) {
          suppressMouseLeave.current = false;
          return;
        }
        if (event.buttons !== 0) {
          threeObj.visible = !threeObj.visible;
          setVisible(threeObj.visible);
        }
      }}
      label={sceneNode.name}
    >
      <SceneNodeUIChildren id={props.id} useSceneTree={props.useSceneTree} />
    </TreeItem>
  );
}
