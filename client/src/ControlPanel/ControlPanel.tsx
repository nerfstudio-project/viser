import { TreeItem, TreeView } from "@mui/lab";
import Tabs from "@mui/material/Tabs";
import Box from "@mui/material/Box";
import React, { MutableRefObject, RefObject } from "react";
import styled from "@emotion/styled";
import Tab from "@mui/material/Tab";
import { IconButton } from "@mui/material";
import { UseSceneTree } from "../SceneTree";
import {
  Visibility,
  VisibilityOff,
  ExpandLessRounded,
  SensorsRounded,
  SensorsOffRounded,
} from "@mui/icons-material";
import { CSS2DObject } from "three/examples/jsm/renderers/CSS2DRenderer";
import { UseGui } from "./GuiState";
import GeneratedControls from "./Generated";
import ServerControls from "./Server";

interface ConnectedStatusProps {
  useGui: UseGui;
}

/* Icon and label telling us the current status of the websocket connection. */
function ConnectionStatus(props: ConnectedStatusProps) {
  const connected = props.useGui((state) => state.websocketConnected);
  const server = props.useGui((state) => state.server);
  const label = props.useGui((state) => state.label);

  const StatusIcon = connected ? SensorsRounded : SensorsOffRounded;
  return (
    <>
      <StatusIcon
        htmlColor={connected ? "#0b0" : "#b00"}
        style={{ transform: "translateY(0.25em) scale(1.2)" }}
      />
      &nbsp; &nbsp;
      {label === "" ? server : label}
    </>
  );
}

interface ControlPanelProps {
  useSceneTree: UseSceneTree;
  useGui: UseGui;
  websocketRef: MutableRefObject<WebSocket | null>;
  wrapperRef: RefObject<HTMLDivElement>;
}

/** Root component for control panel. Parents the websocket interface and a set
 * of control tabs. */
export default function ControlPanel(props: ControlPanelProps) {
  const ControlPanelWrapper = styled(Box)`
    box-sizing: border-box;
    width: 20em;
    z-index: 1;
    position: absolute;
    top: 1em;
    right: 1em;
    margin: 0;
    border-radius: 0.5em;
    max-height: 85%;
    overflow: auto;
    background-color: rgba(255, 255, 255, 0.9);
    box-sizing: border-box;
  `;

  const ControlPanelHandle = styled(Box)`
    line-height: 1.5em;
    cursor: pointer;
    position: relative;
    font-weight: 400;
    color: #777;
    box-sizing: border-box;
    overflow: hidden;
  `;

  const panelWrapperRef = React.useRef<HTMLDivElement>();

  const showGenerated = props.useGui((state) => state.guiNames.length > 0);

  return (
    <ControlPanelWrapper
      sx={{
        border: "1px solid",
        borderColor: "divider",
        "&.hidden": {
          overflow: "hidden",
        },
        "& .panel-contents": {
          opacity: "1.0",
          visibility: "visible",
          maxHeight: "200em",
          transition:
            "visibility 0.1s linear,opacity 0.1s linear, max-height 0.2s ease-in",
        },
        "&.hidden .panel-contents": {
          opacity: "0.0",
          visibility: "hidden",
          maxHeight: "0 !important",
          overflow: "hidden",
          transition:
            "visibility 0.1s linear,opacity 0.1s linear, max-height 0.2s ease-out",
        },
        "& .expand-icon": {
          transform: "rotate(0)",
        },
        "&.hidden .expand-icon": {
          transform: "rotate(180deg)",
        },
      }}
      ref={panelWrapperRef}
    >
      <ControlPanelHandle
        onClick={() => {
          const wrapper = panelWrapperRef.current!;

          // We use a class instead of state for tracking hidden status.
          // No re-render => we can add a (hacky) animation!
          if (wrapper.classList.contains("hidden")) {
            wrapper.classList.remove("hidden");
          } else {
            wrapper.classList.add("hidden");
          }
        }}
      >
        <Box
          component="div"
          sx={{
            padding: "0.2em 3em 0.5em 1em",
          }}
        >
          <ConnectionStatus useGui={props.useGui} />
        </Box>
        <Box
          component="div"
          sx={{
            position: "absolute",
            top: "50%",
            right: "1em",
            transform: "translateY(-48%) scale(1.2)",
            height: "1.5em",
          }}
        >
          <ExpandLessRounded color="action" className="expand-icon" />
        </Box>
      </ControlPanelHandle>
      <Box
        component="div"
        sx={{
          borderTop: "1px solid",
          borderTopColor: "divider",
        }}
        className="panel-contents"
      >
        <ControlPanelContents
          tab_labels={
            showGenerated ? ["Control", "Server", "Scene"] : ["Server", "Scene"]
          }
        >
          {showGenerated ? (
            <Box component="div" sx={{ padding: "0.5em" }}>
              <GeneratedControls
                useGui={props.useGui}
                websocketRef={props.websocketRef}
              />
            </Box>
          ) : null}
          <Box component="div" sx={{ padding: "0.5em" }}>
            <ServerControls
              useGui={props.useGui}
              wrapperRef={props.wrapperRef}
            />
          </Box>
          <TreeView
            sx={{
              padding: "1em 2em 1em 1em",
            }}
          >
            <SceneNodeUI name="" useSceneTree={props.useSceneTree} />
          </TreeView>
        </ControlPanelContents>
      </Box>
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
    <Box
      component="div"
      role="tabpanel"
      hidden={value !== index}
      id={`simple-tabpanel-${index}`}
      aria-labelledby={`simple-tab-${index}`}
      sx={{
        overflow: "auto",
        backgroundColor: "#fff",
      }}
      {...other}
    >
      {children}
    </Box>
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
    <>
      <Box
        component="div"
        sx={{
          paddingLeft: "1em",
          paddingRight: "1em",
          borderBottom: 1,
          borderColor: "divider",
        }}
      >
        <Tabs value={tabState} onChange={handleChange}>
          {props.tab_labels.map((value, index) => {
            return (
              <Tab label={value} key={index} sx={{ fontSize: "0.75em" }} />
            );
          })}
        </Tabs>
      </Box>

      {arrayChildren.map((child, index) => (
        <ControlPanelTabContents value={tabState} index={index} key={index}>
          {child}
        </ControlPanelTabContents>
      ))}
    </>
  );
}

interface SceneNodeUIChildrenProp {
  name: string;
  useSceneTree: UseSceneTree;
}

/** Control panel component for listing children of a scene node. */
export function SceneNodeUIChildren(props: SceneNodeUIChildrenProp) {
  const children = props.useSceneTree(
    (state) => state.nodeFromName[props.name].children
  );
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

interface SceneNodeUIProp {
  name: string;
  useSceneTree: UseSceneTree;
}

/** Control panel component for showing a particular scene node. */
function SceneNodeUI(props: SceneNodeUIProp) {
  const sceneNode = props.useSceneTree((state) => state.nodeFromName[props.name]);
  const threeObj = props.useSceneTree((state) => state.objFromName[props.name]);

  const visible = props.useSceneTree(
    (state) => state.visibilityFromName[props.name]
  );
  const setVisibility = props.useSceneTree((state) => state.setVisibility);
  const ToggleVisibilityIcon = visible ? Visibility : VisibilityOff;

  const itemRef = React.useRef<HTMLElement>(null);

  const labelRef = React.useRef<CSS2DObject>();

  React.useEffect(() => {
    if (threeObj === undefined) return;
    if (threeObj === null) return;

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
    labelRef.current = label;

    if (itemRef.current!.matches(":hover")) {
      threeObj.add(label);
    }

    threeObj.visible = visible;
    return () => {
      threeObj.remove(label);
    };
  }, [threeObj, sceneNode.name, visible]);

  // Flag for indicating when we're dragging across hide/show icons. Makes it
  // easier to toggle visibility for many scene nodes at once.
  const suppressMouseLeave = React.useRef(false);

  const mouseEnter = (event: React.MouseEvent) => {
    // On hover, add an object label to the scene.
    console.log("mouse enter" + sceneNode.name);
    threeObj.add(labelRef.current!);
    event.stopPropagation();
    if (event.buttons !== 0) {
      suppressMouseLeave.current = true;
      setVisibility(props.name, !visible);
    }
  };
  const mouseLeave = (event: React.MouseEvent) => {
    // Remove the object label.
    threeObj.remove(labelRef.current!);
    if (suppressMouseLeave.current) {
      suppressMouseLeave.current = false;
      return;
    }
    if (event.buttons !== 0) {
      setVisibility(props.name, !visible);
    }
  };

  const hideShowIcon = (
    <IconButton
      onClick={(event) => {
        event.stopPropagation();
        setVisibility(props.name, !visible);
      }}
      onMouseEnter={mouseEnter}
      onMouseLeave={mouseLeave}
    >
      <ToggleVisibilityIcon />
    </IconButton>
  );
  const label = (
    <Box component="div" onMouseEnter={mouseEnter} onMouseLeave={mouseLeave}>
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
      <SceneNodeUIChildren name={props.name} useSceneTree={props.useSceneTree} />
    </TreeItem>
  );
}
