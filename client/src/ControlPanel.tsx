import { TreeItem, TreeView } from "@mui/lab";
import Tabs from "@mui/material/Tabs";
import Box from "@mui/material/Box";
import React, { RefObject } from "react";
import styled from "@emotion/styled";
import Tab from "@mui/material/Tab";
import { IconButton } from "@mui/material";
import { UseSceneTree, NodeIdType } from "./SceneTree";
import {
  Visibility,
  VisibilityOff,
  ExpandLessRounded,
  SensorsRounded,
  SensorsOffRounded,
} from "@mui/icons-material";
import { CSS2DObject } from "three/examples/jsm/renderers/CSS2DRenderer";
import { Leva, LevaPanel, useControls, useCreateStore } from "leva";
import { LevaCustomTheme } from "leva/dist/declarations/src/styles";
import { UseGui } from "./GuiState";

const levaTheme: LevaCustomTheme = {
  borderWidths: {
    root: "2px",
    input: "2px",
  },
  fonts: {
    mono: "",
    sans: "",
  },
  fontSizes: {
    root: "1em",
  },
  colors: {
    elevation1: "rgba(255, 255,255, 0)", //Titlebar.
    elevation2: "rgba(255, 255,255, 0)", // Main panel.
    elevation3: "#ffffff", // Inputs.
    accent1: "#ccc",
    accent2: "#e6e6e6",
    accent3: "#ccc",
    highlight1: "#333",
    highlight2: "#000",
    highlight3: "#000",
  },
};

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
  wrapperRef: RefObject<HTMLDivElement>;
  useSceneTree: UseSceneTree;
  useGui: UseGui;
}

/** Root component for control panel. Parents the websocket interface and a set
 * of control tabs. */
export default function ControlPanel(props: ControlPanelProps) {
  const ControlPanelWrapper = styled(Box)`
    box-sizing: border-box;
    width: 18em;
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

  return (
    <ControlPanelWrapper
      sx={{
        border: "1px solid",
        borderColor: "divider",
        "&.hidden": {
          height: "2.7em",
          overflow: "hidden",
        },
        "& .panel-contents": {
          opacity: "1.0",
          display: "block",
          transition: "all 0.5s",
        },
        "&.hidden .panel-contents": {
          opacity: "0.0",
          display: "none",
          transition: "opacity 0.5s",
        },
        "& .expand-icon": {
          transform: "rotate(0)",
        },
        "&.hidden .expand-icon": {
          transform: "rotate(180deg)",
        },
      }}
      ref={panelWrapperRef}
      className="hidden"
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
        <ControlPanelContents tab_labels={["Server", "Scene"]}>
          <Box component="div" sx={{ padding: "1em" }}>
            <CoreControl useGui={props.useGui} />
          </Box>
          <TreeView
            sx={{
              padding: "1em 2em 1em 1em",
            }}
          >
            <SceneNodeUI id={0} useSceneTree={props.useSceneTree} />
          </TreeView>
        </ControlPanelContents>
      </Box>
    </ControlPanelWrapper>
  );
}

interface CoreControlProps {
  useGui: UseGui;
}

/** One tab in the control panel. */
function CoreControl(props: CoreControlProps) {
  const server = props.useGui((state) => state.server);
  const setServer = props.useGui((state) => state.setServer);

  const label = props.useGui((state) => state.label);
  const setLabel = props.useGui((state) => state.setLabel);

  const levaStore = useCreateStore();
  useControls(
    {
      Label: { value: label, onChange: setLabel },
      Server: { value: server, onChange: setServer },
    },
    { store: levaStore }
  );
  return (
    <Box
      component="div"
      sx={{
        "& input": { color: "#777", border: "1px solid #ddd" },
        "& label": { color: "#777" },
      }}
    >
      <LevaPanel
        fill
        flat
        titleBar={false}
        theme={levaTheme}
        store={levaStore}
      />
    </Box>
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
      sx={{ overflow: "auto" }}
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
function SceneNodeUI(props: SceneNodeUIProp) {
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
    setVisible(threeObj.visible);
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
      sx={{
        opacity: visible ? 1.0 : 0.5,
      }}
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
      label={sceneNode.name === "" ? "/" : sceneNode.name}
    >
      <SceneNodeUIChildren id={props.id} useSceneTree={props.useSceneTree} />
    </TreeItem>
  );
}
