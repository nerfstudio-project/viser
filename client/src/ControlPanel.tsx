import { TreeItem, TreeView } from "@mui/lab";
import Tabs from "@mui/material/Tabs";
import Box from "@mui/material/Box";
import React, { MutableRefObject, RefObject } from "react";
import styled from "@emotion/styled";
import Tab from "@mui/material/Tab";
import {
  FormControl,
  IconButton,
  Button,
  InputLabel,
  OutlinedInput,
} from "@mui/material";
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
  // This will currently re-initialize the websocket connection whenever we toggle. Unideal...
  const hidden = React.useRef(true);

  const ControlPanelWrapper = styled(Box)`
    box-sizing: border-box;
    position: absolute;
    width: 20em;
    z-index: 1;
    top: 0;
    right: 0;
    background-color: rgba(255, 255, 255, 0.85);
    overflow-y: scroll;
    max-height: 100%;
    margin: 0;
  `;

  const ControlPanelHandle = styled(Box)`
    line-height: 2em;
    text-align: center;
    padding: 0 1em;
    cursor: pointer;
    font-weight: bold;
  `;

  const [server, setServer] = React.useState("ws://localhost:8080");
  const [label, setLabel] = React.useState("");

  const contentsRef = React.useRef<HTMLDivElement>(null);
  const handleRef = React.useRef<HTMLDivElement>(null);

  useWebsocketInterface(
    props.useSceneTree,
    props.websocketRef,
    props.wrapperRef,
    server,
    () => {
      if (handleRef.current === null) return;
      handleRef.current.style.color = "green";
      handleRef.current.style.backgroundColor = "lightgreen";
    },
    () => {
      if (handleRef.current === null) return;
      handleRef.current.style.color = "darkred";
      handleRef.current.style.backgroundColor = "lightpink";
    }
  );

  const idLabel = React.useId();
  const idServer = React.useId();

  return (
    <ControlPanelWrapper
      sx={{
        borderLeftWidth: "1px",
        borderLeftStyle: "solid",
        borderLeftColor: "divider",
        borderBottomWidth: "1px",
        borderBottomStyle: "solid",
        borderBottomColor: "divider",
      }}
    >
      <ControlPanelHandle
        ref={handleRef}
        onClick={() => {
          if (hidden.current) {
            contentsRef.current!.style.display = "block";
          } else {
            // Throw the contents off the screen. Setting display: none produces a MUI error.
            contentsRef.current!.style.display = "none";
          }
          hidden.current = !hidden.current;
        }}
      >
        {label === "" ? server : label}
      </ControlPanelHandle>
      <Box
        component="div"
        ref={contentsRef}
        // Hidden by default.
        sx={{ display: "none" }}
      >
        <ControlPanelContents tab_labels={["Core", "Scene"]}>
          <Box component="div" sx={{ py: 4 }}>
            <form
              onSubmit={(event) => {
                event.preventDefault();
                setLabel(
                  (document.getElementById(idLabel) as HTMLInputElement).value
                );
                setServer(
                  (document.getElementById(idServer) as HTMLInputElement).value
                );
              }}
            >
              <FormControl fullWidth>
                <InputLabel htmlFor={idLabel}>Label</InputLabel>
                <OutlinedInput
                  id={idLabel}
                  defaultValue={label}
                  label="Label"
                />
              </FormControl>
              <FormControl fullWidth margin="normal">
                <InputLabel htmlFor={idServer}>Server</InputLabel>
                <OutlinedInput
                  id={idServer}
                  defaultValue={server}
                  label="Server"
                />
              </FormControl>
              <FormControl fullWidth margin="normal">
                <Button variant="contained" fullWidth type="submit">
                  Save
                </Button>
              </FormControl>
            </form>
            <Button
              variant="outlined"
              fullWidth
              type="submit"
              onClick={() => {
                if (
                  !props.wrapperRef.current!.style.backgroundImage.startsWith(
                    "url("
                  )
                ) {
                  // TODO: we should consider hiding this button if there's no background available.
                  alert("No background to download!");
                  return;
                }
                const data =
                  props.wrapperRef.current!.style.backgroundImage.split('"')[1];
                const link = document.createElement("a");
                link.download = "background";
                link.href = data;
                document.body.appendChild(link);
                link.click();
                document.body.removeChild(link);
              }}
            >
              Download Background
            </Button>
          </Box>
          <TreeView
            sx={{
              paddingRight: "1em",
            }}
          >
            <SceneNodeUI id={0} useSceneTree={props.useSceneTree} />
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
      label={sceneNode.name === "" ? "/" : sceneNode.name}
    >
      <SceneNodeUIChildren id={props.id} useSceneTree={props.useSceneTree} />
    </TreeItem>
  );
}
