import { ViewerContext } from "../App";
import { Box, ScrollArea, Stack, Tooltip } from "@mantine/core";
import {
  IconCaretDown,
  IconCaretRight,
  IconEye,
  IconEyeOff,
} from "@tabler/icons-react";
import React from "react";
import { caretIcon, tableRow, tableWrapper } from "./SceneTreeTable.css";
import { useDisclosure } from "@mantine/hooks";

/* Table for seeing an overview of the scene tree, toggling visibility, etc. * */
export default function SceneTreeTable() {
  const viewer = React.useContext(ViewerContext)!;
  const childrenName = viewer.useSceneTree(
    (state) => state.nodeFromName[""]!.children,
  );
  return (
    <ScrollArea className={tableWrapper}>
      {childrenName.map((name) => (
        <SceneTreeTableRow
          nodeName={name}
          key={name}
          isParentVisible={true}
          indentCount={0}
        />
      ))}
    </ScrollArea>
  );
}

const SceneTreeTableRow = React.memo(function SceneTreeTableRow(props: {
  nodeName: string;
  isParentVisible: boolean;
  indentCount: number;
}) {
  const viewer = React.useContext(ViewerContext)!;
  const childrenName = viewer.useSceneTree(
    (state) => state.nodeFromName[props.nodeName]!.children,
  );
  const expandable = childrenName.length > 0;

  const [expanded, { toggle: toggleExpanded }] = useDisclosure(false);

  function setOverrideVisibility(name: string, visible: boolean | undefined) {
    const attr = viewer.nodeAttributesFromName.current;
    attr[name]!.overrideVisibility = visible;
    rerenderTable();
  }
  const setLabelVisibility = viewer.useSceneTree(
    (state) => state.setLabelVisibility,
  );

  // For performance, scene node visibility is stored in a ref instead of the
  // zustand state. This means that re-renders for the table need to be
  // triggered manually when visibilities are updated.
  const [, setTime] = React.useState(Date.now());
  function rerenderTable() {
    setTime(Date.now());
  }
  React.useEffect(() => {
    const interval = setInterval(rerenderTable, 200);
    return () => {
      clearInterval(interval);
    };
  }, []);

  const attrs = viewer.nodeAttributesFromName.current[props.nodeName];
  const isVisible =
    (attrs?.overrideVisibility === undefined
      ? attrs?.visibility
      : attrs.overrideVisibility) ?? true;
  const isVisibleEffective = isVisible && props.isParentVisible;
  const VisibleIcon = isVisible ? IconEye : IconEyeOff;

  return (
    <>
      <Box
        className={tableRow}
        style={{
          cursor: expandable ? "pointer" : undefined,
          marginLeft: (props.indentCount * 0.75).toString() + "em",
        }}
        onClick={expandable ? toggleExpanded : undefined}
        onMouseOver={() => setLabelVisibility(props.nodeName, true)}
        onMouseOut={() => setLabelVisibility(props.nodeName, false)}
      >
        <Box
          style={{
            opacity: expandable ? 1 : 0.3,
          }}
        >
          {expanded ? (
            <IconCaretDown className={caretIcon} />
          ) : (
            <IconCaretRight className={caretIcon} />
          )}
        </Box>
        <Box style={{ width: "1.5em", height: "1.5em" }}>
          <Tooltip label="Override visibility">
            <VisibleIcon
              style={{
                cursor: "pointer",
                opacity: isVisibleEffective ? 0.85 : 0.25,
                width: "1.5em",
                height: "1.5em",
              }}
              onClick={(evt) => {
                evt.stopPropagation();
                setOverrideVisibility(props.nodeName, !isVisible);
              }}
            />
          </Tooltip>
        </Box>
        <Box>
          {props.nodeName
            .split("/")
            .filter((part) => part.length > 0)
            .map((part, index, all) => (
              // We set userSelect to prevent users from accidentally
              // selecting text when dragging over the hide/show icons.
              <span key={index} style={{ userSelect: "none" }}>
                <span style={{ opacity: "0.3" }}>
                  {index === all.length - 1 ? "/" : `/${part}`}
                </span>
                {index === all.length - 1 ? part : ""}
              </span>
            ))}
        </Box>
      </Box>
      {expanded
        ? childrenName.map((name) => (
            <SceneTreeTableRow
              nodeName={name}
              isParentVisible={isVisibleEffective}
              key={name}
              indentCount={props.indentCount + 1}
            />
          ))
        : null}
    </>
  );
});
