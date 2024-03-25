import { ViewerContext } from "../App";
import { Box, Flex, Paper, Table } from "@mantine/core";
import { IconCaretDown } from "@tabler/icons-react";
import React from "react";
import { icon, tableWrapper } from "./SceneTreeTable.css";

/* Table for seeing an overview of the scene tree, toggling visibility, etc. * */
export default function SceneTreeTable(props: { compact: boolean }) {
  const viewer = React.useContext(ViewerContext)!;

  const nodeFromName = viewer.useSceneTree((state) => state.nodeFromName);
  const setLabelVisibility = viewer.useSceneTree(
    (state) => state.setLabelVisibility,
  );
  function setOverrideVisibility(name: string, visible: boolean) {
    const attr = viewer.nodeAttributesFromName.current;
    if (attr[name] === undefined) attr[name] = {};
    attr[name]!.overrideVisibility = visible;
  }

  return (
    <Box className={tableWrapper}>
      {Object.keys(nodeFromName).map((name, index) => (
        <Flex key={name}>
          <Box>
            <IconCaretDown className={icon} />
          </Box>
          <Box>{name}</Box>
        </Flex>
      ))}
    </Box>
  );
}
