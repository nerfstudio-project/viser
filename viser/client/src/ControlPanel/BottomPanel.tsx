import { Box, Collapse, Paper } from "@mantine/core";
import React from "react";
import { FloatingPanelContext } from "./FloatingPanel";
import { useDisclosure } from "@mantine/hooks";

export default function BottomPanel({
  children,
}: {
  children: string | React.ReactNode;
}) {
  const panelWrapperRef = React.useRef<HTMLDivElement>(null);
  const [expanded, { toggle: toggleExpanded }] = useDisclosure(true);
  return (
    <FloatingPanelContext.Provider
      value={{
        wrapperRef: panelWrapperRef,
        expanded: expanded,
        toggleExpanded: toggleExpanded,
      }}
    >
      <Paper
        radius="0"
        withBorder
        sx={{
          boxSizing: "border-box",
          width: "100%",
          zIndex: 100,
          position: "fixed",
          bottom: 0,
          left: 0,
          margin: 0,
          overflow: "scroll",
          minHeight: "3.5em",
          maxHeight: "60%",
          transition: "height 0.3s linear",
        }}
        ref={panelWrapperRef}
      >
        {children}
      </Paper>
    </FloatingPanelContext.Provider>
  );
}
BottomPanel.Handle = function BottomPanelHandle({
  children,
}: {
  children: string | React.ReactNode;
}) {
  const panelContext = React.useContext(FloatingPanelContext)!;
  return (
    <Box
      color="red"
      sx={(theme) => ({
        backgroundColor:
          theme.colorScheme == "dark"
            ? theme.colors.dark[5]
            : theme.colors.gray[1],
        lineHeight: "2.5em",
        cursor: "pointer",
        position: "relative",
        fontWeight: 400,
        userSelect: "none",
      })}
      onClick={() => {
        panelContext.toggleExpanded();
      }}
    >
      <Box
        component="div"
        sx={{
          padding: "0.5em 3em 0.5em 0.8em",
        }}
      >
        {children}
      </Box>
    </Box>
  );
};
/** Contents of a panel. */
BottomPanel.Contents = function BottomPanelContents({
  children,
}: {
  children: string | React.ReactNode;
}) {
  const panelContext = React.useContext(FloatingPanelContext)!;
  return <Collapse in={panelContext.expanded}>{children}</Collapse>;
};
