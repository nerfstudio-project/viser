import { Box, Collapse, Divider, Paper } from "@mantine/core";
import React from "react";
import { useDisclosure } from "@mantine/hooks";

const BottomPanelContext = React.createContext<null | {
  wrapperRef: React.RefObject<HTMLDivElement>;
  expanded: boolean;
  toggleExpanded: () => void;
}>(null);

/** A bottom panel is used to display the controls on mobile devices. */
export default function BottomPanel({
  children,
}: {
  children: string | React.ReactNode;
}) {
  const panelWrapperRef = React.useRef<HTMLDivElement>(null);
  const [expanded, { toggle: toggleExpanded }] = useDisclosure(true);
  return (
    <BottomPanelContext.Provider
      value={{
        wrapperRef: panelWrapperRef,
        expanded: expanded,
        toggleExpanded: toggleExpanded,
      }}
    >
      <>
        <Divider
          style={{
            position: "fixed",
            bottom: "0",
            left: "0",
            width: "100%",
            zIndex: 11,
          }}
        />
        <Paper
          radius="0"
          style={{
            boxSizing: "border-box",
            width: "100%",
            zIndex: 10,
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
      </>
    </BottomPanelContext.Provider>
  );
}
BottomPanel.Handle = function BottomPanelHandle({
  children,
}: {
  children: string | React.ReactNode;
}) {
  const panelContext = React.useContext(BottomPanelContext)!;
  return (
    <Box
      color="red"
      style={{
        cursor: "pointer",
        position: "relative",
        fontWeight: 400,
        userSelect: "none",
        display: "flex",
        alignItems: "center",
        padding: "0 0.8em",
        height: "3.5em",
      }}
      onClick={() => {
        panelContext.toggleExpanded();
      }}
    >
      {children}
      {panelContext.expanded && <Divider />}
    </Box>
  );
};

/** Contents of a panel. */
BottomPanel.Contents = function BottomPanelContents({
  children,
}: {
  children: string | React.ReactNode;
}) {
  const panelContext = React.useContext(BottomPanelContext)!;
  return <Collapse in={panelContext.expanded}>{children}</Collapse>;
};

/** Hides contents when panel is collapsed. */
BottomPanel.HideWhenCollapsed = function BottomPanelHideWhenCollapsed({
  children,
}: {
  children: React.ReactNode;
}) {
  const expanded = React.useContext(BottomPanelContext)?.expanded ?? true;
  return expanded ? children : null;
};
