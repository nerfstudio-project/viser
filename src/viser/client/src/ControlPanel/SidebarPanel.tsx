// @refresh reset

import { ActionIcon, Box, Paper, ScrollArea, Tooltip } from "@mantine/core";
import React from "react";
import { useDisclosure } from "@mantine/hooks";
import { IconChevronLeft, IconChevronRight } from "@tabler/icons-react";

export const SidebarPanelContext = React.createContext<null | {
  collapsible: boolean;
  toggleCollapsed: () => void;
}>(null);

/** A fixed or collapsible side panel for displaying controls. */
export default function SidebarPanel({
  children,
  collapsible,
}: {
  children: string | React.ReactNode;
  collapsible: boolean;
}) {
  const [collapsed, { toggle: toggleCollapsed }] = useDisclosure(false);

  const collapsedView = (
    <Box
      sx={(theme) => ({
        /* Animate in when collapsed. */
        position: "absolute",
        top: "0em",
        right: collapsed ? "0em" : "-3em",
        transitionProperty: "right",
        transitionDuration: "0.5s",
        transitionDelay: "0.25s",
        /* Visuals. */
        borderBottomLeftRadius: "0.5em",
        backgroundColor:
          theme.colorScheme == "dark"
            ? theme.colors.dark[5]
            : theme.colors.gray[2],
        padding: "0.5em",
      })}
    >
      <ActionIcon
        onClick={(evt) => {
          evt.stopPropagation();
          toggleCollapsed();
        }}
      >
        <Tooltip zIndex={100} label={"Show sidebar"}>
          {<IconChevronLeft />}
        </Tooltip>
      </ActionIcon>
    </Box>
  );

  return (
    <SidebarPanelContext.Provider
      value={{
        collapsible: collapsible,
        toggleCollapsed: toggleCollapsed,
      }}
    >
      {collapsedView}
      {/* Using an <Aside /> below will break Mantine color inputs. */}
      <Paper
        shadow="xl"
        component={ScrollArea}
        sx={{
          width: collapsed ? 0 : "20em",
          boxSizing: "content-box",
          transition: "width 0.5s 0s",
          zIndex: 10,
        }}
      >
        <Box
          /* Prevent DOM reflow, as well as internals from getting too wide.
           * Hardcoded to match the width of the wrapper element above. */
          w="20em"
        >
          {children}
        </Box>
      </Paper>
    </SidebarPanelContext.Provider>
  );
}

/** Handle object helps us hide, show, and drag our panel.*/
SidebarPanel.Handle = function SidebarPanelHandle({
  children,
}: {
  children: string | React.ReactNode;
}) {
  const { toggleCollapsed, collapsible } =
    React.useContext(SidebarPanelContext)!;

  const collapseSidebarToggleButton = (
    <ActionIcon
      onClick={(evt) => {
        evt.stopPropagation();
        toggleCollapsed();
      }}
    >
      <Tooltip zIndex={100} label={"Collapse sidebar"}>
        {<IconChevronRight stroke={1.625} />}
      </Tooltip>
    </ActionIcon>
  );

  return (
    <Box
      p="xs"
      sx={(theme) => ({
        backgroundColor:
          theme.colorScheme == "dark"
            ? theme.colors.dark[5]
            : theme.colors.gray[1],
        lineHeight: "1.5em",
        fontWeight: 400,
        position: "relative",
        zIndex: 20,
        alignItems: "center",
        display: "flex",
        flexDirection: "row",
      })}
    >
      {children}
      {collapsible ? collapseSidebarToggleButton : null}
    </Box>
  );
};
/** Contents of a panel. */
SidebarPanel.Contents = function SidebarPanelContents({
  children,
}: {
  children: string | React.ReactNode;
}) {
  return children;
};
