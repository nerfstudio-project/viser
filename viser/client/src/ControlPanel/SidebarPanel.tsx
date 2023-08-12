// @refresh reset

import { ActionIcon, Box, Tooltip } from "@mantine/core";
import React from "react";
import { useDisclosure } from "@mantine/hooks";
import { IconChevronLeft, IconChevronRight } from "@tabler/icons-react";

export const SidebarPanelContext = React.createContext<null | {
  collapsible: boolean;
  toggleCollapsed: () => void;
}>(null);

/** Root component for control panel. Parents a set of control tabs.
 * This could be refactored+cleaned up a lot! */
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
        <Tooltip label={"Show sidebar"}>{<IconChevronLeft />}</Tooltip>
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
      <Box
        sx={(theme) => ({
          width: collapsed ? 0 : "20em",
          overflow: "scroll",
          boxSizing: "border-box",
          borderLeft: "1px solid",
          borderColor:
            theme.colorScheme == "light"
              ? theme.colors.gray[4]
              : theme.colors.dark[4],
          transition: "width 0.5s 0s",
          backgroundColor:
            theme.colorScheme == "dark" ? theme.colors.dark[8] : "0xffffff",
        })}
      >
        <Box
          sx={{
            width: "20em", // Prevent DOM reflow.
          }}
        >
          {children}
        </Box>
      </Box>
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
    <Box
      sx={{
        position: "absolute",
        right: "0.5em",
        top: "50%",
        transform: "translateY(-50%)",
        display: collapsible ? undefined : "none",
        zIndex: 100,
      }}
    >
      <ActionIcon
        onClick={(evt) => {
          evt.stopPropagation();
          toggleCollapsed();
        }}
      >
        <Tooltip label={"Collapse sidebar"}>
          {<IconChevronRight stroke={1.625} />}
        </Tooltip>
      </ActionIcon>
    </Box>
  );

  return (
    <Box
      p="sm"
      sx={(theme) => ({
        backgroundColor:
          theme.colorScheme == "dark"
            ? theme.colors.dark[5]
            : theme.colors.gray[1],
        lineHeight: "1.5em",
        fontWeight: 400,
        position: "relative",
        zIndex: 1,
      })}
    >
      {children}
      {collapseSidebarToggleButton}
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
