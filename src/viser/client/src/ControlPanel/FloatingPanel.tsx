// @refresh reset

import {
  Box,
  Collapse,
  Paper,
  ScrollArea,
  useMantineColorScheme,
  useMantineTheme,
} from "@mantine/core";
import React from "react";
import { useDisclosure } from "@mantine/hooks";

// Drag Utils
interface DragEvents {
  move: "touchmove" | "mousemove";
  end: "touchend" | "mouseup";
}
const touchEvents: DragEvents = { move: "touchmove", end: "touchend" };
const mouseEvents: DragEvents = { move: "mousemove", end: "mouseup" };

function isTouchEvent(event: TouchEvent | MouseEvent): event is TouchEvent {
  return event.type === "touchmove";
}
function isMouseEvent(event: TouchEvent | MouseEvent): event is MouseEvent {
  return event.type === "mousemove";
}

const FloatingPanelContext = React.createContext<null | {
  wrapperRef: React.RefObject<HTMLDivElement>;
  expanded: boolean;
  width: string;
  maxHeight: number;
  toggleExpanded: () => void;
  dragHandler: (
    event:
      | React.TouchEvent<HTMLDivElement>
      | React.MouseEvent<HTMLDivElement, MouseEvent>,
  ) => void;
  dragInfo: React.MutableRefObject<{
    dragging: boolean;
    startPosX: number;
    startPosY: number;
    startClientX: number;
    startClientY: number;
  }>;
}>(null);

/** A floating panel for displaying controls. */
export default function FloatingPanel({
  children,
  width,
}: {
  children: string | React.ReactNode;
  width: string;
}) {
  const panelWrapperRef = React.useRef<HTMLDivElement>(null);
  const [expanded, { toggle: toggleExpanded }] = useDisclosure(true);
  const [maxHeight, setMaxHeight] = React.useState(800);

  // Things to track for dragging.
  const dragInfo = React.useRef({
    dragging: false,
    startPosX: 0,
    startPosY: 0,
    startClientX: 0,
    startClientY: 0,
  });

  // Logic for "fixing" panel locations, which keeps the control panel within
  // the bounds of the parent div.
  //
  // For `unfixedOffset`, we use a negative sign to indicate that the panel is
  // positioned relative to the right/bottom bound of the parent.
  const unfixedOffset = React.useRef<{ x?: number; y?: number }>({});
  const computePanelOffset = (
    panelPosition: number,
    panelSize: number,
    parentSize: number,
  ) =>
    Math.abs(panelPosition + panelSize / 2.0) <
    Math.abs(panelPosition - parentSize + panelSize / 2.0)
      ? panelPosition
      : panelPosition - parentSize;

  const panelBoundaryPad = 15;
  function setPanelLocation(x: number, y: number) {
    const panel = panelWrapperRef.current;
    if (panel === null) return [x, y];

    const parent = panel.parentElement;
    if (parent === null) return [x, y];

    let newX = x;
    let newY = y;

    newX = Math.min(
      newX,
      parent.clientWidth - panel.clientWidth - panelBoundaryPad,
    );
    newX = Math.max(newX, panelBoundaryPad);
    newY = Math.min(
      newY,
      parent.clientHeight - panel.clientHeight - panelBoundaryPad,
    );
    newY = Math.max(newY, panelBoundaryPad);

    panel.style.top = `${newY.toString()}px`;
    panel.style.left = `${newX.toString()}px`;

    return [
      computePanelOffset(newX, panel.clientWidth, parent.clientWidth),
      computePanelOffset(newY, panel.clientHeight, parent.clientHeight),
    ];
  }

  // Fix locations on resize.
  React.useEffect(() => {
    const panel = panelWrapperRef.current;
    if (panel === null) return;

    const parent = panel.parentElement;
    if (parent === null) return;

    const observer = new ResizeObserver(() => {
      if (unfixedOffset.current.x === undefined)
        unfixedOffset.current.x = computePanelOffset(
          panel.offsetLeft,
          panel.clientWidth,
          parent.clientWidth,
        );
      if (unfixedOffset.current.y === undefined)
        unfixedOffset.current.y = computePanelOffset(
          panel.offsetTop,
          panel.clientHeight,
          parent.clientHeight,
        );

      const newMaxHeight =
        parent.clientHeight - panelBoundaryPad * 2 - 2.5 * 16;
      maxHeight !== newMaxHeight && setMaxHeight(newMaxHeight);

      let newX = unfixedOffset.current.x;
      let newY = unfixedOffset.current.y;
      while (newX < 0) newX += parent.clientWidth;
      while (newY < 0) newY += parent.clientHeight;
      setPanelLocation(newX, newY);
    });
    observer.observe(panel);
    observer.observe(parent);
    return () => {
      observer.disconnect();
    };
  });

  const dragHandler = (
    event:
      | React.TouchEvent<HTMLDivElement>
      | React.MouseEvent<HTMLDivElement, MouseEvent>,
  ) => {
    const state = dragInfo.current;
    const panel = panelWrapperRef.current;
    if (!panel) return;
    if (event.type == "touchstart") {
      event = event as React.TouchEvent<HTMLDivElement>;
      state.startClientX = event.touches[0].clientX;
      state.startClientY = event.touches[0].clientY;
    } else {
      event = event as React.MouseEvent<HTMLDivElement, MouseEvent>;
      state.startClientX = event.clientX;
      state.startClientY = event.clientY;
    }
    state.startPosX = panel.offsetLeft;
    state.startPosY = panel.offsetTop;
    const eventNames = event.type == "touchstart" ? touchEvents : mouseEvents;
    function dragListener(event: MouseEvent | TouchEvent) {
      // Minimum motion.
      let deltaX = 0;
      let deltaY = 0;
      if (isTouchEvent(event)) {
        event = event as TouchEvent;
        deltaX = event.touches[0].clientX - state.startClientX;
        deltaY = event.touches[0].clientY - state.startClientY;
      } else if (isMouseEvent(event)) {
        event = event as MouseEvent;
        deltaX = event.clientX - state.startClientX;
        deltaY = event.clientY - state.startClientY;
      }
      if (Math.abs(deltaX) <= 3 && Math.abs(deltaY) <= 3) return;

      state.dragging = true;
      const newX = state.startPosX + deltaX;
      const newY = state.startPosY + deltaY;
      [unfixedOffset.current.x, unfixedOffset.current.y] = setPanelLocation(
        newX,
        newY,
      );
    }
    window.addEventListener(eventNames.move, dragListener);
    window.addEventListener(
      eventNames.end,
      () => {
        if (event.type == "touchstart") {
          state.dragging = false;
        }
        window.removeEventListener(eventNames.move, dragListener);
      },
      { once: true },
    );
  };

  return (
    <FloatingPanelContext.Provider
      value={{
        wrapperRef: panelWrapperRef,
        expanded: expanded,
        width: width,
        maxHeight: maxHeight,
        toggleExpanded: toggleExpanded,
        dragHandler: dragHandler,
        dragInfo: dragInfo,
      }}
    >
      <Paper
        radius="xs"
        shadow="0.1em 0 1em 0 rgba(0,0,0,0.1)"
        style={{
          boxSizing: "border-box",
          width: width,
          zIndex: 10,
          position: "absolute",
          top: "1em",
          right: "1em",
          margin: 0,
          "& .expandIcon": {
            transform: "rotate(0)",
          },
          overflow: "hidden",
        }}
        ref={panelWrapperRef}
      >
        {children}
      </Paper>
    </FloatingPanelContext.Provider>
  );
}

/** Handle object helps us hide, show, and drag our panel.*/
FloatingPanel.Handle = function FloatingPanelHandle({
  children,
}: {
  children: string | React.ReactNode;
}) {
  const panelContext = React.useContext(FloatingPanelContext)!;
  const theme = useMantineTheme();
  const isDarkMode = useMantineColorScheme().colorScheme === "dark";

  return (
    <>
      <Box
        style={{
          borderRadius: "0.2em 0.2em 0 0",
          lineHeight: "1.5em",
          cursor: "pointer",
          position: "relative",
          fontWeight: 400,
          userSelect: "none",
          display: "flex",
          alignItems: "center",
          padding: "0 0.75em",
          height: "2.75em",
        }}
        onClick={() => {
          const state = panelContext.dragInfo.current;
          if (state.dragging) {
            state.dragging = false;
            return;
          }
          panelContext.toggleExpanded();
        }}
        onTouchStart={(event) => {
          panelContext.dragHandler(event);
        }}
        onMouseDown={(event) => {
          panelContext.dragHandler(event);
        }}
      >
        {children}
      </Box>
      <Box
        mx="xs"
        style={{
          borderBottomWidth: panelContext.expanded ? "1px" : 0,
          borderBottomStyle: "solid",
          borderColor: isDarkMode ? theme.colors.dark[4] : theme.colors.gray[3],
        }}
      ></Box>
    </>
  );
};
/** Contents of a panel. */
FloatingPanel.Contents = function FloatingPanelContents({
  children,
}: {
  children: string | React.ReactNode;
}) {
  const context = React.useContext(FloatingPanelContext)!;
  return (
    <Collapse in={context.expanded}>
      <ScrollArea.Autosize mah={context.maxHeight}>
        <Box
          /* Prevent internals from getting too wide. Needs to match the
           * width of the wrapper element above. */
          style={{ width: context.width }}
        >
          {children}
        </Box>
      </ScrollArea.Autosize>
    </Collapse>
  );
};

/** Hides contents when floating panel is collapsed. */
FloatingPanel.HideWhenCollapsed = function FloatingPanelHideWhenCollapsed({
  children,
}: {
  children: React.ReactNode;
}) {
  const expanded = React.useContext(FloatingPanelContext)?.expanded ?? true;
  return expanded ? children : null;
};
