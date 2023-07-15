// @refresh reset

import { Box, Paper } from "@mantine/core";
import { IconCaretUp } from "@tabler/icons-react";
import React from "react";
import { isMouseEvent, isTouchEvent, mouseEvents, touchEvents } from "../Utils";

export const FloatingPanelRefContext =
  React.createContext<React.RefObject<HTMLDivElement> | null>(null);

/** Root component for control panel. Parents a set of control tabs.
 * This could be refactored+cleaned up a lot! */
export default function FloatingPanel({
  children,
}: {
  children: string | React.ReactNode;
}) {
  const panelWrapperRef = React.useRef<HTMLDivElement>(null);
  return (
    <FloatingPanelRefContext.Provider value={panelWrapperRef}>
      <Paper
        radius="sm"
        withBorder
        sx={{
          boxSizing: "border-box",
          width: "20em",
          zIndex: 1,
          position: "absolute",
          top: "1em",
          right: "1em",
          margin: 0,
          "&.hidden": {
            overflow: "hidden",
          },
          "& .panel-contents": {
            opacity: "1.0",
            visibility: "visible",
            height: "auto",
            transition: "visibility 0.2s linear,opacity 0.2s linear",
          },
          "&.hidden .panel-contents": {
            opacity: "0.0",
            visibility: "hidden",
            height: "0 !important",
            border: "0",
            position: "absolute",
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
        {children}
      </Paper>
    </FloatingPanelRefContext.Provider>
  );
}

/** Handle object helps us hide, show, and drag our panel.*/
FloatingPanel.Handle = function FloatingPanelHandle({
  children,
}: {
  children: string | React.ReactNode;
}) {
  const panelWrapperRef = React.useContext(FloatingPanelRefContext)!;
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
    parentSize: number
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
      parent.clientWidth - panel.clientWidth - panelBoundaryPad
    );
    newX = Math.max(newX, panelBoundaryPad);
    newY = Math.min(
      newY,
      parent.clientHeight - panel.clientHeight - panelBoundaryPad
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

    panel.style.maxHeight = `${(
      parent.clientHeight -
      panelBoundaryPad * 2
    ).toString()}px`;

    const observer = new ResizeObserver(() => {
      if (unfixedOffset.current.x === undefined)
        unfixedOffset.current.x = computePanelOffset(
          panel.offsetLeft,
          panel.clientWidth,
          parent.clientWidth
        );
      if (unfixedOffset.current.y === undefined)
        unfixedOffset.current.y = computePanelOffset(
          panel.offsetTop,
          panel.clientHeight,
          parent.clientHeight
        );

      panel.style.maxHeight = `${(
        parent.clientHeight -
        panelBoundaryPad * 2
      ).toString()}px`;

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
      | React.MouseEvent<HTMLDivElement, MouseEvent>
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
        newY
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
      { once: true }
    );
  };

  return (
    <Box
      sx={(theme) => ({
        backgroundColor:
          theme.colorScheme === "dark"
            ? theme.colors.dark[5]
            : theme.colors.gray[1],
        lineHeight: "1.5em",
        cursor: "pointer",
        position: "relative",
        fontWeight: 400,
        boxSizing: "border-box",
        overflow: "hidden",
        userSelect: "none",
      })}
      onClick={() => {
        const state = dragInfo.current;
        if (state.dragging) {
          state.dragging = false;
          return;
        }

        const wrapper = panelWrapperRef.current;
        if (!wrapper) return;
        wrapper.classList.toggle("hidden");
      }}
      onTouchStart={(event) => {
        dragHandler(event);
      }}
      onMouseDown={(event) => {
        dragHandler(event);
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
      <Box
        component="div"
        sx={{
          position: "absolute",
          top: "50%",
          right: "0.5em",
          transform: "translateY(-48%) scale(0.8)",
          height: "1.5em",
          opacity: "0.5",
        }}
      >
        <IconCaretUp className="expand-icon" />
      </Box>
    </Box>
  );
};
/** Contents of a panel. */
FloatingPanel.Contents = function FloatingPanelContents({
  children,
}: {
  children: string | React.ReactNode;
}) {
  return <Box className="panel-contents">{children}</Box>;
};
