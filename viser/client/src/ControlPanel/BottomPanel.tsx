import { Box, Paper } from "@mantine/core";
import { IconCaretDown } from "@tabler/icons-react";
import React from "react";
import { isMouseEvent, isTouchEvent, mouseEvents, touchEvents } from "../Utils";

const BottomPanelRefContext =
  React.createContext<React.RefObject<HTMLDivElement> | null>(null);

const pagePercent = (start: number, end: number) => {
  return (start - end) / window.innerHeight;
};
const getHeight = (element: HTMLElement) => {
  let size = 0;
  const children = element.children;
  for (let i = 0; i < children.length; i++) {
    size += children[i].clientHeight;
  }
  return size;
};

export default function BottomPanel({
  children,
}: {
  children: string | React.ReactNode;
}) {
  const panelWrapperRef = React.useRef<HTMLDivElement>(null);
  return (
    <BottomPanelRefContext.Provider value={panelWrapperRef}>
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
          height: "3.5em",
          transition: "height 0.3s linear",
        }}
        ref={panelWrapperRef}
        className="hidden"
      >
        {children}
      </Paper>
    </BottomPanelRefContext.Provider>
  );
}
BottomPanel.Handle = function FloatingPanelHandle({
  children,
}: {
  children: string | React.ReactNode;
}) {
  const panelWrapperRef = React.useContext(BottomPanelRefContext)!;
  // Things to track for dragging.
  const dragInfo = React.useRef({
    dragging: false,
    startHeight: 0,
    hidden: true,
  });

  const dragHandler = (
    event:
      | React.TouchEvent<HTMLDivElement>
      | React.MouseEvent<HTMLDivElement, MouseEvent>
  ) => {
    const state = dragInfo.current;
    const panel = panelWrapperRef.current;
    if (panel === null) return;
    state.startHeight = panel.clientHeight;
    panel.style.transition = "none";
    let hidePanel = state.hidden;
    const eventNames = event.type == "touchstart" ? touchEvents : mouseEvents;
    function dragListener(event: TouchEvent | MouseEvent) {
      let pos = 0;
      if (isTouchEvent(event)) {
        pos = window.innerHeight - event.touches[0].clientY;
      } else if (isMouseEvent(event)) {
        pos = window.innerHeight - event.clientY;
      }

      state.dragging = true;
      if (!panel) {
        return;
      }
      panel.style.height = pos + "px";
      const change = pagePercent(state.startHeight, panel.clientHeight);
      if ((!state.hidden && change > 0.1) || (state.hidden && change > -0.1)) {
        panel.classList.add("hidden");
        hidePanel = true;
      } else if (
        (!state.hidden && change <= 0.05) ||
        (state.hidden && change <= -0.1)
      ) {
        panel.classList.remove("hidden");
        hidePanel = false;
      }
    }
    window.addEventListener(eventNames.move, dragListener);
    window.addEventListener(
      eventNames.end,
      () => {
        state.dragging = false;
        state.hidden = hidePanel;
        window.removeEventListener(eventNames.move, dragListener);
        panel.style.transition = "height 0.3s linear";
        if (state.hidden) {
          panel.style.height = "3.5em";
        } else {
          panel.style.height = getHeight(panel) + "px";
        }
      },
      { once: true }
    );
  };

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
        const state = dragInfo.current;
        const panel = panelWrapperRef.current;
        if (panel === null) return;

        if (state.dragging) {
          state.dragging = false;
          return;
        }

        panel.classList.toggle("hidden");
        if (state.hidden) {
          panel.style.height = getHeight(panel) + "px";
          state.hidden = false;
        } else {
          panel.style.height = "3.5em";
          state.hidden = true;
        }
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
    </Box>
  );
};
/** Contents of a panel. */
BottomPanel.Contents = function FloatingPanelContents({
  children,
}: {
  children: string | React.ReactNode;
}) {
  const panelWrapperRef = React.useContext(BottomPanelRefContext)!;
  const contentRef = React.useRef<HTMLDivElement>(null);
  React.useEffect(() => {
    const panel = panelWrapperRef.current;
    const content = contentRef.current;
    if (panel === null) return;
    if (content === null) return;

    const observer = new ResizeObserver(() => {
      if (!panel.classList.contains("hidden")) {
        panel.style.height = getHeight(panel) + "px";
      }
    });
    observer.observe(content);
    return () => {
      observer.disconnect();
    };
  });
  return (
    <Box className="panel-contents" ref={contentRef}>
      {children}
    </Box>
  );
};
