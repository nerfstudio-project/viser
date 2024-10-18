import { globalStyle } from "@vanilla-extract/css";

globalStyle(".mantine-ScrollArea-scrollbar", {
  zIndex: 100,
});

globalStyle("html", {
  fontSize: "90%",
  "@media": {
    "(max-width: 767px)": {
      fontSize: "80%",
    },
  },
});
