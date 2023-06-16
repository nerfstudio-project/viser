import { Button, Grid, IconButton, SvgIcon } from "@mui/material";
import { useContext } from "react";
import { ViewerContext } from ".";
import { Message } from "./WebsocketMessages";

import * as Icons from "@mui/icons-material";

// Type helpers.
type IconName = keyof typeof Icons;
type ArrayElement<ArrayType extends readonly unknown[]> =
  ArrayType extends readonly (infer ElementType)[] ? ElementType : never;
type NoNull<T> = Exclude<T, null>;
type TitlebarContent = NoNull<
  (Message & { type: "ThemeConfigurationMessage" })["titlebar_content"]
>;

// We inherit props directly from message contents.
export function TitlebarButton(
  props: ArrayElement<NoNull<TitlebarContent["buttons"]>>
) {
  if (props.icon !== null && props.text === null) {
    return (
      <IconButton href={props.href ?? ""}>
        <SvgIcon component={Icons[props.icon as IconName] ?? null} />
      </IconButton>
    );
  }
  return (
    <Button
      variant={props.variant ?? "contained"}
      href={props.href ?? ""}
      sx={{
        marginY: "0.4em",
        marginX: "0.125em",
        alignItems: "center",
      }}
      size="small"
      target="_blank"
      startIcon={
        props.icon ? (
          <SvgIcon component={Icons[props.icon as IconName] ?? null} />
        ) : null
      }
    >
      {props.text ?? ""}
    </Button>
  );
}

export function TitlebarImage(props: NoNull<TitlebarContent["image"]>) {
  const image = (
    <img
      src={props.image_url}
      alt={props.image_alt}
      style={{
        height: "2em",
        marginLeft: "0.125em",
        marginRight: "0.125em",
      }}
    />
  );
  if (props.href == null) {
    return image;
  }
  return <a href={props.href}>{image}</a>;
}

export function Titlebar() {
  const viewer = useContext(ViewerContext)!;
  const content = viewer.useGui((state) => state.theme.titlebar_content);

  if (content == null) {
    return null;
  }

  const buttons = content.buttons;
  const imageData = content.image;

  return (
    <Grid
      container
      sx={{
        width: "100%",
        zIndex: "1000",
        backgroundColor: "rgba(255, 255, 255, 0.85)",
        borderBottom: "1px solid",
        borderBottomColor: "divider",
        direction: "row",
        justifyContent: "space-between",
        alignItems: "center",
        paddingX: "0.875em",
        height: "2.5em",
      }}
    >
      <Grid
        item
        xs="auto"
        component="div"
        sx={{
          display: "flex",
          direction: "row",
          alignItems: "center",
          justifyContent: "left",
          overflow: "visible",
        }}
      >
        {buttons?.map((btn) => TitlebarButton(btn))}
      </Grid>
      <Grid
        item
        xs={3}
        component="div"
        sx={{
          display: "flex",
          direction: "row",
          alignItems: "center",
          justifyContent: "right",
        }}
      >
        {imageData !== null ? TitlebarImage(imageData) : null}
      </Grid>
    </Grid>
  );
}
