import { useContext } from "react";
import { ViewerContext } from ".";
import { Message } from "./WebsocketMessages";
import { Button, Grid } from "@mantine/core";
import {
  IconBrandGithub,
  IconFileDescription,
  IconKeyboard,
} from "@tabler/icons-react";

// Type helpers.
type ArrayElement<ArrayType extends readonly unknown[]> =
  ArrayType extends readonly (infer ElementType)[] ? ElementType : never;
type NoNull<T> = Exclude<T, null>;
type TitlebarContent = NoNull<
  (Message & { type: "ThemeConfigurationMessage" })["titlebar_content"]
>;
function assertUnreachable(x: never): never {
  throw new Error("Didn't expect to get here", x);
}

// We inherit props directly from message contents.
export function TitlebarButton(
  props: ArrayElement<NoNull<TitlebarContent["buttons"]>>
) {
  let icon = undefined;
  switch (props.icon) {
    case null:
      break;
    case "GitHub":
      icon = <IconBrandGithub />;
      break;
    case "Description":
      icon = <IconFileDescription />;
      break;
    case "Keyboard":
      icon = <IconKeyboard />;
      break;
    default:
      assertUnreachable(props.icon);
  }
  return (
    <Button
      component="a"
      variant="outline"
      href={props.href || undefined}
      size="sm"
      target="_blank"
      leftIcon={icon}
      mr="xs"
    >
      {props.text}
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
        margin: "0 0.5em",
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
      sx={(theme) => ({
        width: "100%",
        justifyContent: "space-between",
        alignItems: "center",
        borderBottom: "1px solid",
        margin: 0,
        borderColor: theme.colors.gray[4],
      })}
    >
      <Grid.Col
        xs="auto"
        sx={{
          display: "flex",
          alignItems: "center",
          justifyContent: "left",
          overflow: "visible",
        }}
      >
        {buttons?.map((btn) => TitlebarButton(btn))}
      </Grid.Col>
      <Grid.Col
        xs={3}
        sx={{
          display: "flex",
          alignItems: "center",
          justifyContent: "right",
        }}
      >
        {imageData !== null ? TitlebarImage(imageData) : null}
      </Grid.Col>
    </Grid>
  );
}
