import { useContext } from "react";
import { ViewerContext } from ".";
import { Message } from "./WebsocketMessages";
import { Box, Button, MantineTheme, useMantineTheme } from "@mantine/core";
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
  let Icon = null;
  switch (props.icon) {
    case null:
      break;
    case "GitHub":
      Icon = IconBrandGithub;
      break;
    case "Description":
      Icon = IconFileDescription;
      break;
    case "Keyboard":
      Icon = IconKeyboard;
      break;
    default:
      assertUnreachable(props.icon);
  }
  return (
    <Button
      component="a"
      variant="outline"
      href={props.href || undefined}
      compact
      target="_blank"
      leftIcon={Icon === null ? null : <Icon size="1em" />}
      ml="sm"
      color="gray"
    >
      {props.text}
    </Button>
  );
}

export function TitlebarImage(props: NoNull<TitlebarContent["image"]>, theme: MantineTheme) {
  let imageSource: string;
  if (props.dark_image_url == null || theme.colorScheme == "light") {
    imageSource = props.image_url
  } else {
    imageSource = props.dark_image_url
  }
  const image = (
    <img
      src={imageSource}
      alt={props.image_alt}
      style={{
        height: "1.8em",
        margin: "0 0.5em",
      }}
    />
  )

  if (props.href == null) {
    return image;
  }
  return <a href={props.href}>{image}</a>;
}

export function Titlebar() {
  const viewer = useContext(ViewerContext)!;
  const content = viewer.useGui((state) => state.theme.titlebar_content);
  const theme = useMantineTheme();

  if (content == null) {
    return null;
  }

  const buttons = content.buttons;
  const imageData = content.image;

  return (
    <Box
      p="xs"
      sx={(theme) => ({
        width: "100%",
        margin: 0,
        display: "flex",
        alignItems: "center",
        borderBottom: "1px solid",
        borderColor:
          theme.colorScheme == "light"
            ? theme.colors.gray[4]
            : theme.colors.dark[4],
      })}
    >
      {imageData !== null ? TitlebarImage(imageData, theme) : null}
      {buttons?.map((btn) => TitlebarButton(btn))}
    </Box>
  );
}
