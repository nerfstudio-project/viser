import { ViewerContext } from ".";
import { ThemeConfigurationMessage } from "./WebsocketMessages";
import { Burger, Button, Container, Group, Header, Paper } from "@mantine/core";
import {
  IconBrandGithub,
  IconFileDescription,
  IconKeyboard,
} from "@tabler/icons-react";
import { useDisclosure } from "@mantine/hooks";
import { useContext } from "react";

// Type helpers.
type ArrayElement<ArrayType extends readonly unknown[]> =
  ArrayType extends readonly (infer ElementType)[] ? ElementType : never;
type NoNull<T> = Exclude<T, null>;
type TitlebarContent = NoNull<ThemeConfigurationMessage["titlebar_content"]>;
function assertUnreachable(x: never): never {
  throw new Error("Didn't expect to get here", x);
}

function getIcon(icon: ArrayElement<NoNull<TitlebarContent["buttons"]>>["icon"]) {
  let Icon = null;
  switch (icon) {
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
      assertUnreachable(icon);
  }
  return Icon;
}


// We inherit props directly from message contents.
export function TitlebarButton(
  props: ArrayElement<NoNull<TitlebarContent["buttons"]>>
) {
  const Icon = getIcon(props.icon);
  return (
    <Button
      component="a"
      variant="default"
      href={props.href || undefined}
      compact
      target="_blank"
      leftIcon={Icon === null ? null : <Icon size="1em" />}
      ml="sm"
      color="gray"
      sx={(theme) => ({
        [theme.fn.smallerThan('sm')]: {
          display: "none",
        }
      })}
    >
      {props.text}
    </Button>
  );
}

export function MobileTitlebarButton(
  props: ArrayElement<NoNull<TitlebarContent["buttons"]>>
) {
  const Icon = getIcon(props.icon);
  return (
  <Button
      m="sm"
      component="a"
      variant="default"
      href={props.href || undefined}
      target="_blank"
      leftIcon={Icon === null ? null : <Icon size="1.5em" />}
      ml="sm"
      color="gray"
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
        height: "1.8em",
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

  const [burgerOpen, burgerHandlers] = useDisclosure(false);

  if (content == null) {
    return null;
  }

  const buttons = content.buttons;
  const imageData = content.image;

  return (
    <Header p="xs" height="3.2em" sx={(theme) => ({
      margin: 0,
      borderBottom: "1px solid",
      borderColor:
        theme.colorScheme == "light"
          ? theme.colors.gray[4]
          : theme.colors.dark[4],
    })}>
      <Container fluid sx={() => ({
        display: "flex",
        alignItems: "center"
      })}>
        <Group sx={() => ({ marginRight: "auto" })}>
          {imageData !== null ? TitlebarImage(imageData) : null}
        </Group>
        <Group sx={() => ({
          flexWrap: 'nowrap',
          overflowX: "scroll",
          msOverflowStyle: "none",
          scrollbarWidth: "none",
          "&::-webkit-scrollbar": {
            display: "none"
          }
        })}>
          {buttons?.map((btn) => TitlebarButton(btn))}
        </Group>
        <Burger size="sm"
          opened={burgerOpen}
          onClick={burgerHandlers.toggle}
          title={!burgerOpen ? "Open navigation" : "Close navigation"}
          sx={(theme) => ({
            [theme.fn.largerThan('sm')]: {
              display: "none",
            }
          })}></Burger>
      </Container>
      <Paper sx={(theme) => ({
        [theme.fn.largerThan('sm')]: {
          display: "none"
        },
        display: "flex",
        flexDirection: "column",
        position: "relative",
        top: 0,
        left: "-0.625rem",
        zIndex: 10000000,
        height: burgerOpen ? "calc(100vh - 2.375em)" : '0',
        width: "100vw",
        transition: "all 0.5s",
        overflow: burgerOpen ? "scroll" : "hidden",
        padding: burgerOpen ? "1rem" : "0"
      })}>
        {buttons?.map((btn) => MobileTitlebarButton(btn))}
      </Paper>
    </Header>
  );
}
