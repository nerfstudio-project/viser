import {
  AllComponentProps,
  GuiAddFolderMessage,
  GuiAddTabGroupMessage,
} from "../WebsocketMessages";
import { ViewerContext, ViewerContextContents } from "../App";
import { makeThrottledMessageSender } from "../WebsocketFunctions";
import { computeRelativeLuminance } from "./GuiState";
import {
  Collapse,
  Image,
  Paper,
  Tabs,
  TabsValue,
  useMantineTheme,
} from "@mantine/core";

import {
  Box,
  Button,
  Checkbox,
  ColorInput,
  Flex,
  NumberInput,
  Select,
  Slider,
  Text,
  TextInput,
  Tooltip,
} from "@mantine/core";
import React from "react";
import Markdown from "../Markdown";
import { ErrorBoundary } from "react-error-boundary";
import { useDisclosure } from "@mantine/hooks";
import { IconChevronDown, IconChevronUp } from "@tabler/icons-react";
import { GuiGenerateContext, SetProps } from "./GuiState";
import GeneratedComponent from "../GeneratedComponent";


function GeneratedComponentFromId<T extends AllComponentProps>({ id }: { id: string }) {
  const viewer = React.useContext(ViewerContext)!;
  const props = viewer.useGui((state) => state.guiPropsFromId[id]) ?? {};
  const attributes = viewer.useGui((state) => state.guiAttributeFromId[id]) ?? {};
  const contextValue = React.useContext(GuiGenerateContext)!;
  const update = (callback: SetProps<AllComponentProps>) => viewer.useGui((state) => state.setProps<T>(id, callback));
  return <GuiGenerateContext.Provider value={{ ...contextValue, id, update }}>
    <GeneratedComponent {...props} />
  </GuiGenerateContext.Provider>;
}

/** Root of generated inputs. */
export default function GeneratedGuiContainer({
  // We need to take viewer as input in drei's <Html /> elements, where contexts break.
  containerId,
  folderDepth,
}: {
  containerId: string;
  folderDepth?: number;
}) {
  const viewer = React.useContext(ViewerContext)!;

  const guiIdSet =
    viewer.useGui((state) => state.guiIdSetFromContainerId[containerId]) ?? {};

  // Render each GUI element in this container.
  const guiIdArray = [...Object.keys(guiIdSet)];
  const guiOrderFromId = viewer!.useGui((state) => state.guiOrderFromId);
  if (guiIdSet === undefined) return null;

  const guiIdOrderPairArray = guiIdArray.map((id) => ({
    id: id,
    order: guiOrderFromId[id],
  }));
  const out = (
    <Box pt="0.75em">
      {guiIdOrderPairArray
        .sort((a, b) => a.order - b.order)
        .map((pair, index) => <GeneratedInput
              id={pair.id}
              viewer={viewer}
              folderDepth={folderDepth ?? 0}
              last={index === guiIdOrderPairArray.length - 1}
            />
        )}
    </Box>
  );
  return out;
}

/** A single generated GUI element. */
function GeneratedInput({
  id,
  viewer,
  folderDepth,
  last,
}: {
  id: string;
  viewer?: ViewerContextContents;
  folderDepth: number;
  last: boolean;
}) {
  // Handle GUI input types.
  if (viewer === undefined) viewer = React.useContext(ViewerContext)!;
  const conf = viewer.useGui((state) => state.guiConfigFromId[id]);

  // Handle nested containers.
  if (conf.type == "GuiAddFolderMessage")
    return (
  <></>
    );
  if (conf.type == "GuiAddTabGroupMessage")
    return <></>;

  const messageSender = makeThrottledMessageSender(viewer.websocketRef, 50);
  function updateValue(value: any) {
    setGuiValue(conf.id, value);
    messageSender({ type: "GuiUpdateMessage", id: conf.id, value: value });
  }

  const setGuiValue = viewer.useGui((state) => state.setGuiValue);
  const value =
    viewer.useGui((state) => state.guiValueFromId[conf.id]) ??
    conf.initial_value;
  const theme = useMantineTheme();

  let { visible, disabled } =
    viewer.useGui((state) => state.guiAttributeFromId[conf.id]) || {};

  visible = visible ?? true;
  disabled = disabled ?? false;

}
