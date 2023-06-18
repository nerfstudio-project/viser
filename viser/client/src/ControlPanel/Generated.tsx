import React from "react";
import { ViewerContext } from "..";
import { GuiConfig } from "./GuiState";
import {
  Box,
  Button,
  Flex,
  NumberInput,
  Slider,
  Stack,
  Text,
} from "@mantine/core";
import { makeThrottledMessageSender } from "../WebsocketInterface";

type Folder = {
  inputs: GuiConfig[];
  subfolders: { [key: string]: Folder };
};

function isFolder(x: any): x is Folder {
  return x && x["is_folder"];
}

export default function Generated() {
  const viewer = React.useContext(ViewerContext)!;
  const guiConfigFromId = viewer.useGui((state) => state.guiConfigFromId);

  const guiTree: Folder = { inputs: [], subfolders: {} };

  Object.keys(guiConfigFromId).forEach((id) => {
    const conf = guiConfigFromId[id];
    // TODO logic for populating folder structure.
    guiTree.inputs.push(conf);
  });

  // TODO: support folders.
  return (
    <Stack spacing="xs">
      {guiTree.inputs.map((conf) => (
        <GeneratedInput key={conf.id} conf={conf} />
      ))}
    </Stack>
  );
}

function GeneratedInput({ conf }: { conf: GuiConfig }) {
  const viewer = React.useContext(ViewerContext)!;
  const messageSender = makeThrottledMessageSender(viewer.websocketRef, 50);

  function sendUpdate(value: any) {
    messageSender({ type: "GuiUpdateMessage", id: conf.id, value: value });
  }

  const setGuiValue = viewer.useGui((state) => state.setGuiValue);
  const value = viewer.useGui((state) => state.guiValueFromId[conf.id]);

  let labeled = true;
  let input = null;
  switch (conf.type) {
    case "GuiAddButtonMessage":
      labeled = false;
      input = (
        <Button
          my="xs"
          fullWidth
          onClick={() =>
            messageSender({
              type: "GuiUpdateMessage",
              id: conf.id,
              value: true,
            })
          }
        >
          {conf.label}
        </Button>
      );
      break;
    case "GuiAddSliderMessage":
      input = (
        <>
          <Slider
            size="sm"
            value={value ?? conf.initial_value}
            min={conf.min}
            max={conf.max}
            step={conf.step ?? undefined}
            onChange={(value) => {
              setGuiValue(conf.id, value);
              sendUpdate(value);
            }}
            marks={[{ value: conf.min }, { value: conf.max }]}
            sx={{ flexGrow: 1 }}
          />
          <Flex justify="space-between">
            <Text fz="xs" c="dimmed">
              {conf.min}
            </Text>
            <Text fz="xs" c="dimmed">
              {conf.max}
            </Text>
          </Flex>
        </>
      );
      break;
    case "GuiAddNumberMessage":
      input = (
        <NumberInput
          value={value ?? conf.initial_value}
          precision={conf.precision}
          min={conf.min ?? undefined}
          max={conf.max ?? undefined}
          step={conf.step}
          size="xs"
          onChange={(value) => {
            setGuiValue(conf.id, value as number);
            sendUpdate(value);
          }}
        />
      );
  }

  if (labeled) return <LabeledInput label={conf.label} input={input} />;
  else return input;
}

/** GUI input with a label horizontally placed to the left of it. */
function LabeledInput(props: { label: string; input: React.ReactNode }) {
  return (
    <Flex align="top">
      <Box sx={{ width: "6em" }}>
        <Text c="dimmed" fz="sm">
          {props.label}
        </Text>
      </Box>
      <Box sx={{ flexGrow: 1, height: "2em" }}>{props.input}</Box>
    </Flex>
  );
}
