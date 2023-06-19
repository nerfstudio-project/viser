import React from "react";
import { ViewerContext } from "..";
import { GuiConfig } from "./GuiState";
import {
  Accordion,
  Box,
  Button,
  Checkbox,
  ColorInput,
  Flex,
  NumberInput,
  Select,
  Slider,
  Space,
  Stack,
  Text,
  TextInput,
  Tooltip,
} from "@mantine/core";
import { makeThrottledMessageSender } from "../WebsocketInterface";

type Folder = {
  inputs: GuiConfig[];
  subfolders: { [key: string]: Folder };
};

/** Root of generated inputs. */
export default function Generated() {
  const viewer = React.useContext(ViewerContext)!;
  const guiConfigFromId = viewer.useGui((state) => state.guiConfigFromId);

  const guiTree: Folder = { inputs: [], subfolders: {} };

  [...Object.keys(guiConfigFromId)]
    .sort((a, b) => guiConfigFromId[a].order - guiConfigFromId[b].order)
    .forEach((id) => {
      const conf = guiConfigFromId[id];

      // Iterate into subfolder for this GUI element.
      // Could be optimized.
      let folder = guiTree;
      conf.folder_labels.forEach((folder_label) => {
        if (folder.subfolders[folder_label] === undefined)
          folder.subfolders[folder_label] = { inputs: [], subfolders: {} };
        folder = folder.subfolders[folder_label];
      });

      folder.inputs.push(conf);
    });

  return (
    <>
      <Space h="xs" />
      <GeneratedFolder folder={guiTree} />
    </>
  );
}

function GeneratedFolder({ folder }: { folder: Folder }) {
  return (
    <Stack spacing="xs">
      {folder.inputs.map((conf) => (
        <GeneratedInput key={conf.id} conf={conf} />
      ))}
      <Accordion
        chevronPosition="right"
        multiple
        defaultValue={Object.keys(folder.subfolders)}
        styles={(theme) => ({
          label: { padding: "0.625rem 0.2rem" },
          item: { border: 0 },
          control: { paddingLeft: 0 },
          content: {
            borderLeft: "1px solid",
            borderLeftColor: theme.colors.gray[4],
            paddingRight: "0",
            paddingLeft: "0.5rem",
            paddingBottom: 0,
            paddingTop: 0,
            marginBottom: "0.5rem",
            marginLeft: "0.1rem",
          },
        })}
      >
        {Object.keys(folder.subfolders).map((folder_label) => (
          <Accordion.Item key={folder_label} value={folder_label}>
            <Accordion.Control>{folder_label}</Accordion.Control>
            <Accordion.Panel>
              <GeneratedFolder folder={folder.subfolders[folder_label]} />
            </Accordion.Panel>
          </Accordion.Item>
        ))}
      </Accordion>
    </Stack>
  );
}

/** A single generated GUI element. */
function GeneratedInput({ conf }: { conf: GuiConfig }) {
  const viewer = React.useContext(ViewerContext)!;
  const messageSender = makeThrottledMessageSender(viewer.websocketRef, 50);

  function updateValue(value: any) {
    setGuiValue(conf.id, value);
    messageSender({ type: "GuiUpdateMessage", id: conf.id, value: value });
  }

  // TODO: the types here could potentially be made much stronger.
  const setGuiValue = viewer.useGui((state) => state.setGuiValue);
  const value =
    viewer.useGui((state) => state.guiValueFromId[conf.id]) ??
    conf.initial_value;

  let { visible, disabled } =
    viewer.useGui((state) => state.guiAttributeFromId[conf.id]) || {};

  visible = visible ?? true;
  disabled = disabled ?? false;

  if (!visible) return <></>;

  let labeled = true;
  let input = null;
  switch (conf.type) {
    case "GuiAddButtonMessage":
      labeled = false;
      input = (
        <Button
          id={conf.id}
          fullWidth
          onClick={() =>
            messageSender({
              type: "GuiUpdateMessage",
              id: conf.id,
              value: true,
            })
          }
          style={{ height: "1.875rem" }}
          disabled={disabled}
          size="sm"
        >
          {conf.label}
        </Button>
      );
      break;
    case "GuiAddSliderMessage":
      input = (
        <Flex justify="space-between">
          <Box sx={{ flexGrow: 1 }}>
            <Slider
              id={conf.id}
              size="sm"
              pt="0.3rem"
              showLabelOnHover={false}
              min={conf.min}
              max={conf.max}
              step={conf.step ?? undefined}
              precision={conf.precision}
              value={value}
              onChange={updateValue}
              marks={[{ value: conf.min }, { value: conf.max }]}
              disabled={disabled}
            />
            <Flex justify="space-between" sx={{ marginTop: "-0.2em" }}>
              <Text fz="0.7rem" c="dimmed">
                {conf.min}
              </Text>
              <Text fz="0.7rem" c="dimmed">
                {conf.max}
              </Text>
            </Flex>
          </Box>
          <NumberInput
            value={value}
            onChange={updateValue}
            size="xs"
            min={conf.min}
            max={conf.max}
            hideControls
            step={conf.step ?? undefined}
            precision={conf.precision}
            sx={{ width: "3rem", height: "1rem", minHeight: "1rem" }}
            styles={{ input: { padding: "0.3rem" } }}
            ml="xs"
          />
        </Flex>
      );
      break;
    case "GuiAddNumberMessage":
      input = (
        <NumberInput
          id={conf.id}
          value={value ?? conf.initial_value}
          precision={conf.precision}
          min={conf.min ?? undefined}
          max={conf.max ?? undefined}
          step={conf.step}
          size="xs"
          onChange={updateValue}
          disabled={disabled}
        />
      );
      break;
    case "GuiAddTextMessage":
      input = (
        <TextInput
          value={value ?? conf.initial_value}
          size="xs"
          onChange={(value) => {
            updateValue(value.target.value);
          }}
          disabled={disabled}
        />
      );
      break;
    case "GuiAddCheckboxMessage":
      input = (
        <Checkbox
          id={conf.id}
          checked={value ?? conf.initial_value}
          size="xs"
          onChange={(value) => {
            updateValue(value.target.checked);
          }}
          disabled={disabled}
        />
      );
      break;
    case "GuiAddVector2Message":
      input = (
        <VectorInput
          id={conf.id}
          n={2}
          value={value ?? conf.initial_value}
          onChange={updateValue}
          min={conf.min}
          max={conf.max}
          step={conf.step}
          precision={conf.precision}
          disabled={disabled}
        />
      );
      break;
    case "GuiAddVector3Message":
      input = (
        <VectorInput
          id={conf.id}
          n={3}
          value={value ?? conf.initial_value}
          onChange={updateValue}
          min={conf.min}
          max={conf.max}
          step={conf.step}
          precision={conf.precision}
          disabled={disabled}
        />
      );
      break;
    case "GuiAddDropdownMessage":
      input = (
        <Select
          id={conf.id}
          value={value}
          data={conf.options}
          onChange={updateValue}
        />
      );
      break;
    case "GuiAddRgbMessage":
      input = (
        <ColorInput
          disabled={disabled}
          size="xs"
          value={rgbToHex(value)}
          onChange={(v) => updateValue(hexToRgb(v))}
          format="hex"
        />
      );
      break;
    case "GuiAddRgbaMessage":
      input = (
        <ColorInput
          disabled={disabled}
          size="xs"
          value={rgbaToHex(value)}
          onChange={(v) => updateValue(hexToRgba(v))}
          format="hexa"
        />
      );
      break;
    case "GuiAddButtonGroupMessage":
      input = (
        <Flex justify="space-between" columnGap="xs">
          {conf.options.map((option, index) => (
            <Button
              key={index}
              onClick={() =>
                messageSender({
                  type: "GuiUpdateMessage",
                  id: conf.id,
                  value: option,
                })
              }
              style={{ flexGrow: 1, width: 0 }}
              disabled={disabled}
              compact
              size="sm"
              variant="outline"
            >
              {option}
            </Button>
          ))}
        </Flex>
      );
  }

  if (conf.hint !== null)
    input = (
      <Tooltip label={conf.hint} multiline w="15rem" withArrow openDelay={500}>
        {input}
      </Tooltip>
    );

  if (labeled)
    return <LabeledInput id={conf.id} label={conf.label} input={input} />;
  else return input;
}

function VectorInput(
  props:
    | {
        id: string;
        n: 2;
        value: [number, number];
        min: [number, number] | null;
        max: [number, number] | null;
        step: number;
        precision: number;
        onChange: (value: number[]) => void;
        disabled: boolean;
      }
    | {
        id: string;
        n: 3;
        value: [number, number, number];
        min: [number, number, number] | null;
        max: [number, number, number] | null;
        step: number;
        precision: number;
        onChange: (value: number[]) => void;
        disabled: boolean;
      }
) {
  return (
    <Flex justify="space-between" style={{ columnGap: "0.3rem" }}>
      {[...Array(props.n).keys()].map((i) => (
        <NumberInput
          id={i == 0 ? props.id : undefined}
          key={i}
          value={props.value[i]}
          onChange={(v) => {
            const updated = [...props.value];
            updated[i] = v === "" ? 0.0 : v;
            props.onChange(updated);
          }}
          size="xs"
          styles={{
            root: { flexGrow: 1, width: 0 },
            input: {
              paddingLeft: "0.3rem",
              paddingRight: "1.1rem",
              textAlign: "right",
            },
            rightSection: { width: "1.0rem" },
            control: {
              width: "0.875rem",
            },
          }}
          precision={props.precision}
          step={props.step}
          min={props.min === null ? undefined : props.min[i]}
          max={props.max === null ? undefined : props.max[i]}
          disabled={props.disabled}
        />
      ))}
    </Flex>
  );
}

/** GUI input with a label horizontally placed to the left of it. */
function LabeledInput(props: {
  id: string;
  label: string;
  input: React.ReactNode;
}) {
  return (
    <Flex align="center">
      <Box w="6em" pr="xs">
        <Text
          c="dimmed"
          fz="sm"
          lh="1.15em"
          unselectable="off"
          sx={{ wordWrap: "break-word" }}
        >
          <label htmlFor={props.id}> {props.label}</label>
        </Text>
      </Box>
      <Box sx={{ flexGrow: 1 }}>{props.input}</Box>
    </Flex>
  );
}

// Color conversion helpers.

function rgbToHex([r, g, b]: [number, number, number]): string {
  const hexR = r.toString(16).padStart(2, "0");
  const hexG = g.toString(16).padStart(2, "0");
  const hexB = b.toString(16).padStart(2, "0");
  return `#${hexR}${hexG}${hexB}`;
}

function hexToRgb(hexColor: string): [number, number, number] {
  const hex = hexColor.slice(1); // Remove the # in #ffffff.
  const r = parseInt(hex.substring(0, 2), 16);
  const g = parseInt(hex.substring(2, 4), 16);
  const b = parseInt(hex.substring(4, 6), 16);
  return [r, g, b];
}
function rgbaToHex([r, g, b, a]: [number, number, number, number]): string {
  const hexR = r.toString(16).padStart(2, "0");
  const hexG = g.toString(16).padStart(2, "0");
  const hexB = b.toString(16).padStart(2, "0");
  const hexA = a.toString(16).padStart(2, "0");
  return `#${hexR}${hexG}${hexB}${hexA}`;
}

function hexToRgba(hexColor: string): [number, number, number, number] {
  const hex = hexColor.slice(1); // Remove the # in #ffffff.
  const r = parseInt(hex.substring(0, 2), 16);
  const g = parseInt(hex.substring(2, 4), 16);
  const b = parseInt(hex.substring(4, 6), 16);
  const a = parseInt(hex.substring(6, 8), 16);
  return [r, g, b, a];
}
