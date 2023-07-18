import {
  GuiAddFolderMessage,
  GuiAddTabGroupMessage,
} from "../WebsocketMessages";
import { ViewerContext } from "../App";
import { makeThrottledMessageSender } from "../WebsocketFunctions";
import { GuiConfig } from "./GuiState";
import { Image, Tabs, TabsValue } from "@mantine/core";

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
  Text,
  TextInput,
  Tooltip,
} from "@mantine/core";
import React from "react";

/** Root of generated inputs. */
export default function GeneratedGuiContainer({
  containerId,
}: {
  containerId: string;
}) {
  const viewer = React.useContext(ViewerContext)!;
  const guiIdSet = viewer.useGui(
    (state) => state.guiIdSetFromContainerId[containerId]
  );
  const guiConfigFromId = viewer.useGui((state) => state.guiConfigFromId);

  // Render each GUI element in this container.
  const out =
    guiIdSet === undefined ? null : (
      <>
        {[...Object.keys(guiIdSet)]
          .map((id) => guiConfigFromId[id])
          .sort((a, b) => a.order - b.order)
          .map((conf, index) => {
            return (
              <GeneratedInput conf={conf} key={conf.id} first={index == 0} />
            );
          })}
      </>
    );

  return out;
}

/** A single generated GUI element. */
function GeneratedInput({ conf, first }: { conf: GuiConfig; first: boolean }) {
  // Handle nested containers.
  if (conf.type == "GuiAddFolderMessage")
    return <GeneratedFolder conf={conf} first={first} />;
  if (conf.type == "GuiAddTabGroupMessage")
    return <GeneratedTabGroup conf={conf} />;

  // Handle GUI input types.
  const viewer = React.useContext(ViewerContext)!;
  const messageSender = makeThrottledMessageSender(viewer.websocketRef, 50);
  function updateValue(value: any) {
    setGuiValue(conf.id, value);
    messageSender({ type: "GuiUpdateMessage", id: conf.id, value: value });
  }

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
          stepHoldDelay={500}
          stepHoldInterval={(t) => Math.max(1000 / t ** 2, 25)}
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
          searchable
          maxDropdownHeight={400}
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
    input = // We need to add <Box /> for inputs that we can't assign refs to.
      (
        <Tooltip
          label={conf.hint}
          multiline
          w="15rem"
          withArrow
          openDelay={500}
        >
          <Box>{input}</Box>
        </Tooltip>
      );

  if (labeled)
    input = <LabeledInput id={conf.id} label={conf.label} input={input} />;

  return (
    <Box pt={first ? "sm" : undefined} pb="xs" px="sm">
      {input}
    </Box>
  );
}

function GeneratedFolder({
  conf,
  first,
}: {
  conf: GuiAddFolderMessage;
  first: boolean;
}) {
  return (
    <Accordion
      chevronPosition="right"
      multiple
      pt={first ? "sm" : undefined}
      pb="xs"
      px="sm"
      defaultValue={[conf.label]}
      styles={(theme) => ({
        label: { padding: "0.5rem 0.4rem" },
        item: { border: 0 },
        control: { paddingLeft: 0 },
        content: {
          borderLeft: "1px solid",
          borderLeftColor:
            theme.colorScheme === "light"
              ? theme.colors.gray[3]
              : theme.colors.dark[5],
          padding: 0,
          marginBottom: 0,
          marginLeft: "0.05rem",
        },
      })}
    >
      <Accordion.Item value={conf.label}>
        <Accordion.Control>{conf.label}</Accordion.Control>
        <Accordion.Panel>
          <GeneratedGuiContainer containerId={conf.id} />
        </Accordion.Panel>
      </Accordion.Item>
    </Accordion>
  );
}

function GeneratedTabGroup({ conf }: { conf: GuiAddTabGroupMessage }) {
  const [tabState, setTabState] = React.useState<TabsValue>("0");
  const icons = conf.tab_icons_base64;

  return (
    <Tabs radius="xs" value={tabState} onTabChange={setTabState}>
      <Tabs.List>
        {conf.tab_labels.map((label, index) => (
          <Tabs.Tab
            value={index.toString()}
            key={index}
            icon={
              icons[index] === null ? undefined : (
                <Image
                  height="1.0rem"
                  sx={(theme) => ({
                    filter:
                      theme.colorScheme == "dark" ? "invert(1)" : undefined,
                  })}
                  src={"data:image/svg+xml;base64," + icons[index]}
                />
              )
            }
          >
            {label}
          </Tabs.Tab>
        ))}
      </Tabs.List>
      {conf.tab_container_ids.map((containerId, index) => (
        <Tabs.Panel value={index.toString()} key={containerId}>
          <GeneratedGuiContainer containerId={containerId} />
        </Tabs.Panel>
      ))}
    </Tabs>
  );
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
          id={i === 0 ? props.id : undefined}
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
          stepHoldDelay={500}
          stepHoldInterval={(t) => Math.max(1000 / t ** 2, 25)}
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
