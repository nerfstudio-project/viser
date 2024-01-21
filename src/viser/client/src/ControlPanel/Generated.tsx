import {
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

/** Root of generated inputs. */
export default function GeneratedGuiContainer({
  // We need to take viewer as input in drei's <Html /> elements, where contexts break.
  containerId,
  viewer,
  folderDepth,
}: {
  containerId: string;
  viewer?: ViewerContextContents;
  folderDepth?: number;
}) {
  if (viewer === undefined) viewer = React.useContext(ViewerContext)!;

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
        .map((pair, index) => (
          <GeneratedInput
            key={pair.id}
            id={pair.id}
            viewer={viewer}
            folderDepth={folderDepth ?? 0}
            last={index === guiIdOrderPairArray.length - 1}
          />
        ))}
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
    );
  if (conf.type == "GuiAddTabGroupMessage")
    return <GeneratedTabGroup conf={conf} />;

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

  if (!visible) return <></>;

  let inputColor =
    computeRelativeLuminance(theme.fn.primaryColor()) > 50.0
      ? theme.colors.gray[9]
      : theme.white;

  let labeled = true;
  let input = null;
  switch (conf.type) {
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
          radius="xs"
          value={value}
          data={conf.options}
          onChange={updateValue}
          searchable
          maxDropdownHeight={400}
          size="xs"
          styles={{
            input: {
              padding: "0.5em",
              letterSpacing: "-0.5px",
              minHeight: "1.625rem",
              height: "1.625rem",
            },
          }}
          // zIndex of dropdown should be >modal zIndex.
          // On edge cases: it seems like existing dropdowns are always closed when a new modal is opened.
          zIndex={1000}
          withinPortal={true}
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
              size="xs"
              variant="outline"
            >
              {option}
            </Button>
          ))}
        </Flex>
      );
  }
}

function GeneratedFolder({
  conf,
  folderDepth,
  viewer,
}: {
  conf: GuiAddFolderMessage;
  folderDepth: number;
  viewer: ViewerContextContents;
}) {
  const [opened, { toggle }] = useDisclosure(conf.expand_by_default);
  const guiIdSet = viewer.useGui(
    (state) => state.guiIdSetFromContainerId[conf.id],
  );
  const isEmpty = guiIdSet === undefined || Object.keys(guiIdSet).length === 0;

  const ToggleIcon = opened ? IconChevronUp : IconChevronDown;
  return (
    <Paper
      withBorder
      pt="0.0625em"
      mx="xs"
      mt="xs"
      mb="sm"
      sx={{ position: "relative" }}
    >
      <Paper
        sx={{
          fontSize: "0.875em",
          position: "absolute",
          padding: "0 0.375em 0 0.375em",
          top: 0,
          left: "0.375em",
          transform: "translateY(-50%)",
          cursor: isEmpty ? undefined : "pointer",
          userSelect: "none",
          fontWeight: 500,
        }}
        onClick={toggle}
      >
        {conf.label}
        <ToggleIcon
          style={{
            width: "0.9em",
            height: "0.9em",
            strokeWidth: 3,
            top: "0.1em",
            position: "relative",
            marginLeft: "0.25em",
            marginRight: "-0.1em",
            opacity: 0.5,
            display: isEmpty ? "none" : undefined,
          }}
        />
      </Paper>
      <Collapse in={opened && !isEmpty} pt="0.2em">
        <GeneratedGuiContainer
          containerId={conf.id}
          folderDepth={folderDepth + 1}
        />
      </Collapse>
      <Collapse in={!(opened && !isEmpty)}>
        <Box p="xs"></Box>
      </Collapse>
    </Paper>
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
      },
) {
  return (
    <Flex justify="space-between" style={{ columnGap: "0.5em" }}>
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
              paddingLeft: "0.5em",
              paddingRight: "1.75em",
              textAlign: "right",
              minHeight: "1.875em",
              height: "1.875em",
            },
            rightSection: { width: "1.2em" },
            control: {
              width: "1.1em",
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
  folderDepth: number;
}) {
  return (
    <Flex align="center">
      <Box
        // The per-layer offset here is just eyeballed.
        w={`${7.25 - props.folderDepth * 0.6375}em`}
        pr="xs"
        sx={{ flexShrink: 0, position: "relative" }}
      >
        <Text
          c="dimmed"
          fz="0.875em"
          fw="450"
          lh="1.375em"
          lts="-0.75px"
          unselectable="off"
          sx={{
            width: "100%",
            boxSizing: "content-box",
          }}
        >
          <label htmlFor={props.id}>{props.label}</label>
        </Text>
      </Box>
      <Box sx={{ flexGrow: 1 }}>{props.input}</Box>
    </Flex>
  );
}
