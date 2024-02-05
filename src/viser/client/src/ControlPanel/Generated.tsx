import {
  GuiAddFolderMessage,
  GuiAddTabGroupMessage,
} from "../WebsocketMessages";
import { ViewerContext, ViewerContextContents } from "../App";
import { makeThrottledMessageSender } from "../WebsocketFunctions";
import { computeRelativeLuminance } from "./GuiState";
import { v4 as uuid } from "uuid";
import {
  Collapse,
  Image,
  Paper,
  Progress,
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
import { MultiSlider } from "./MultiSlider";
import React from "react";
import Markdown from "../Markdown";
import { ErrorBoundary } from "react-error-boundary";
import { useDisclosure } from "@mantine/hooks";
import { IconCheck, IconChevronDown, IconChevronUp } from "@tabler/icons-react";
import { notifications } from "@mantine/notifications";
import { pack } from "msgpackr";

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
  const fileUploadRef = React.useRef<HTMLInputElement>(null);
  if (viewer === undefined) viewer = React.useContext(ViewerContext)!;
  const conf = viewer.useGui((state) => state.guiConfigFromId[id]);
  const messageSender = makeThrottledMessageSender(viewer.websocketRef, 50);
  const { isUploading, upload } = useFileUpload({ viewer, componentId: id });

  // Handle nested containers.
  if (conf.type == "GuiAddFolderMessage")
    return (
      <Box pb={!last ? "0.125em" : 0}>
        <GeneratedFolder
          conf={conf}
          folderDepth={folderDepth}
          viewer={viewer}
        />
      </Box>
    );
  if (conf.type == "GuiAddTabGroupMessage")
    return <GeneratedTabGroup conf={conf} />;
  if (conf.type == "GuiAddMarkdownMessage") {
    let { visible } =
      viewer.useGui((state) => state.guiAttributeFromId[conf.id]) || {};
    visible = visible ?? true;
    if (!visible) return <></>;
    return (
      <Box pb="xs" px="sm" style={{ maxWidth: "95%" }}>
        <ErrorBoundary
          fallback={<Text align="center">Markdown Failed to Render</Text>}
        >
          <Markdown>{conf.markdown}</Markdown>
        </ErrorBoundary>
      </Box>
    );
  }

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
  let containerProps = {};
  switch (conf.type) {
    case "GuiAddButtonMessage":
      labeled = false;
      if (conf.color !== null) {
        inputColor =
          computeRelativeLuminance(
            theme.colors[conf.color][theme.fn.primaryShade()],
          ) > 50.0
            ? theme.colors.gray[9]
            : theme.white;
      }

      input = (
        <Button
          id={conf.id}
          fullWidth
          color={conf.color ?? undefined}
          onClick={() =>
            messageSender({
              type: "GuiUpdateMessage",
              id: conf.id,
              value: true,
            })
          }
          style={{ height: "2.125em" }}
          styles={{ inner: { color: inputColor + " !important" } }}
          disabled={disabled}
          size="sm"
          leftIcon={
            conf.icon_base64 === null ? undefined : (
              <Image
                /*^In Safari, both the icon's height and width need to be set, otherwise the icon is clipped.*/
                height="1em"
                width="1em"
                opacity={disabled ? 0.3 : 1.0}
                mr="-0.125em"
                sx={
                  inputColor === theme.white
                    ? {
                        // Make the color white.
                        filter: !disabled ? "invert(1)" : undefined,
                      }
                    : // Icon will be black by default.
                      undefined
                }
                src={"data:image/svg+xml;base64," + conf.icon_base64}
              />
            )
          }
        >
          {conf.label}
        </Button>
      );
      break;
    case "GuiAddUploadButtonMessage":
      labeled = false;
      if (isUploading) disabled = true;
      if (conf.color !== null) {
        inputColor =
          computeRelativeLuminance(
            theme.colors[conf.color][theme.fn.primaryShade()],
          ) > 50.0
            ? theme.colors.gray[9]
            : theme.white;
      }
      input = (
        <>
          <input
            type="file"
            style={{ display: "none" }}
            id={`file_upload_${conf.id}`}
            name="file"
            accept={conf.mime_type}
            ref={fileUploadRef}
            onChange={(e) => {
              const input = e.target as HTMLInputElement;
              if (!input.files) return;
              upload(input.files[0]);
            }}
          />
          <Button
            id={conf.id}
            fullWidth
            color={conf.color ?? undefined}
            onClick={() => {
              if (fileUploadRef.current === null) return;
              fileUploadRef.current.value = fileUploadRef.current.defaultValue;
              fileUploadRef.current.click();
            }}
            style={{ height: "2.125em" }}
            styles={{ inner: { color: inputColor + " !important" } }}
            disabled={disabled}
            size="sm"
            leftIcon={
              conf.icon_base64 === null ? undefined : (
                <Image
                  /*^In Safari, both the icon's height and width need to be set, otherwise the icon is clipped.*/
                  height="1em"
                  width="1em"
                  opacity={disabled ? 0.3 : 1.0}
                  mr="-0.125em"
                  sx={
                    inputColor === theme.white
                      ? {
                          // Make the color white.
                          filter: !disabled ? "invert(1)" : undefined,
                        }
                      : // Icon will be black by default.
                        undefined
                  }
                  src={"data:image/svg+xml;base64," + conf.icon_base64}
                />
              )
            }
          >
            {conf.label}
          </Button>
        </>
      );
      break;
    case "GuiAddSliderMessage":
      input = (
        <Flex justify="space-between">
          <Slider
            id={conf.id}
            size="xs"
            thumbSize={0}
            style={{ flexGrow: 1 }}
            styles={(theme) => ({
              thumb: {
                background: theme.fn.primaryColor(),
                borderRadius: "0.1rem",
                height: "0.75rem",
                width: "0.625rem",
              },
              trackContainer: {
                zIndex: 3,
                position: "relative",
              },
              markLabel: {
                transform: "translate(-50%, 0.03rem)",
                fontSize: "0.6rem",
                textAlign: "center",
              },
              marksContainer: {
                left: "0.2rem",
                right: "0.2rem",
              },
              markWrapper: {
                position: "absolute",
                top: `0.03rem`,
                ...(conf.marks === null
                  ? /*  Shift the mark labels so they don't spill too far out the left/right when we only have min and max marks. */
                    {
                      ":first-child": {
                        "div:nth-child(2)": {
                          transform: "translate(-0.2rem, 0.03rem)",
                        },
                      },
                      ":last-child": {
                        "div:nth-child(2)": {
                          transform: "translate(-90%, 0.03rem)",
                        },
                      },
                    }
                  : {}),
              },
              mark: {
                border: "0px solid transparent",
                background:
                  theme.colorScheme === "dark"
                    ? theme.colors.dark[4]
                    : theme.colors.gray[2],
                width: "0.42rem",
                height: "0.42rem",
                transform: `translateX(-50%)`,
              },
              markFilled: {
                background: disabled
                  ? theme.colorScheme === "dark"
                    ? theme.colors.dark[3]
                    : theme.colors.gray[4]
                  : theme.fn.primaryColor(),
              },
            })}
            pt="0.2em"
            showLabelOnHover={false}
            min={conf.min}
            max={conf.max}
            step={conf.step ?? undefined}
            precision={conf.precision}
            value={value}
            onChange={updateValue}
            marks={
              conf.marks === null
                ? [
                    {
                      value: conf.min,
                      label: `${parseInt(conf.min.toFixed(6))}`,
                    },
                    {
                      value: conf.max,
                      label: `${parseInt(conf.max.toFixed(6))}`,
                    },
                  ]
                : conf.marks
            }
            disabled={disabled}
          />
          <NumberInput
            value={value}
            onChange={(newValue) => {
              // Ignore empty values.
              newValue !== "" && updateValue(newValue);
            }}
            size="xs"
            min={conf.min}
            max={conf.max}
            hideControls
            step={conf.step ?? undefined}
            precision={conf.precision}
            sx={{ width: "3rem" }}
            styles={{
              input: {
                padding: "0.375em",
                letterSpacing: "-0.5px",
                minHeight: "1.875em",
                height: "1.875em",
              },
            }}
            ml="xs"
          />
        </Flex>
      );
      break;
    case "GuiAddMultiSliderMessage":
      input = (
        <MultiSlider
          id={conf.id}
          size="xs"
          thumbSize={0}
          styles={(theme) => ({
            thumb: {
              background: theme.fn.primaryColor(),
              borderRadius: "0.1rem",
              height: "0.75rem",
              width: "0.625rem",
            },
            trackContainer: {
              zIndex: 3,
              position: "relative",
            },
            markLabel: {
              transform: "translate(-50%, 0.03rem)",
              fontSize: "0.6rem",
              textAlign: "center",
            },
            marksContainer: {
              left: "0.2rem",
              right: "0.2rem",
            },
            markWrapper: {
              position: "absolute",
              top: `0.03rem`,
              ...(conf.marks === null
                ? /*  Shift the mark labels so they don't spill too far out the left/right when we only have min and max marks. */
                  {
                    ":first-child": {
                      "div:nth-child(2)": {
                        transform: "translate(-0.2rem, 0.03rem)",
                      },
                    },
                    ":last-child": {
                      "div:nth-child(2)": {
                        transform: "translate(-90%, 0.03rem)",
                      },
                    },
                  }
                : {}),
            },
            mark: {
              border: "0px solid transparent",
              background:
                theme.colorScheme === "dark"
                  ? theme.colors.dark[4]
                  : theme.colors.gray[2],
              width: "0.42rem",
              height: "0.42rem",
              transform: `translateX(-50%)`,
            },
            markFilled: {
              background: disabled
                ? theme.colorScheme === "dark"
                  ? theme.colors.dark[3]
                  : theme.colors.gray[4]
                : theme.fn.primaryColor(),
            },
          })}
          pt="0.2em"
          showLabelOnHover={false}
          min={conf.min}
          max={conf.max}
          step={conf.step ?? undefined}
          precision={conf.precision}
          value={value}
          onChange={updateValue}
          marks={
            conf.marks === null
              ? [
                  {
                    value: conf.min,
                    label: `${parseInt(conf.min.toFixed(6))}`,
                  },
                  {
                    value: conf.max,
                    label: `${parseInt(conf.max.toFixed(6))}`,
                  },
                ]
              : conf.marks
          }
          disabled={disabled}
          fixedEndpoints={conf.fixed_endpoints}
          minRange={conf.min_range || undefined}
        />
      );

      if (conf.marks?.some((x) => x.label) || conf.marks === null)
        containerProps = { ...containerProps, mb: "xs" };
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
          onChange={(newValue) => {
            // Ignore empty values.
            newValue !== "" && updateValue(newValue);
          }}
          styles={{
            input: {
              minHeight: "1.625rem",
              height: "1.625rem",
            },
          }}
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
          styles={{
            input: {
              minHeight: "1.625rem",
              height: "1.625rem",
              padding: "0 0.5em",
            },
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
          styles={{
            icon: {
              color: inputColor + " !important",
            },
          }}
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
          withinPortal
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
          // zIndex of dropdown should be >modal zIndex.
          // On edge cases: it seems like existing dropdowns are always closed when a new modal is opened.
          dropdownZIndex={1000}
          withinPortal
          styles={{
            input: { height: "1.625rem", minHeight: "1.625rem" },
            icon: { transform: "scale(0.8)" },
          }}
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
          // zIndex of dropdown should be >modal zIndex.
          // On edge cases: it seems like existing dropdowns are always closed when a new modal is opened.
          dropdownZIndex={1000}
          withinPortal
          styles={{ input: { height: "1.625rem", minHeight: "1.625rem" } }}
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

  if (conf.hint !== null)
    input = // We need to add <Box /> for inputs that we can't assign refs to.
      (
        <Tooltip
          zIndex={100}
          label={conf.hint}
          multiline
          w="15rem"
          withArrow
          openDelay={500}
          withinPortal
        >
          <Box
            sx={{
              display:
                // For checkboxes, we want to make sure that the wrapper
                // doesn't expand to the full width of the parent. This will
                // de-center the tooltip.
                conf.type === "GuiAddCheckboxMessage"
                  ? "inline-block"
                  : "block",
            }}
          >
            {input}
          </Box>
        </Tooltip>
      );

  if (labeled)
    input = (
      <LabeledInput
        id={conf.id}
        label={conf.label}
        input={input}
        folderDepth={folderDepth}
      />
    );

  return (
    <Box pb="0.5em" px="xs" {...containerProps}>
      {input}
    </Box>
  );
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

function GeneratedTabGroup({ conf }: { conf: GuiAddTabGroupMessage }) {
  const [tabState, setTabState] = React.useState<TabsValue>("0");
  const icons = conf.tab_icons_base64;

  return (
    <Tabs
      radius="xs"
      value={tabState}
      onTabChange={setTabState}
      sx={{ marginTop: "-0.75em" }}
    >
      <Tabs.List>
        {conf.tab_labels.map((label, index) => (
          <Tabs.Tab
            value={index.toString()}
            key={index}
            icon={
              icons[index] === null ? undefined : (
                <Image
                  /*^In Safari, both the icon's height and width need to be set, otherwise the icon is clipped.*/
                  height={"1.125em"}
                  width={"1.125em"}
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
      },
) {
  return (
    <Flex justify="space-between" columnGap="0.5em">
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

function useFileUpload({
  viewer,
  componentId,
}: {
  componentId: string;
  viewer: ViewerContextContents;
}) {
  const websocketRef = viewer.websocketRef;
  const updateUploadState = viewer.useGui((state) => state.updateUploadState);
  const uploadState = viewer.useGui(
    (state) => state.uploadsInProgress[componentId],
  );
  const totalBytes = uploadState?.totalBytes;

  // Cache total bytes string
  const totalBytesString = React.useMemo(() => {
    if (totalBytes === undefined) return "";
    let displaySize = totalBytes;
    const displayUnits = ["B", "K", "M", "G", "T", "P"];
    let displayUnitIndex = 0;
    while (displaySize >= 100 && displayUnitIndex < displayUnits.length - 1) {
      displaySize /= 1024;
      displayUnitIndex += 1;
    }
    return `${displaySize.toFixed(1)}${displayUnits[displayUnitIndex]}`;
  }, [totalBytes]);

  // Update notification status
  React.useEffect(() => {
    if (uploadState === undefined) return;
    const { notificationId, filename } = uploadState;
    if (uploadState.uploadedBytes === 0) {
      // Show notification.
      notifications.show({
        id: notificationId,
        title: "Uploading " + `${filename} (${totalBytesString})`,
        message: <Progress size="sm" value={0} />,
        autoClose: false,
        withCloseButton: false,
        loading: true,
      });
    } else {
      // Update progress.
      const progressValue = uploadState.uploadedBytes / uploadState.totalBytes;
      const isDone = progressValue === 1.0;
      notifications.update({
        id: notificationId,
        title: "Uploading " + `${filename} (${totalBytesString})`,
        message: !isDone ? (
          <Progress
            size="sm"
            // Default transition time is 100ms.
            // In Mantine v7, the transitionDuration prop can be used.
            styles={{ bar: { transition: "width 10ms linear" } }}
            value={100 * progressValue}
          />
        ) : (
          "File uploaded successfully."
        ),
        autoClose: isDone,
        withCloseButton: isDone,
        loading: !isDone,
        icon: isDone ? <IconCheck /> : undefined,
      });
    }
  }, [uploadState, totalBytesString]);

  const isUploading =
    uploadState !== undefined &&
    uploadState.uploadedBytes < uploadState.totalBytes;

  async function upload(file: File) {
    const chunkSize = 512 * 1024; // bytes
    const numChunks = Math.ceil(file.size / chunkSize);
    const transferUuid = uuid();
    const notificationId = "upload-" + transferUuid;

    const send = (message: Parameters<typeof pack>[0]) =>
      websocketRef.current?.send(pack(message));

    // Begin upload by setting initial state
    updateUploadState({
      componentId: componentId,
      uploadedBytes: 0,
      totalBytes: file.size,
      filename: file.name,
      notificationId,
    });

    send({
      type: "FileTransferStart",
      source_component_id: componentId,
      transfer_uuid: transferUuid,
      filename: file.name,
      mime_type: file.type,
      size_bytes: file.size,
      part_count: numChunks,
    });

    for (let i = 0; i < numChunks; i++) {
      const start = i * chunkSize;
      const end = (i + 1) * chunkSize;
      const chunk = file.slice(start, end);
      const buffer = await chunk.arrayBuffer();

      send({
        type: "FileTransferPart",
        source_component_id: componentId,
        transfer_uuid: transferUuid,
        part: i,
        content: new Uint8Array(buffer),
      });
    }
  }

  return {
    isUploading,
    upload,
  };
}
