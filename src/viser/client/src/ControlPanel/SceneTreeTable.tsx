import {
  IconCaretDown,
  IconCaretRight,
  IconEye,
  IconEyeOff,
  IconPencil,
  IconDeviceFloppy,
  IconX,
} from "@tabler/icons-react";
import React from "react";
import {
  caretIcon,
  editIconWrapper,
  propsWrapper,
  tableRow,
  tableWrapper,
} from "./SceneTreeTable.css";
import { useDisclosure } from "@mantine/hooks";
import { useForm } from "@mantine/form";
import { ViewerContext } from "../App";
import {
  Box,
  Flex,
  ScrollArea,
  TextInput,
  Tooltip,
  ColorInput,
} from "@mantine/core";

function EditNodeProps({
  nodeName,
  close,
}: {
  nodeName: string;
  close: () => void;
}) {
  const viewer = React.useContext(ViewerContext)!;
  const node = viewer.useSceneTree((state) => state.nodeFromName[nodeName]);
  const updateSceneNode = viewer.useSceneTree((state) => state.updateSceneNode);

  if (node === undefined) {
    return null;
  }

  // We'll use JSON, but add support for Infinity.
  // We use infinity for point cloud rendering norms.
  function stringify(value: any) {
    if (value == Number.POSITIVE_INFINITY) {
      return "Infinity";
    } else {
      return JSON.stringify(value);
    }
  }
  function parse(value: string) {
    if (value === "Infinity") {
      return Number.POSITIVE_INFINITY;
    } else {
      return JSON.parse(value);
    }
  }

  const props = node.message.props;
  console.log(props);
  const initialValues = Object.fromEntries(
    Object.entries(props)
      .filter(([, value]) => !(value instanceof Uint8Array))
      .map(([key, value]) => [key, stringify(value)]),
  );

  const form = useForm({
    initialValues: {
      ...initialValues,
    },
    validate: {
      ...Object.fromEntries(
        Object.keys(initialValues).map((key) => [
          key,
          (value: string) => {
            try {
              parse(value);
              return null;
            } catch (e) {
              return "Invalid JSON";
            }
          },
        ]),
      ),
    },
  });

  const handleSubmit = (values: Record<string, string>) => {
    Object.entries(values).forEach(([key, value]) => {
      if (value !== initialValues[key]) {
        try {
          const parsedValue = parse(value);
          updateSceneNode(nodeName, { [key]: parsedValue });
          // Update the form value to match the parsed value
          form.setFieldValue(key, stringify(parsedValue));
        } catch (e) {
          console.error("Failed to parse JSON:", e);
        }
      }
    });
  };

  return (
    <Box
      className={propsWrapper}
      component="form"
      onSubmit={form.onSubmit(handleSubmit)}
    >
      <Box
        style={{
          position: "absolute",
          top: "0.3em",
          right: "0.4em",
        }}
      >
        <Tooltip label={"Close props"}>
          <IconX
            style={{
              cursor: "pointer",
              width: "1em",
              height: "1em",
              display: "block",
              color: "--mantine-color-error",
              opacity: "0.7",
            }}
            onClick={(evt) => {
              evt.stopPropagation();
              close();
            }}
          />
        </Tooltip>
      </Box>
      {Object.entries(props).map(([key, value]) => {
        if (value instanceof Uint8Array) {
          return null;
        }

        const isDirty = form.values[key] !== initialValues[key];

        return (
          <Flex key={key} align="center">
            <Box size="sm" fz="xs" style={{ flexGrow: "1" }}>
              {key.charAt(0).toUpperCase() + key.slice(1).split("_").join(" ")}
            </Box>
            <Flex gap="xs" w="9em">
              {(() => {
                // Check if this is a color property
                try {
                  const parsedValue = parse(form.values[key]);
                  const isColorProp =
                    key.toLowerCase().includes("color") &&
                    Array.isArray(parsedValue) &&
                    parsedValue.length === 3 &&
                    parsedValue.every((v) => typeof v === "number");

                  if (isColorProp) {
                    // Convert RGB array [0-1] to hex color
                    const rgbToHex = (r: number, g: number, b: number) => {
                      const toHex = (n: number) => {
                        const hex = Math.round(n).toString(16);
                        return hex.length === 1 ? "0" + hex : hex;
                      };
                      return "#" + toHex(r) + toHex(g) + toHex(b);
                    };

                    // Convert hex color to RGB array [0-1]
                    const hexToRgb = (hex: string) => {
                      const r = parseInt(hex.slice(1, 3), 16);
                      const g = parseInt(hex.slice(3, 5), 16);
                      const b = parseInt(hex.slice(5, 7), 16);
                      return [r, g, b];
                    };

                    return (
                      <ColorInput
                        size="xs"
                        styles={{
                          input: { height: "1.625rem", minHeight: "1.625rem" },
                          // icon: { transform: "scale(0.8)" },
                        }}
                        w="100%"
                        value={rgbToHex(
                          parsedValue[0],
                          parsedValue[1],
                          parsedValue[2],
                        )}
                        onChange={(hex) => {
                          const rgb = hexToRgb(hex);
                          form.setFieldValue(key, stringify(rgb));
                        }}
                        onKeyDown={(e) => {
                          if (e.key === "Enter") {
                            e.preventDefault();
                            form.onSubmit(handleSubmit)();
                          }
                        }}
                        rightSection={
                          <IconDeviceFloppy
                            style={{
                              width: "1rem",
                              height: "1rem",
                              opacity: isDirty ? 1.0 : 0.3,
                              cursor: isDirty ? "pointer" : "default",
                            }}
                            onClick={() => {
                              if (isDirty) {
                                form.onSubmit(handleSubmit)();
                              }
                            }}
                          />
                        }
                      />
                    );
                  }
                } catch (e) {
                  // If parsing fails, fall back to TextInput
                }

                // Default TextInput for non-color properties
                return (
                  <TextInput
                    size="xs"
                    styles={{
                      input: {
                        height: "1.625rem",
                        minHeight: "1.625rem",
                        width: "100%",
                      },
                      // icon: { transform: "scale(0.8)" },
                    }}
                    w="100%"
                    {...form.getInputProps(key)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter") {
                        e.preventDefault();
                        form.onSubmit(handleSubmit)();
                      }
                    }}
                    rightSection={
                      <IconDeviceFloppy
                        style={{
                          width: "1rem",
                          height: "1rem",
                          opacity: isDirty ? 1.0 : 0.3,
                          cursor: isDirty ? "pointer" : "default",
                        }}
                        onClick={() => {
                          if (isDirty) {
                            form.onSubmit(handleSubmit)();
                          }
                        }}
                      />
                    }
                  />
                );
              })()}
            </Flex>
          </Flex>
        );
      })}
      <Box fz="xs" opacity="0.5">
        Changes will be reset on refresh.
      </Box>
    </Box>
  );
}

/* Table for seeing an overview of the scene tree, toggling visibility, etc. * */
export default function SceneTreeTable() {
  const viewer = React.useContext(ViewerContext)!;
  const childrenName = viewer.useSceneTree(
    (state) => state.nodeFromName[""]!.children,
  );
  return (
    <ScrollArea className={tableWrapper}>
      {childrenName.map((name) => (
        <SceneTreeTableRow
          nodeName={name}
          key={name}
          isParentVisible={true}
          indentCount={0}
        />
      ))}
    </ScrollArea>
  );
}

const SceneTreeTableRow = React.memo(function SceneTreeTableRow(props: {
  nodeName: string;
  isParentVisible: boolean;
  indentCount: number;
}) {
  const viewer = React.useContext(ViewerContext)!;
  const childrenName = viewer.useSceneTree(
    (state) => state.nodeFromName[props.nodeName]!.children,
  );
  const expandable = childrenName.length > 0;

  const [expanded, { toggle: toggleExpanded }] = useDisclosure(false);

  function setOverrideVisibility(name: string, visible: boolean | undefined) {
    const attr = viewer.nodeAttributesFromName.current;
    attr[name]!.overrideVisibility = visible;
    rerenderTable();
  }
  const setLabelVisibility = viewer.useSceneTree(
    (state) => state.setLabelVisibility,
  );

  // For performance, scene node visibility is stored in a ref instead of the
  // zustand state. This means that re-renders for the table need to be
  // triggered manually when visibilities are updated.
  const [, setTime] = React.useState(Date.now());
  function rerenderTable() {
    setTime(Date.now());
  }
  React.useEffect(() => {
    const interval = setInterval(rerenderTable, 200);
    return () => {
      clearInterval(interval);
    };
  }, []);

  const attrs = viewer.nodeAttributesFromName.current[props.nodeName];
  const isVisible =
    (attrs?.overrideVisibility === undefined
      ? attrs?.visibility
      : attrs.overrideVisibility) ?? true;
  const isVisibleEffective = isVisible && props.isParentVisible;
  const VisibleIcon = isVisible ? IconEye : IconEyeOff;

  const [modalOpened, { open: openEditModal, close: closeEditModal }] =
    useDisclosure(false);

  return (
    <>
      <Box
        className={tableRow}
        style={{
          cursor: expandable ? "pointer" : undefined,
          marginLeft: (props.indentCount * 0.75).toString() + "em",
        }}
        onClick={expandable ? toggleExpanded : undefined}
        onMouseOver={() => setLabelVisibility(props.nodeName, true)}
        onMouseOut={() => setLabelVisibility(props.nodeName, false)}
      >
        <Box
          style={{
            opacity: expandable ? 1 : 0.3,
          }}
        >
          {expanded ? (
            <IconCaretDown className={caretIcon} />
          ) : (
            <IconCaretRight className={caretIcon} />
          )}
        </Box>
        <Box style={{ width: "1.5em", height: "1.5em" }}>
          <Tooltip label="Override visibility">
            <VisibleIcon
              style={{
                cursor: "pointer",
                opacity: isVisibleEffective ? 0.85 : 0.25,
                width: "1.5em",
                height: "1.5em",
                display: "block",
              }}
              onClick={(evt) => {
                evt.stopPropagation();
                setOverrideVisibility(props.nodeName, !isVisible);
              }}
            />
          </Tooltip>
        </Box>
        <Box style={{ flexGrow: "1" }}>
          {props.nodeName
            .split("/")
            .filter((part) => part.length > 0)
            .map((part, index, all) => (
              // We set userSelect to prevent users from accidentally
              // selecting text when dragging over the hide/show icons.
              <span key={index} style={{ userSelect: "none" }}>
                <span style={{ opacity: "0.3" }}>
                  {index === all.length - 1 ? "/" : `/${part}`}
                </span>
                {index === all.length - 1 ? part : ""}
              </span>
            ))}
        </Box>
        {!modalOpened ? (
          <Box
            className={editIconWrapper}
            style={{
              width: "1.25em",
              height: "1.25em",
              display: "block",
              transition: "opacity 0.2s",
            }}
          >
            <Tooltip label={"Override props (experimental)"}>
              <IconPencil
                style={{
                  cursor: "pointer",
                  width: "1.25em",
                  height: "1.25em",
                  display: "block",
                }}
                onClick={(evt) => {
                  evt.stopPropagation();
                  openEditModal();
                }}
              />
            </Tooltip>
          </Box>
        ) : null}
      </Box>
      {modalOpened ? (
        <EditNodeProps nodeName={props.nodeName} close={closeEditModal} />
      ) : null}
      {expanded
        ? childrenName.map((name) => (
            <SceneTreeTableRow
              nodeName={name}
              isParentVisible={isVisibleEffective}
              key={name}
              indentCount={props.indentCount + 1}
            />
          ))
        : null}
    </>
  );
});
