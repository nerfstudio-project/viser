import { ViewerContext } from "../App";
import { ActionIcon, Modal } from "@mantine/core";
import { useDisclosure } from "@mantine/hooks";
import {
  IconCaretDown,
  IconCaretRight,
  IconEye,
  IconEyeOff,
  IconMaximize,
} from "@tabler/icons-react";
import { MantineReactTable } from "mantine-react-table";
import { MRT_ColumnDef } from "mantine-react-table";
import React from "react";

interface SceneTreeTableRow {
  name: string;
  visible: any; // Annotating this with ReactNode gives an error below, not sure why.
  subRows: SceneTreeTableRow[];
}

/* Table for seeing an overview of the scene tree, toggling visibility, etc. * */
export default function SceneTreeTable(props: { compact: boolean }) {
  const viewer = React.useContext(ViewerContext)!;

  const nodeFromName = viewer.useSceneTree((state) => state.nodeFromName);
  const setLabelVisibility = viewer.useSceneTree(
    (state) => state.setLabelVisibility
  );
  function setVisible(name: string, visible: boolean) {
    const attr = viewer.nodeAttributesFromName.current;
    if (attr[name] === undefined) attr[name] = {};
    attr[name]!.visibility = visible;
    rerenderTable();
  }

  // For performance, scene node visibility is stored in a ref instead of the
  // zustand state. This means that re-renders for the table need to be
  // triggered manually when visibilities are updated.
  const [, setTime] = React.useState(Date.now());
  function rerenderTable() {
    setTime(Date.now());
  }
  React.useEffect(() => {
    const interval = setInterval(rerenderTable, 500);
    return () => {
      clearInterval(interval);
    };
  }, []);

  // Debouncing to suppress onMouseEnter and onMouseDown events from
  // re-renders.
  const debouncedReady = React.useRef(false);
  debouncedReady.current = false;
  setTimeout(() => {
    debouncedReady.current = true;
  }, 50);

  function getSceneTreeSubRows(
    parentName: string,
    parentCount: number,
    isParentVisible: boolean
  ): SceneTreeTableRow[] {
    const node = nodeFromName[parentName];
    if (node === undefined) return [];

    return node.children.map((childName) => {
      const isVisible =
        viewer.nodeAttributesFromName.current[childName]?.visibility ?? true;
      const isVisibleEffective = isVisible && isParentVisible;

      const VisibleIcon = isVisible ? IconEye : IconEyeOff;
      return {
        name: childName,
        visible: (
          <ActionIcon
            onMouseDown={() => {
              const isVisible =
                viewer.nodeAttributesFromName.current[childName]?.visibility ??
                true;
              if (debouncedReady.current) {
                setVisible(childName, !isVisible);
              }
            }}
            onClick={(evt) => {
              // Don't propagate click events to the row containing the icon.
              //
              // If we don't stop propagation, clicking the visibility icon
              // will also expand/collapse nodes in the scene tree.
              evt.stopPropagation();
            }}
            onMouseEnter={(event) => {
              if (event.buttons !== 0) {
                const isVisible =
                  viewer.nodeAttributesFromName.current[childName]
                    ?.visibility ?? true;
                if (debouncedReady.current) {
                  setVisible(childName, !isVisible);
                }
              }
            }}
            sx={{ opacity: isVisibleEffective ? "1.0" : "0.5" }}
          >
            <VisibleIcon />
          </ActionIcon>
        ),
        subRows: getSceneTreeSubRows(
          childName,
          parentCount + 1,
          isVisibleEffective
        ),
      };
    });
  }
  const data = getSceneTreeSubRows("", 0, true);
  const columns = React.useMemo<MRT_ColumnDef<SceneTreeTableRow>[]>(
    () => [
      {
        accessorKey: "visible", //simple recommended way to define a column
        header: "",
        size: 20,
      },
      {
        accessorKey: "name", //simple recommended way to define a column
        header: "Name",
        Cell: function (props) {
          const row = props.row;
          const cell = props.cell;

          const CaretIcon = row.getIsExpanded()
            ? IconCaretDown
            : IconCaretRight;
          return (
            <>
              <CaretIcon
                style={{
                  opacity: row.subRows?.length === 0 ? "0.0" : "0.4",
                  marginLeft: `${(0.75 * row.depth).toString()}em`,
                }}
                size="1em"
              />
              {(cell.getValue() as string)
                .split("/")
                .filter((part) => part.length > 0)
                .map((part, index, all) => (
                  // We set userSelect to prevent users from accidentally
                  // selecting text when dragging over the hide/show icons.
                  <span key={index} style={{ userSelect: "none" }}>
                    <span style={{ opacity: "0.4" }}>
                      {index === all.length - 1 ? "/" : `/${part}`}
                    </span>
                    {index === all.length - 1 ? part : ""}
                  </span>
                ))}
            </>
          );
        },
      },
    ],
    []
  );

  const [sceneTreeOpened, { open: openSceneTree, close: closeSceneTree }] =
    useDisclosure(false);
  return (
    <>
      {props.compact && (
        <Modal
          padding="0"
          withCloseButton={false}
          opened={sceneTreeOpened}
          onClose={closeSceneTree}
          size="xl"
          centered
        >
          <SceneTreeTable compact={false} />
        </Modal>
      )}
      <MantineReactTable
        columns={columns}
        data={data}
        enableExpanding={false}
        filterFromLeafRows
        enableDensityToggle={false}
        enableRowSelection={!props.compact}
        enableHiding={false}
        enableGlobalFilter
        enableColumnActions={false}
        enableTopToolbar
        enableBottomToolbar={false}
        enableColumnFilters={false}
        enablePagination={false}
        initialState={{ density: "xs", expanded: true }}
        mantineExpandAllButtonProps={{
          size: "sm",
        }}
        mantineExpandButtonProps={{ size: "sm", sx: { width: "0 !important" } }}
        mantineSelectAllCheckboxProps={{ size: "sm" }}
        mantineSelectCheckboxProps={{ size: "sm" }}
        mantineTableProps={{
          verticalSpacing: 2,
        }}
        mantineTableContainerProps={{ sx: { maxHeight: "30em" } }}
        mantinePaginationProps={{
          showRowsPerPage: false,
          showFirstLastPageButtons: false,
        }}
        mantineTableBodyRowProps={({ row }) => ({
          onPointerOver: () => {
            setLabelVisibility(row.getValue("name"), true);
          },
          onPointerOut: () => {
            setLabelVisibility(row.getValue("name"), false);
          },
          ...(row.subRows === undefined || row.subRows.length === 0
            ? {}
            : {
                onClick: () => {
                  row.toggleExpanded();
                },
                sx: {
                  cursor: "pointer",
                },
              }),
        })}
        enableFullScreenToggle={false}
        // Show/hide buttons.
        renderTopToolbarCustomActions={
          props.compact
            ? () => {
                return (
                  <ActionIcon onClick={openSceneTree}>
                    <IconMaximize />
                  </ActionIcon>
                );
              }
            : ({ table }) => {
                // For setting disabled, doesn't always give the right behavior:
                //     table.getIsSomeRowsSelected()
                //
                const disabled =
                  table.getFilteredSelectedRowModel().flatRows.length === 0;
                return (
                  <div style={{ display: "flex", gap: "8px" }}>
                    <ActionIcon
                      color="green"
                      disabled={disabled}
                      variant="filled"
                      onClick={() => {
                        table.getSelectedRowModel().flatRows.map((row) => {
                          setVisible(row.getValue("name"), true);
                        });
                      }}
                    >
                      <IconEye />
                    </ActionIcon>
                    <ActionIcon
                      color="gray"
                      disabled={disabled}
                      variant="filled"
                      onClick={() => {
                        table.getSelectedRowModel().flatRows.map((row) => {
                          setVisible(row.getValue("name"), false);
                        });
                      }}
                    >
                      <IconEyeOff />
                    </ActionIcon>
                  </div>
                );
              }
        }
        // Row virtualization helps us reduce overhead when we have a lot of rows.
        enableRowVirtualization
      />
    </>
  );
}
