import { ActionIcon } from "@mantine/core";
import { IconEye, IconEyeOff } from "@tabler/icons-react";
import { MantineReactTable } from "mantine-react-table";
import { MRT_ColumnDef } from "mantine-react-table";
import React from "react";
import { ViewerContext } from "..";

interface SceneTreeTableRow {
  name: string;
  visible: any; // Annotating this with ReactNode gives an error below, not sure why.
  subRows: SceneTreeTableRow[];
}

/* Table for seeing an overview of the scene tree, toggling visibility, etc. * */
export default function SceneTreeTable() {
  const viewer = React.useContext(ViewerContext)!;

  const attributesFromName = viewer.useSceneTree(
    (state) => state.attributesFromName
  );
  const nodeFromName = viewer.useSceneTree((state) => state.nodeFromName);
  const setVisibility = viewer.useSceneTree((state) => state.setVisibility);

  function getSceneTreeSubRows(
    parentName: string,
    isParentVisible: boolean
  ): SceneTreeTableRow[] {
    const node = nodeFromName[parentName];
    if (node === undefined) return [];

    return node.children.map((childName) => {
      const isVisible = attributesFromName[childName]?.visibility
        ? true
        : false;
      const isVisibleEffective = isVisible && isParentVisible;
      return {
        name: childName,
        visible: (
          <ActionIcon
            onClick={() => {
              setVisibility(childName, !isVisible);
            }}
            sx={{ opacity: isVisibleEffective ? "1.0" : "0.5" }}
          >
            {isVisible ? <IconEye /> : <IconEyeOff />}
          </ActionIcon>
        ),
        subRows: getSceneTreeSubRows(childName, isVisibleEffective),
      };
    });
  }

  const data = getSceneTreeSubRows("", true);
  const columns = React.useMemo<MRT_ColumnDef<SceneTreeTableRow>[]>(
    () => [
      {
        accessorKey: "visible", //simple recommended way to define a column
        header: "Visible",
        size: 50,
      },
      {
        accessorKey: "name", //simple recommended way to define a column
        header: "Name",
      },
    ],
    []
  );

  return (
    <MantineReactTable
      columns={columns}
      data={data}
      enableExpanding
      filterFromLeafRows
      enableDensityToggle={false}
      enableRowSelection
      enableHiding={false}
      enableGlobalFilter
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
        verticalSpacing: "sm",
      }}
      mantinePaginationProps={{
        showRowsPerPage: false,
        showFirstLastPageButtons: false,
      }}
      enableFullScreenToggle={false}
      // Show/hide buttons.
      renderTopToolbarCustomActions={({ table }) => {
        // For setting disabled, doesn't always give the right behavior:
        //     table.getIsSomeRowsSelected()
        //
        const disabled =
          table.getFilteredSelectedRowModel().flatRows.length == 0;
        return (
          <div style={{ display: "flex", gap: "8px" }}>
            <ActionIcon
              color="green"
              disabled={disabled}
              variant="filled"
              onClick={() => {
                table.getSelectedRowModel().flatRows.map((row) => {
                  setVisibility(row.getValue("name"), true);
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
                  setVisibility(row.getValue("name"), false);
                });
              }}
            >
              <IconEyeOff />
            </ActionIcon>
          </div>
        );
      }}
      // Row virtualization helps us reduce overhead when we have a lot of rows.
      enableRowVirtualization
    />
  );
}
