import * as React from "react";
import { Button, Flex } from "@mantine/core";
import { ViserInputComponent } from "./common";
import { GuiAddButtonGroupMessage } from "../WebsocketMessages";
import { GuiComponentContext } from "../ControlPanel/GuiComponentContext";

export default function ButtonGroupComponent({
  id,
  hint,
  label,
  visible,
  disabled,
  options,
}: GuiAddButtonGroupMessage) {
  const { messageSender } = React.useContext(GuiComponentContext)!;
  if (!visible) return <></>;
  return (
    <ViserInputComponent {...{ id, hint, label }}>
      <Flex justify="space-between" columnGap="xs">
        {options.map((option, index) => (
          <Button
            key={index}
            onClick={() =>
              messageSender({
                type: "GuiUpdateMessage",
                id: id,
                updates: { value: option },
              })
            }
            style={{ flexGrow: 1, width: 0 }}
            disabled={disabled}
            size="compact-xs"
            variant="outline"
          >
            {option}
          </Button>
        ))}
      </Flex>
    </ViserInputComponent>
  );
}
