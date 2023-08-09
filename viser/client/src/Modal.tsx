import { ViewerContext } from "./App";
import { GuiModalMessage } from "./WebsocketMessages";
import GeneratedGuiContainer from "./ControlPanel/Generated";
import { Modal } from "@mantine/core";
import { useContext } from "react";

export function ViserModal() {
  const viewer = useContext(ViewerContext)!;

  const modalList = viewer.useGui((state) => state.modals);
  const modals = modalList.map((conf, index) => {
    return <GeneratedModal key={conf.id} conf={conf} index={index} />;
  });

  return modals;
}

function GeneratedModal({
  conf,
  index,
}: {
  conf: GuiModalMessage;
  index: number;
}) {
  const viewer = useContext(ViewerContext)!;
  const popModal = viewer.useGui((state) => state.popModal);
  const removeContainer = viewer.useGui((state) => state.removeGuiContainer);

  return (
    <Modal
      opened={true}
      title={conf.title}
      onClose={() => {
        removeContainer(conf.id);
        popModal();
      }}
      centered
      zIndex={10000 + index}
    >
      <GeneratedGuiContainer containerId={conf.id} />
    </Modal>
  );
}
