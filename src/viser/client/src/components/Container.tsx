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