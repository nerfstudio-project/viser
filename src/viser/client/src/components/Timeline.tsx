import React from "react";
import { TimelineMessage } from "../WebsocketMessages";
import { Box, Flex, Slider, Text, ActionIcon } from "@mantine/core";
import { IconPlayerPlay, IconPlayerPause } from "@tabler/icons-react";
import { ViewerContext } from "../ViewerContext";
import { useThrottledMessageSender } from "../WebsocketUtils";

/** Root component that reads timeline state and conditionally renders */
export function TimelineSlider() {
  const viewer = React.useContext(ViewerContext)!;
  const conf = viewer.useGui((state) => state.timeline);

  // Don't render if no timeline or not visible
  if (conf === null || !conf.props.visible) {
    return null;
  }

  return <TimelineSliderInner {...conf} />;
}

/** Format seconds to HH:MM:SS */
function formatTime(seconds: number): string {
  const hrs = Math.floor(seconds / 3600);
  const mins = Math.floor((seconds % 3600) / 60);
  const secs = Math.floor(seconds % 60);
  return `${hrs}:${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
}

/** Inner component that renders the actual timeline UI */
function TimelineSliderInner({ value, props }: TimelineMessage) {
  const { min, max, step, precision } = props;
  const [isPlaying, setIsPlaying] = React.useState(false);
  const [localValue, setLocalValue] = React.useState(value);
  const messageSender = useThrottledMessageSender(50).send;

  // Update local value when prop changes
  React.useEffect(() => {
    setLocalValue(value);
  }, [value]);

  // Update slider value (throttled to avoid message spam)
  const updateValue = (newValue: number) => {
    setLocalValue(newValue);
    messageSender({
      type: "GuiUpdateMessage",
      uuid: "__timeline__",
      updates: { value: newValue },
    });
  };

  // Handle play button click
  const handlePlayClick = () => {
    const newPlayingState = !isPlaying;
    setIsPlaying(newPlayingState);

    // Send play event to Python
    messageSender({
      type: "GuiUpdateMessage",
      uuid: "__timeline__",
      updates: { play: true },
    });
  };

  // Calculate elapsed and remaining time
  const elapsedTime = formatTime(localValue);
  const remainingTime = formatTime(max - localValue);

  return (
    <Box
      style={{
        position: "fixed",
        bottom: 0,
        left: "25%",
        width: "50%",
        backgroundColor: "var(--mantine-color-body)",
        borderTop: "1px solid var(--mantine-color-default-border)",
        padding: "1em",
        zIndex: 1000,
        boxShadow: "0 -2px 10px rgba(0, 0, 0, 0.1)",
      }}
    >
      <Flex gap="md" align="center">
        {/* Play/Pause Button */}
        <ActionIcon
          size="lg"
          variant="transparent"
          onClick={handlePlayClick}
          aria-label={isPlaying ? "Pause" : "Play"}
          style={{
            backgroundColor: "transparent",
            border: "none",
          }}
        >
          {isPlaying ? (
            <IconPlayerPause size={18} />
          ) : (
            <IconPlayerPlay size={18} />
          )}
        </ActionIcon>

        {/* Elapsed Time */}
        <Text
          size="sm"
          style={{ 
            fontVariantNumeric: "tabular-nums",
            minWidth: "4.5rem",
            textAlign: "right",
            backgroundColor: "transparent",
            border: "none",
          }}
        >
          {elapsedTime}
        </Text>

        {/* Slider */}
        <Slider
          style={{ flexGrow: 1 }}
          size="sm"
          min={min}
          max={max}
          step={step}
          precision={precision}
          value={localValue}
          onChange={updateValue}
          marks={[]}
          styles={{
            markLabel: { display: 'none' }
          }}
        />

        {/* Remaining Time */}
        <Text
          size="sm"
          style={{ 
            fontVariantNumeric: "tabular-nums",
            minWidth: "4.5rem",
            textAlign: "left",
            backgroundColor: "transparent",
            border: "none",
          }}
        >
          {remainingTime}
        </Text>
      </Flex>
    </Box>
  );
}
