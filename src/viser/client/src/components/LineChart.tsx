import React from "react";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from "recharts";
import { Box, Paper, Text } from "@mantine/core";
import { GuiLineChartMessage } from "../WebsocketMessages";
import { folderWrapper } from "./Folder.css";

interface DataPoint {
  x: number;
  y: number;
  [key: string]: number;
}

function processSeriesData(seriesData: GuiLineChartMessage["props"]["_series_data"]): DataPoint[] {
  if (seriesData.length === 0) return [];

  // Create a map of x values to data points
  const xValueMap = new Map<number, DataPoint>();

  // Populate the map with all x values and their corresponding y values for each series
  seriesData.forEach((series) => {
    series.data.forEach((point) => {
      if (!xValueMap.has(point.x)) {
        xValueMap.set(point.x, { x: point.x, y: 0 });
      }
      const dataPoint = xValueMap.get(point.x)!;
      dataPoint[series.name] = point.y;
    });
  });

  // Convert map to array and sort by x value
  return Array.from(xValueMap.values()).sort((a, b) => a.x - b.x);
}

export default function LineChartComponent({
  props: { visible, title, x_label, y_label, _series_data, height },
}: GuiLineChartMessage) {
  if (!visible) return <></>;

  const data = processSeriesData(_series_data);

  // Generate colors for series that don't have explicit colors
  const getColor = (index: number, customColor?: string | null) => {
    if (customColor) return customColor;
    
    const colors = [
      "#8884d8", "#82ca9d", "#ffc658", "#ff7c7c", "#8dd1e1",
      "#d084d0", "#ffb347", "#87ceeb", "#98fb98", "#f0e68c"
    ];
    return colors[index % colors.length];
  };

  return (
    <Paper className={folderWrapper} withBorder style={{ padding: "12px" }}>
      {title && (
        <Text size="sm" weight={600} align="center" mb="xs">
          {title}
        </Text>
      )}
      <Box style={{ width: "100%", height: height }}>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
            <XAxis 
              dataKey="x" 
              stroke="#666"
              fontSize={12}
              label={x_label ? { value: x_label, position: "insideBottom", offset: -5 } : undefined}
            />
            <YAxis 
              stroke="#666"
              fontSize={12}
              label={y_label ? { value: y_label, angle: -90, position: "insideLeft" } : undefined}
            />
            <Tooltip 
              contentStyle={{ 
                backgroundColor: "#fff", 
                border: "1px solid #ccc", 
                borderRadius: "4px",
                fontSize: "12px"
              }}
            />
            {_series_data.length > 1 && (
              <Legend 
                wrapperStyle={{ fontSize: "12px" }}
              />
            )}
            {_series_data.map((series, index) => (
              <Line
                key={series.name}
                type="monotone"
                dataKey={series.name}
                stroke={getColor(index, series.color)}
                strokeWidth={2}
                dot={{ r: 3 }}
                activeDot={{ r: 4 }}
                connectNulls={false}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </Box>
    </Paper>
  );
}