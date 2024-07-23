import { notifications } from "@mantine/notifications";
import { detect } from "detect-browser";
import { useEffect } from "react";

export function BrowserWarning() {
  useEffect(() => {
    const browser = detect();

    // Browser version are based loosely on support for SIMD, OffscreenCanvas.
    //
    // https://caniuse.com/?search=simd
    // https://caniuse.com/?search=OffscreenCanvas
    if (browser === null || browser.version === null) {
      console.log("Failed to detect browser");
      notifications.show({
        title: "Could not detect browser version",
        message:
          "Your browser version could not be detected. It may not be supported.",
        autoClose: false,
        color: "red",
      });
    } else {
      const version = parseFloat(browser.version);
      console.log(`Detected ${browser.name} version ${version}`);
      if (
        (browser.name === "chrome" && version < 91) ||
        (browser.name === "edge" && version < 91) ||
        (browser.name === "firefox" && version < 89) ||
        (browser.name === "opera" && version < 77) ||
        (browser.name === "safari" && version < 16.4)
      )
        notifications.show({
          title: "Unsuppported browser",
          message: `Your browser (${browser.name.slice(0, 1).toUpperCase() + browser.name.slice(1)}/${browser.version}) is outdated, which may cause problems. Consider updating.`,
          autoClose: false,
          color: "red",
        });
    }
  });
  return null;
}
