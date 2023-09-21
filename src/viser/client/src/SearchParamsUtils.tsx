/** Utilities for interacting with the URL search parameters.
 *
 * This lets us specify the websocket server + port from the URL. */

export const searchParamKey = "websocket";

export function syncSearchParamServer(server: string) {
  setServerParams([server]);
}

function setServerParams(serverParams: string[]) {
  // No need to update the URL bar if the websocket port matches the HTTP port.
  // So if we navigate to http://localhost:8081, this should by default connect to ws://localhost:8081.
  if (
    serverParams.length === 1 &&
    (window.location.host.includes(
      serverParams[0].replace("ws://", "").replace("/", "")
    ) ||
      window.location.host.includes(
        serverParams[0].replace("wss://", "").replace("/", "")
      ))
  )
    serverParams = [];

  window.history.replaceState(
    null,
    "Viser",
    // We could use URLSearchParams() to build this string, but that would escape
    // it. We're going to just not escape the string. :)
    serverParams.length === 0
      ? window.location.href.split("?")[0]
      : `?${serverParams.map((s) => `${searchParamKey}=${s}`).join("&")}`
  );
}
