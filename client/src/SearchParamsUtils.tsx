/** Utilities for interacting with the URL search parameters.
 *
 * This lets us specify the websocket server + port from the URL. */

const key = "server";

export function getServersFromSearchParams() {
  return new URLSearchParams(window.location.search).getAll(key);
}

export function syncSearchParamServer(panelKey: number, server: string) {
  // Add/update servers in the URL bar.
  const serverParams = getServersFromSearchParams();
  if (panelKey >= serverParams.length) {
    serverParams.push(server);
  } else {
    serverParams[panelKey] = server;
  }
  window.history.replaceState(
    null,
    "Viser",
    // We could URLSearchParams() to build this string, but that would escape
    // it. We're going to just not escape the string. :)
    "?" + serverParams.map((s) => key + "=" + s).join("&")
  );
}

export function truncateSearchParamServers(length: number) {
  const serverParams = getServersFromSearchParams().slice(0, length);
  window.history.replaceState(
    null,
    "Viser",
    // We could URLSearchParams() to build this string, but that would escape
    // it. We're going to just not escape the string. :)
    "?" + serverParams.map((s) => key + "=" + s).join("&")
  );
}
