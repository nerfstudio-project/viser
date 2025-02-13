import { GuiHtmlMessage } from "../WebsocketMessages";

function HtmlComponent({ props }: GuiHtmlMessage) {
  if (!props.visible) return <></>;
  return <div dangerouslySetInnerHTML={{ __html: props.content }} />;
}

export default HtmlComponent;
