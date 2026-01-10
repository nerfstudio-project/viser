/**
 * Simple Vite plugin to compress the inlined HTML output.
 * Uses browser's native DecompressionStream for decompression at runtime.
 *
 * This is a simplified alternative to vite-plugin-singlefile-compression
 * that doesn't add problematic import.meta.url polyfills.
 */

import { Plugin } from "vite";
import { gzipSync } from "zlib";

// Base64 encoding that's safe for embedding in HTML.
function toBase64(buffer: Buffer): string {
  return buffer.toString("base64");
}

// Minimal decompression loader script.
// Decodes base64, decompresses with DecompressionStream, and injects both CSS and JS.
// Waits for DOMContentLoaded to ensure the root element exists before React runs.
const loaderScript = `
(async()=>{
  const d=document.currentScript.dataset;
  const decompress=async(b64)=>{
    const b=atob(b64);
    const a=Uint8Array.from(b,c=>c.charCodeAt(0));
    const s=new DecompressionStream("gzip");
    const w=s.writable.getWriter();
    w.write(a);w.close();
    return new Response(s.readable).text();
  };
  if(d.s){const e=document.createElement("style");e.textContent=await decompress(d.s);document.head.appendChild(e);}
  if(d.c){
    const code=await decompress(d.c);
    const run=()=>{const e=document.createElement("script");e.type="module";e.textContent=code;document.head.appendChild(e);};
    if(document.readyState==="loading"){document.addEventListener("DOMContentLoaded",run);}else{run();}
  }
})();
`
  .trim()
  .replace(/\n/g, "");

export function compressHtml(): Plugin {
  return {
    name: "compress-html",
    enforce: "post",
    generateBundle(_, bundle) {
      for (const [fileName, chunk] of Object.entries(bundle)) {
        if (fileName.endsWith(".html") && chunk.type === "asset") {
          let html = chunk.source as string;
          const originalSize = Buffer.byteLength(html, "utf8");

          // Find and compress the inline style.
          const styleMatch = html.match(/<style[^>]*>([\s\S]*?)<\/style>/);
          let styleAttr = "";
          if (styleMatch) {
            const compressed = gzipSync(Buffer.from(styleMatch[1], "utf8"), {
              level: 9,
            });
            styleAttr = ` data-s="${toBase64(compressed)}"`;
            html = html.replace(styleMatch[0], "");
          }

          // Find and compress the inline script module.
          const scriptMatch = html.match(
            /<script type="module" crossorigin>([\s\S]*?)<\/script>/,
          );
          let scriptAttr = "";
          if (scriptMatch) {
            const compressed = gzipSync(Buffer.from(scriptMatch[1], "utf8"), {
              level: 9,
            });
            scriptAttr = ` data-c="${toBase64(compressed)}"`;
            html = html.replace(scriptMatch[0], "");
          }

          if (!styleMatch && !scriptMatch) {
            console.log(
              "[compress-html] No inline style or script found, skipping compression",
            );
            continue;
          }

          // Insert our loader script before </head>.
          const loaderTag = `<script${styleAttr}${scriptAttr}>${loaderScript}</script>`;
          html = html.replace("</head>", `${loaderTag}</head>`);

          const newSize = Buffer.byteLength(html, "utf8");
          console.log(
            `[compress-html] ${fileName}: ${(originalSize / 1024).toFixed(1)} KiB -> ${(newSize / 1024).toFixed(1)} KiB`,
          );

          chunk.source = html;
        }
      }
    },
  };
}
