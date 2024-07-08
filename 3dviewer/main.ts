import { serveDir, serveFile } from "jsr:@std/http/file-server";
import { FormDataReader } from "https://deno.land/x/oak/mod.ts";
const kv = await Deno.openKv();
const clients = new Map<string, WebSocket[]>(); // watch ID <=> websocket.

async function getFile(fileId: string): Promise<Uint8Array | null> {
    let offset = 0;
    const chunks = [];
    let chunkExists = true;

    while (chunkExists) {
        const key = `${fileId}_chunk${offset}`;
        console.log("getting chunk: " + key);
        const chunk = await kv.get([ "filechunks", key ]);

        if ((chunk.value ?? null) !== null) {
            chunks.push(chunk.value);
            offset++;
        } else {
            chunkExists = false;
        }
    }

    if (chunks.length === 0) {
        return null;
    }

    // Concatenate all chunks into a single Uint8Array
    const totalLength = chunks.reduce((acc, chunk) => acc + chunk.length, 0);
    const completeUint8Array = new Uint8Array(totalLength);
    let currentOffset = 0;

    chunks.forEach(chunk => {
        completeUint8Array.set(chunk, currentOffset);
        currentOffset += chunk.length;
    });

    return completeUint8Array;
}

async function setFile(fileId: string, completeUint8Array: Uint8Array): Promise<void> {
    const chunkSize = 64 * 1024; // 64 KB (Deno KV Store limit per key)
    let offset = 0;
    let chunkIndex = 0;

    while (offset < completeUint8Array.length) {
        const chunk = completeUint8Array.slice(offset, offset + chunkSize);
        const key = `${fileId}_chunk${chunkIndex}`;

        await kv.set([ "filechunks", key ], chunk);

        offset += chunkSize;
        chunkIndex++;
    }
    const chunkCount = chunkIndex;
    const key = `${fileId}_chunk${chunkIndex}`;
    await kv.delete([ "filechunks", key ]);
    return chunkIndex;
}

Deno.serve(async (req: Request) => {
  const pathname = new URL(req.url).pathname;

  if (pathname === "/") {
    return serveFile(req, "./index.html");
  }

  if (pathname === "/index.html") {
    const headers = new Headers();
    headers.set('Location', '/');
    return new Response(null, {
      status: 302,
      headers,
    });
  }

  if (pathname.startsWith("/assets")) {
    return serveDir(req, {
      fsRoot: "assets",
      urlRoot: "assets",
    });
  }

  if (req.method === 'POST' && pathname.startsWith('/upload/') && pathname.length > 8) {
    const id = pathname.substr(8);
    try {
      const formData = await req.formData(); // Parse multipart form data
      const files = [];
      let fileIdx = 0;
      for await (const [key, value] of formData.entries()) {
        if (value instanceof File) {
          const arrayBuffer = await value.arrayBuffer();
          const content = new Uint8Array(arrayBuffer);
          console.log(`Received file: ${value.name}, content-type: ${value.type}, content length: ${content.length}`);
          const chunkCount = await setFile(`${id}/${fileIdx}`, content);
          files.push({ name: value.name, contentLength: value.size, contentType: value.type, fileId: `${id}/${fileIdx}`, chunkCount: chunkCount });
          fileIdx++;
        } else {
          console.log(`Received field ${key}: ${value}`);
        }
      }
      if (!files.length) {
        return new Response('Files required', { status: 403 });
      }
      if (files.length > 2) {
        return new Response('Max 2 files to be accepted', { status: 403 });
      }
      const paths = files.map(el => `/upload/${el.fileId}/${crypto.randomUUID()}/${el.name}`);
      await kv.set(["files", id], files);
      if (clients.has(id)) {
        clients.get(id)?.forEach((socket) => {
            if (socket.readyState == WebSocket.OPEN) {
                socket.send(JSON.stringify({ result: "fileready", id: id, count: files.length, paths: paths }));
            }
        });
        console.log(`Broadcasted message to group ${id}`);
      } else {
        console.log(`No WebSocket clients in group ${id}`);
      }
      return new Response(null, { status: 200 });
    } catch (error) {
      console.error('Error handling file upload:', error);
      return new Response("Error handling file upload.", { status: 500 });
    }
  }

  if (req.method === 'GET' && pathname.startsWith("/upload/") && pathname.length > 8) {
    let idAndIdx = pathname.substr(8);
    let id = idAndIdx, idx = 0;
    const pos = idAndIdx.indexOf('/');
    if (pos !== -1) {
        id = idAndIdx.substr(0, pos);
        idx = parseInt(idAndIdx.substr(pos + 1));
        if (idx < 0) idx = 0;
    }

    console.log(`get kv file by id ${id}, idx ${idx}`);
    const files = await kv.get(["files", id]);
    console.log(`got kv file by id ${id}: ${JSON.stringify(files)} len ${files.length}`);
    if (!files || !files.value || !files.value.length || idx >= files.value.length || !files.value[idx]) {
      return new Response("404: Not Found", { status: 404 });
    }
    const content = await getFile(files.value[idx].fileId);
    if (!content) {
        return new Response("404: Not Found", { status: 404 });
    }
    const headers = new Headers();
    headers.set('Content-Length', content.length);
    headers.set('Content-Type', 'application/octet-stream');
    headers.set('Content-Disposition', `attachment; filename="${files.value[idx].name}"`);
    return new Response(content, {
      status: 200,
      headers,
    });
  }

  if (pathname === "/ws") {
    if (req.headers.get("upgrade") === "websocket") {
        const { socket, response } = Deno.upgradeWebSocket(req);
        let watchIdForThisSocket;
        socket.onmessage = (msg) => {
          try {
            const obj = JSON.parse(msg.data);
            if (obj.cmd === 'ping') {
              socket.send(JSON.stringify({ result: 'pong', id: watchIdForThisSocket }));
            }
            else if (obj.cmd === 'watch') {
              const id = obj.id;
              if ((id ?? null) === null) {
                console.log('no id in watch command');
                return;
              }

              if (!clients.has(id)) {
                clients.set(id, []);
              }
              clients.get(id)?.push(socket);
              watchIdForThisSocket = id;
              console.log(`Added WebSocket to group ${id}, count is ${clients.get(id).length}`);
              socket.send(JSON.stringify({ result: 'watching', id: id }));

              (async function() {
                  const result = await kv.get(["files", id]);
                  if (result.value) {
                      const files = result.value;
                      const paths = files.map(el => `/upload/${el.fileId}/${crypto.randomUUID()}/${el.name}`);
                      socket.send(JSON.stringify({ result: "fileready", id: id, count: files.length, paths: paths }));
                  }
              })();
            }
          } catch(e) {
            console.log('unable to parse ws json', e);
          }
        };
        socket.onclose = () => {
            if ((watchIdForThisSocket ?? null) !== null) {
                const arr = clients.get(watchIdForThisSocket);
                if (arr) {
                    clients.set(
                        watchIdForThisSocket,
                        arr.filter((sock) => sock !== socket)
                    );
                }
                const newCount = clients.get(watchIdForThisSocket)?.length;
                if (!newCount) clients.delete(watchIdForThisSocket);
                console.log(`WebSocket closed (watched ${watchIdForThisSocket}, count is ${newCount})`);
            } else {
                console.log('WebSocket closed');
            }
        };
        return response;
    } else {
        return new Response(null, { status: 501 });
    }
  }

  return new Response("404: Not Found", { status: 404 });
});
