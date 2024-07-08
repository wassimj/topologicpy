@ECHO OFF
cd /D "%~dp0"
deno run --unstable-kv --allow-net --allow-read ./main.ts
