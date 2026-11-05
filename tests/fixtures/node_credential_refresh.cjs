// Run the released TypeScript SDK in a separate process; only OAuth transport is fake.
const fs = require("node:fs");
const { setTimeout: sleep } = require("node:timers/promises");
const [sdk, path, first] = process.argv.slice(2);
const open = fs.openSync;
fs.openSync = function (name, ...args) {
  try { return open.call(this, name, ...args); }
  catch (error) {
    if (name === path + ".lock-v1" && error.code === "EEXIST")
      fs.writeFileSync(path + ".node-waiting", "");
    throw error;
  }
};
require(sdk).refresh({
  credentialsPath: path, profile: "mine", onlyIfExpiring: true,
  async fetch() {
    if (first !== "node") throw Error("Node repeated the Python refresh");
    fs.writeFileSync(path + ".node-refreshing", "");
    const deadline = Date.now() + 10000;
    while (!fs.existsSync(path + ".python-waiting")) {
      if (Date.now() >= deadline) throw Error("Python did not contend");
      await sleep(10);
    }
    return Response.json({access_token: "node", refresh_token: "next", token_type: "Bearer", expires_in: 3600});
  },
}).then(creds => { if (creds.accessToken !== first) throw Error("unexpected token"); })
  .catch(error => { console.error(error); process.exitCode = 1; });
