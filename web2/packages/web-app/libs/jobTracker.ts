import fetch from "isomorphic-unfetch";
import Callbacks from "./callbacks";

// enum types {
//   "all" = "all",
//   "public" = "public"
// }

let _all = {};
let _completed = {};
let _incomplete = {};
let _failed = {};
let _deleted = {};
let _public = {};
let _shared = {};

const callbacks = new Callbacks({
  public: [],
  all: [],
  completed: [],
  shared: [],
  deleted: [],
  failed: []
});

export const addCallback = callbacks.add;
export const removeCallback = callbacks.remove;

export default {
  get all() {
    return _all;
  },
  get public() {
    return _public;
  }
};

if (typeof window !== undefined) {
  fetch("http://localhost:9001/api/jobs/list/all/public")
    .then(r => r.json())
    .then(data => {
      console.info("data", data);
    })
    .catch(e => {
      console.info(e.message);
      console.info("failed to fetch", e.message);
    });
}
