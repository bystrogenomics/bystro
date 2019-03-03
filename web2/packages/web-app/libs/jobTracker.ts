import fetch from "isomorphic-unfetch";
import Callbacks from "./callbacks";

let _all = {};
let _completed = {};
let _incomplete = {};
let _failed = {};
let _deleted = {};
let _public = {};
let _shared = {};

export default {
  get all() {
    return _all;
  }
};

enum types {
  "all" = "all",
  "public" = "public"
}

const callbacks = new Callbacks();

export const addCallback = callbacks.add;
export const removeCallback = callbacks.remove;

_all = {
  1: "something",
  2: "else"
};

for (let i = 0; i < 1000; i++) {
  callbacks.call(types.all);
}
