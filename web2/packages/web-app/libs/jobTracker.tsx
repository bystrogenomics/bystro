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
  "all" = "all"
}

const _callbacks: { [property: string]: (() => void)[] } = {
  all: [],
  incomplete: []
};

export const addCallback = (type: string, action: () => void) => {
  _callbacks[type].push(action);

  return _callbacks.length;
};

// TODO: optimize
export const removeCallback = (type: string, elem: number) => {
  if (elem == 1) {
    _callbacks[type].shift();
  } else if (elem == _callbacks[type].length) {
    _callbacks[type].pop();
  } else {
    _callbacks[type].splice(elem - 1, 1);
  }
};

const callCallback = (type: types) => {
  _callbacks[type].forEach(v => {
    v();
  });
};

setTimeout(() => {
  _all = {
    1: "something",
    2: "else"
  };

  callCallback(types.all);
}, 2000);
