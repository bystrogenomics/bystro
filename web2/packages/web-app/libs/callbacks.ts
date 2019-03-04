export default class Callbacks {
  private _callbacks: { [type: string]: (() => void)[] };
  private _callbackTimeouts: { [type: string]: NodeJS.Timeout } = {};

  constructor(callbacks: { [type: string]: (() => void)[] }) {
    this._callbacks = callbacks;
  }

  add = (type: string, action: () => void) => {
    this._callbacks[type].push(action);
    console.info("added", this._callbacks[type]);
    return this._callbacks[type].length;
  };

  remove = (type: string, elem: number) => {
    if (this._callbacks[type].length === 0) {
      return;
    }

    // Shift appears to be faster than splice
    if (elem == 1) {
      this._callbacks[type].shift();
    } else if (elem === this._callbacks[type].length) {
      this._callbacks[type].pop();
    } else {
      this._callbacks[type].splice(elem - 1, 1);
    }
  };

  call = (type: string) => {
    if (this._callbacks[type].length === 0) {
      return;
    }

    if (this._callbackTimeouts[type]) {
      clearTimeout(this._callbackTimeouts[type]);
      delete this._callbackTimeouts[type];
    }

    this._callbackTimeouts[type] = setTimeout(() => {
      this._callbacks[type].forEach(v => v());
    }, 100);
  };
}
