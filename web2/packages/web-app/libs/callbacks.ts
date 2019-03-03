export default class Callbacks {
  private _callbacks: { [type: string]: any[] };
  private _callbackTimeouts: { [type: string]: NodeJS.Timeout } = {};

  constructor() {
    this._callbacks = {
      all: [],
      incomplete: []
    };
  }

  add = (type: string, action: () => void) => {
    this._callbacks[type].push(action);

    this._callbacks.length;
  };

  remove = (type: string, elem: number) => {
    if (elem == 1) {
      this._callbacks[type].shift();
    } else if (elem == this._callbacks[type].length) {
      this._callbacks[type].pop();
    } else {
      this._callbacks[type].splice(elem - 1, 1);
    }
  };

  call = (type: string) => {
    if (!this._callbacks[type]) {
      return;
    }

    if (this._callbackTimeouts[type]) {
      clearTimeout(this._callbackTimeouts[type]);
      delete this._callbackTimeouts[type];
    }

    this._callbackTimeouts[type] = setTimeout(
      ctx => {
        ctx._callbacks[type].forEach(v => v());
      },
      100,
      this
    );
  };
}
