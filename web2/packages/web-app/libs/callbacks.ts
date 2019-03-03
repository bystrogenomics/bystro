export default class Callbacks {
  private _callbacks: { [type: string]: any[] };

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
    } else if (elem == _callbacks[type].length) {
      this._callbacks[type].pop();
    } else {
      this._callbacks[type].splice(elem - 1, 1);
    }
  };

  call = (type: types) => {
    if (!this._callbacks[type]) {
      return;
    }

    this._callbacks[type].forEach(v => {
      v();
    });
  };
}
