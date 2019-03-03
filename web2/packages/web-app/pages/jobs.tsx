import React from "react";
import { view } from "react-easy-state";
import jobStore, { addCallback, removeCallback } from "../libs/jobTracker";

console.info("job tracker", jobStore);

let _callbackId = null;
class Jobs extends React.Component {
  state = {
    allJobs: {}
  };

  constructor(props: any) {
    super(props);

    _callbackId = addCallback("all", () => {
      this.setState({
        allJobs: jobStore.all
      });
    });
  }

  componentWillUnmount() {
    removeCallback(_callbackId);
  }

  render() {
    return Object.keys(this.state.allJobs).map((key, idx) => (
      <div key={idx}>
        {key}:{this.state.allJobs[key]}
      </div>
    ));
  }
}

export default view(Jobs);
