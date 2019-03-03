import React from "react";
import { view } from "react-easy-state";
import jobStore, { addCallback, removeCallback } from "../libs/jobTracker";

console.info("job tracker", jobStore);

let _callbackId: number;
class Jobs extends React.Component {
  state = {
    jobs: {}
  };

  static async getInitialProps({ query }: any) {
    return {
      type: query.type || "public"
    };
  }

  constructor(props: any) {
    super(props);

    _callbackId = addCallback(props.type, () => {
      this.setState({
        jobs: jobStore[props.type]
      });
    });
  }

  componentWillUnmount() {
    removeCallback(_callbackId);
  }

  render() {
    return Object.keys(this.state.jobs).map((key, idx) => (
      <div key={idx}>
        {key}:{this.state.jobs[key]}
      </div>
    ));
  }
}

export default view(Jobs);
