import React from "react";
import { view } from "react-easy-state";
import jobStore, { addCallback, removeCallback } from "../libs/jobTracker";

let _callbackId: number;
class Jobs extends React.Component {
  state = {
    jobs: {},
    jobType: ""
  };

  static async getInitialProps({ query }: any) {
    return {
      type: query.type || "public"
    };
  }

  constructor(props: any) {
    super(props);

    this.state.jobType = props.type;

    _callbackId = addCallback(props.type, () => {
      this.setState({
        jobs: jobStore[props.type]
      });
    });
  }

  componentWillUnmount() {
    removeCallback(this.state.jobType, _callbackId);
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
