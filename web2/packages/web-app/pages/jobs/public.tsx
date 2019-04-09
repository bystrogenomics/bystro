import React from "react";
import { view } from "react-easy-state";
import jobTracker, { addCallback, removeCallback } from "../../libs/jobTracker";

let _callbackId: number;
const queryType = "public";

class Jobs extends React.Component {
  state = {
    jobs: jobTracker.public,
    jobType: queryType
  };

  static async getInitialProps({ query }: any) {
    return {
      type: query.type || queryType
    };
  }

  constructor(props: any) {
    super(props);

    this.state.jobType = props.type;

    _callbackId = addCallback(props.type, data => {
      console.info("updated", data);
      this.setState(() => ({
        jobs: data
      }));
    });

    console.info("data", this.state.jobs);
  }

  componentWillUnmount() {
    removeCallback(this.state.jobType, _callbackId);
  }

  render() {
    return Object.keys(this.state.jobs).map((key, idx) => (
      <div key={idx}>{this.state.jobs[key]["_id"]}</div>
    ));
  }
}

export default view(Jobs);
