import React from "react";
import { view } from "react-easy-state";
import jobTracker, { addCallback, removeCallback } from "../../libs/jobTracker";

let _callbackId: number;
const queryType = "public";

// https://stackoverflow.com/questions/46709773/typescript-react-rendering-an-array-from-a-stateless-functional-component
const JobList = (props: { jobs: any }) => (
  <React.Fragment>
    {Object.keys(props.jobs).map((key, idx) => (
      <p key={idx}>{props.jobs[key].name}</p>
    ))}
  </React.Fragment>
);

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
      this.setState(() => ({
        jobs: data
      }));
    });
  }

  componentWillUnmount() {
    removeCallback(this.state.jobType, _callbackId);
  }

  render() {
    return (
      <div>
        <JobList jobs={this.state.jobs} />
      </div>
    );
  }
}

export default view(Jobs);
