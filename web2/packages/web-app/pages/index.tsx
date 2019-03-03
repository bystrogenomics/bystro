import DefaultView from "../components/DefaultView/DefaultView";
import "../styles/pages/index.scss";
import jobTracker from "../libs/jobTracker";

export default () => (
  <div id="index" className="center">
    <h1>
      <a href="https://github.com/akotlar/bystro" target="_blank">
        Bystro
      </a>
    </h1>
    <div className="subtitle">
      Variant annotation & filtering for any size data.
    </div>
    <div className="subtitle">
      Please cite our{" "}
      <a target="_blank" href="https://doi.org/10.1186/s13059-018-1387-3">
        <b>Genome Biology</b>
      </a>{" "}
      paper!
    </div>
    <p>
      <DefaultView />
    </p>
  </div>
);
