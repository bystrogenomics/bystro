import { withRouter } from "next/router";
import Link from "next/link";

export default () => (
  <div id="header">
    <Link href="/jobs">
      <a>Jobs</a>
    </Link>
  </div>
);
