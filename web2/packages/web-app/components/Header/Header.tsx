import { withRouter } from "next/router";
import Link from "next/link";

export default () => (
  <div id="header">
    <Link href="/">
      <a className="link home">/</a>
    </Link>
    <Link href="/jobs" prefetch>
      <a className="link">Jobs</a>
    </Link>
  </div>
);
