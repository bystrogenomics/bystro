import { withRouter } from "next/router";
import Link from "next/link";

import "./header.scss";

export default () => (
  <div id="header">
    <Link href="/">
      <a className="link home">/</a>
    </Link>
    <Link href="/jobs/public" prefetch>
      <a className="link">Public</a>
    </Link>
  </div>
);
