import Link from "next/link";
import "./defaultView.scss";

export default () => (
  <span className="default-view">
    <Link href="/public" prefetch>
      <a>Try</a>
    </Link>
    <Link href="/help" prefetch>
      <a>Guide</a>
    </Link>
  </span>
);
