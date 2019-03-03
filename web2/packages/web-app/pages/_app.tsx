import React from "react";
import App, { Container } from "next/app";
import Header from "../components/Header/Header";

import "../styles/main.scss";

class MyApp extends App {
  static async getInitialProps({ Component, ctx }) {
    let pageProps = {};

    if (Component.getInitialProps) {
      pageProps = await Component.getInitialProps(ctx);
    }

    return { pageProps };
  }

  render() {
    const { Component, pageProps } = this.props;

    return (
      <div id="site-theme">
        <Header />
        <Container>
          <Component {...pageProps} />
        </Container>
      </div>
    );
  }
}

export default MyApp;
