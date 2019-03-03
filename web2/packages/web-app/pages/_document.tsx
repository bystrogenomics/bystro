import Document, { Head, Main, NextScript } from "next/document";

class MyDocument extends Document {
  static async getInitialProps(ctx) {
    const initialProps = await Document.getInitialProps(ctx);
    return { ...initialProps };
  }

  render() {
    return (
      <html>
        <Head>
          <meta name="viewport" content="width=device-width, initial-scale=1" />
          <meta charSet="utf-8" />
          <link
            href="https://fonts.googleapis.com/css?family=Raleway"
            rel="stylesheet"
          />
        </Head>
        <body style={{ margin: 0 }}>
          <Main />
          <NextScript />
        </body>
      </html>
    );
  }
}

export default MyDocument;
