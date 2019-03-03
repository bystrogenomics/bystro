const withTypescript = require("@zeit/next-typescript");
const withCss = require("@zeit/next-css");
const withSass = require("@zeit/next-sass");

const ForkTsCheckerWebpackPlugin = require("fork-ts-checker-webpack-plugin");

const webpack = {
  webpack(config, options) {
    // Do not run type checking twice:
    if (options.isServer) config.plugins.push(new ForkTsCheckerWebpackPlugin());

    return config;
  }
};

module.exports = withCss(withSass(withTypescript(webpack)));
