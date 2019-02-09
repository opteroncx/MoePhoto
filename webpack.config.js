const webpack = require('webpack')
const MiniCssExtractPlugin = require("mini-css-extract-plugin")
const TerserPlugin = require('terser-webpack-plugin')
const CleanWebpackPlugin = require('clean-webpack-plugin')
const ManifestPlugin = require('webpack-manifest-plugin')

module.exports = (env, argv) => {
  return {
    mode: 'production',
    target: 'web',
    entry: {
      app: './src/js/app.js',
      progress: './src/js/progress.js'
    },
    output: {
      filename: 'static/js/[name].js',
      chunkFilename: 'static/js/[chunkhash].js',
      path: __dirname,
      publicPath: '/'
    },
    externals: [],
    module: {
      rules: [{
        test: /\.(sa|sc|c)ss$/,
        exclude: /\.min\.css/,
        use: [
          { loader: MiniCssExtractPlugin.loader },
          { loader: 'css-loader', options: { importLoaders: 1 } },
          'postcss-loader'
        ]
      }, {
        test: /\.min\.css/,
        use: [
          { loader: MiniCssExtractPlugin.loader },
          { loader: 'css-loader' }
        ]
      }, {
        test: /\.(png|jpg|gif)$/,
        loader: 'url-loader',
        options: {
          limit: 10000,
          outputPath: 'static/bg'
        }
      }, {
        test: /\.(svg|eot|ttf|woff|woff2)$/,
        loader: 'url-loader',
        options: {
          limit: 10000,
          outputPath: 'static/fonts'
        }
      }]
    },
    optimization: {
      splitChunks: {
        name: false,
        cacheGroups: {
          commons: {
            name: "common",
            chunks: "initial",
            minChunks: 2
          },
          vendors: {
            test: /[\\/]node_modules[\\/]/,
            name: "vendors",
            chunks: "all"
          },
          styles: {
            name: 'styles',
            test: /\.css$/,
            chunks: 'all',
            enforce: true
          }
        }
      },
      minimizer: [
        new TerserPlugin({
          exclude: /\.min\.js/,
          parallel: 4,
          terserOptions: {
            toplevel: true,
            compress: {
              arguments: true,
              ecma: 6,
              keep_infinity: true,
              passes: 2,
              toplevel: true,
              unsafe: true
            },
            mangle: true,
            output: {
              ecma: 6,
              shebang: false
            }
          }
        })
      ]
    },
    plugins: [
      new webpack.ProgressPlugin(),
      new ManifestPlugin({
        fileName: 'static/manifest.json'
      }),
      new webpack.ProvidePlugin({
        $: 'jquery',
        jQuery: 'jquery',
        'window.jQuery': 'jquery'
      }),
      new MiniCssExtractPlugin({ filename: "static/css/[contenthash].css" }),
      new CleanWebpackPlugin(['static/css', 'static/js', 'static/fonts', 'static/bg'])
    ]
  }
}