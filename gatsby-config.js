const config = require('./config/website');

module.exports = {
  siteMetadata: {
    title: config.siteTitle,
    description: config.siteDescription,
    twitter: config.twitter,
    siteUrl: config.siteUrl,
    siteLogo: config.siteLogo,
    siteBanner: config.siteBanner,
  },
  plugins: [
    // MARKDOWN
    {
      resolve: `gatsby-transformer-remark`,
      options: {
        plugins: [
          {
            resolve: `gatsby-remark-copy-linked-files`,
            options: {
              destinationDir: `path/to/dir`,
              ignoreFileExtensions: [`png`, `jpg`, `jpeg`, `bmp`, `tiff`],
            },
          },
          {
            resolve: `gatsby-remark-katex`,
            options: {
              // Add any KaTeX options from https://github.com/KaTeX/KaTeX/blob/master/docs/options.md here
              strict: `ignore`
            }
          },
          `gatsby-remark-embedder`,
          {
            resolve: `gatsby-remark-autolink-headers`,
            options: {
              className: `gatsby-remark-autolink`,
              maintainCase: true,
              removeAccents: true,
            },
          },
          {
            resolve: `gatsby-remark-prismjs`,
            options: {
              classPrefix: "language-",
              inlineCodeMarker: null,
              aliases: {},
              showLineNumbers: true,
              noInlineHighlight: false,
            }
          },
          {
            resolve: `gatsby-remark-images`,
            options: {
              maxWidth: 590,
              showCaptions: true
            }
          },
          // `gatsby-plugin-social-banners`,
          `gatsby-plugin-twitter`,
          {
            resolve: `gatsby-plugin-nprogress`,
            options: {
              // Setting a color is optional.
              color: `tomato`,
              // Disable the loading spinner.
              showSpinner: true,
            },
          },
          `gatsby-plugin-sitemap`,
          `gatsby-plugin-robots-txt`,
          `gatsby-plugin-netlify`,
          `gatsby-plugin-offline`,
          {
            resolve: `gatsby-plugin-manifest`,
            options: {
              name: `GatsbyJS`,
              short_name: `GatsbyJS`,
              start_url: `/`,
              background_color: `#f7f0eb`,
              theme_color: `#a2466c`,
              display: `standalone`,
            }
          },
          {
            resolve: `gatsby-plugin-canonical-urls`,
            options: {
              siteUrl: `https://www.ahmedbesbes.com`,
            },
          },
          {
            resolve: 'gatsby-plugin-page-progress',
            options: {
              includePaths: [{ regex: '^/blog' }, { regex: '^/case-studies' }],
              excludePaths: ['/'],
              height: 3,
              prependToBody: false,
              color: `#663399`,
            }
          },
          {
            resolve: `gatsby-remark-social-cards`,
            options: {
              title: {
                // This is the frontmatter field the title should come from
                field: "title",
                // Currently only supports DejaVuSansCondensed
                // More fonts coming soon!
                font: "DejaVuSansCondensed",
                color: "black", // black|white
                size: 32, // 16|24|32|48|64
                style: "bold", // normal|bold|italic
                x: null, // Will default to xMargin
                y: null, // Will default to yMargin
              },
              meta: {
                // The parts array is what generates the bottom text
                // Pass an array with strings and objects
                // The following array will generate:
                // "- Author Name » September 13"
                // The objects are used to pull data from your markdown's
                // frontmatter. { field: "author" } pulls the author set
                // in the frontmatter. { field: "category" } would pull
                // the category set. Any field can be used as parts
                // Note: Only pass the "format" property on date fields
                parts: [
                  "- ",
                  { field: "author" },
                  " » ",
                  { field: "date", format: "mmmm dS" },
                ],
                // Currently only supports DejaVuSansCondensed
                // More fonts coming soon!
                font: "DejaVuSansCondensed",
                color: "black", // black|white
                size: 24, // 16|24|32|48|64
                style: "normal", // normal|bold|italic
                x: null, // Will default to xMargin
                y: null, // Will default to cardHeight - yMargin - size
              },
              background: "#B1ECE8", // Background color for the card
              xMargin: 15, // Edge margin used when x value is not set
              yMargin: 15,// Edge margin used when y value is not set
            }
          }
        ]
      }
    },

    // SOURCE FILE SYSTEM -
    // SOURCE JSON
    // `gatsby-transformer-json`,
    // {
    //   resolve: `gatsby-source-filesystem`,
    //   options: {
    //     path: `${__dirname}/content/json`,
    //   },
    // },
    // SOURCE MARKDOWN
    {
      resolve: `gatsby-source-filesystem`,
      options: {
        name: 'case-studies',
        path: `${__dirname}/content/case-studies`,
      },
    },
    {
      resolve: `gatsby-source-filesystem`,
      options: {
        name: 'blog',
        path: `${__dirname}/content/blog/`,
      },
    },


    // IMAGE TRANSFORMER
    `gatsby-transformer-sharp`,
    `gatsby-plugin-sharp`,
    {
      resolve: `gatsby-source-filesystem`,
      options: {
        name: `images`,
        path: `src/static/images`,
      },
    },

    // manifest & helmet
    `gatsby-plugin-react-helmet`,

    {
      resolve: `gatsby-plugin-manifest`,
      options: {
        name: config.siteTitleAlt,
        short_name: config.siteShortName,
        start_url: `/`,
        background_color: config.backgroundColor,
        theme_color: config.themeColor,
        display: `standalone`,
        icon: config.siteLogo,
      },
    },
    // this (optional) plugin enables Progressive Web App + Offline functionality
    // To learn more, visit: https://gatsby.dev/offline
    // `gatsby-plugin-offline`,

    // fonts
    // https://fonts.googleapis.com/css?family=Karla:400,700|Montserrat:400,600,700,900&display=swap
    // families: ['Karla&display=swap', 'Montserrat:400,700,900&display=swap']
    // {
    //   resolve: 'gatsby-plugin-web-font-loader',
    //   options: {
    //     custom: {
    //       families: [
    //         'Karla',
    //         'Montserrat:n4,n7,n9'
    //       ],
    //       urls: [
    //         'https://fonts.googleapis.com/css?family=Karla&display=swap',
    //         'https://fonts.googleapis.com/css?family=Montserrat:400,700,900&display=swap'
    //       ]
    //     }
    //   }
    // },

    // NProgress
    {
      resolve: `gatsby-plugin-nprogress`,
      options: {
        color: `#6D83F2`,
        showSpinner: false,
      },
    },
    {
      resolve: 'gatsby-plugin-google-analytics',
      options: {
        trackingId: config.googleAnalyticsID,
        head: true,
      },
    },
    // others
    {
      resolve: 'gatsby-plugin-robots-txt',
      options: {
        host: config.siteUrl,
        sitemap: `${config.siteUrl}/sitemap.xml`,
        env: {
          development: {
            policy: [{ userAgent: '*', disallow: ['/'] }]
          },
          production: {
            policy: [{ userAgent: '*', allow: '/', disallow: '/goodies' }]
          }
        }
      }
    },
    {
      resolve: `gatsby-plugin-sitemap`,
      options: {
        exclude: [`/blog/tags/*`, `/goodies`],
      }
    },
    `gatsby-plugin-styled-components`,
    `gatsby-plugin-root-import`
  ],
}
