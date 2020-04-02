const meta = {
  // Metadata
  siteTitle: 'Zuhair Abbas - Data Science Portfolio',
  siteDescription: 'Zuhair Abbas - Data Scientist in the making.',
  siteTitleAlt: 'Zuhair Abbas',
  siteShortName: 'Zuhair Abbas',
  siteUrl: 'https://zuhairabs.netlify.com', // No trailing slash!
}

const social = {
  siteLogo: `src/static/logo.svg`,
  siteBanner: `${meta.siteUrl}/images/social-banner.png`,
  twitter: '@zuhairabs',
}

const website = {
  ...meta,
  ...social,
  disqusShortName: 'zuhairabs',
  googleAnalyticsID: "UA-79148290-1",
  
  // Manifest
  themeColor: '#6D83F2',
  backgroundColor: '#6D83F2',
}

module.exports = website
