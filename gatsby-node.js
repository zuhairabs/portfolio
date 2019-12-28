const axios = require('axios');
const { createRemoteFileNode } = require(`gatsby-source-filesystem`);
const { createFilePath } = require(`gatsby-source-filesystem`);
const path = require('path');

const slugify = require('./src/components/slugify.js');

exports.onCreateNode = ({ node, getNode, actions }) => {
  // const { createNodeField } = actions;

  // if (node.internal.type !== 'MarkdownRemark') return;

  // const fileNode = getNode(node.parent);
  // const slugFromTitle = slugify(node.frontmatter.title);

  // // sourceInstanceName defined if its a blog or case-studie
  // const sourceInstanceName = fileNode.sourceInstanceName;

  // // extract the name of the file because we need to sort by it's name
  // // `001-blahblah`
  // const fileIndex = fileNode.name.substr(2, 1);

  // // create slug nodes
  // createNodeField({
  //   node,
  //   name: 'slug',
  //   // value will be {blog||case-studies}/my-title
  //   value: '/' + sourceInstanceName + '/' + slugFromTitle
  // });

  // // adds a posttype field to extinguish between blog and case-study
  // createNodeField({
  //   node,
  //   name: 'posttype',
  //   // value will be {blog||case-studies}
  //   value: sourceInstanceName
  // });

  // // if sourceInstanceName is case-studies then add the fileIndex field because we need
  // // this to sort the Projects with their respective file name `001-blahblah`
  // if (sourceInstanceName == 'case-studies') {
  //   createNodeField({
  //     node,
  //     name: 'fileIndex',
  //     value: fileIndex
  //   })
  // }

  const { createNodeField } = actions;

  if (node.internal.type === 'MarkdownRemark') {
    const fileNode = getNode(node.parent);
    const sourceInstanceName = fileNode.sourceInstanceName;
    const fileIndex = fileNode.name.substr(2, 1);
    if (typeof node.frontmatter.slug !== 'undefined') {
      const slugFromSlug = slugify(node.frontmatter.slug)
      createNodeField({
          node,
          name: 'slug',
          // value will be {blog||case-studies}/my-title
          value: '/' + sourceInstanceName + '/' + slugFromSlug
      });
    } else {
      const slugFromTitle = slugify(node.frontmatter.title)
      createNodeField({
          node,
          name: 'slug',
          // value will be {blog||case-studies}/my-title
          value: '/' + sourceInstanceName + '/' + slugFromTitle
      });
    }
    createNodeField({
        node,
        name: 'posttype',
        // value will be {blog||case-studies}
        value: sourceInstanceName
      });
    if (sourceInstanceName == 'case-studies') {
        createNodeField({
          node,
          name: 'fileIndex',
          value: fileIndex
        })
      }   
  }
}

exports.createPages = ({ actions, graphql }) => {
  const { createPage, createRedirect} = actions;

  createRedirect({ fromPath: '/how-to-score-08134-in-titanic-kaggle-challenge.html', toPath: '/blog/kaggle-titanic-competition', isPermanent: true })
  createRedirect({ fromPath: '/how-to-mine-newsfeed-data-and-extract-interactive-insights-in-python.html', toPath: '/blog/interactive-topic-mining', isPermanent: true })
  createRedirect({ fromPath: '/sentiment-analysis-on-twitter-using-word2vec-and-keras.html', toPath: '/blog/sentiment-analysis-with-keras-and-word2-vec', isPermanent: true })
  createRedirect({ fromPath: '/understanding-deep-convolutional-neural-networks-with-a-practical-use-case-in-tensorflow-and-keras.html', toPath: '/blog/introduction-to-cnns', isPermanent: true })
  createRedirect({ fromPath: '/overview-and-benchmark-of-traditional-and-deep-learning-models-in-text-classification.html', toPath: '/blog/benchmarking-sentiment-analysis-models', isPermanent: true })
  createRedirect({ fromPath: '/automate-the-diagnosis-of-knee-injuries-with-deep-learning-part-1-an-overview-of-the-mrnet-dataset.html', toPath: '/blog/acl-tear-detection-part-1', isPermanent: true })
  createRedirect({ fromPath: '/automate-the-diagnosis-of-knee-injuries-with-deep-learning-part-2-building-an-acl-tear-classifier.html', toPath: '/blog/acl-tear-detection-part-2', isPermanent: true })
  createRedirect({ fromPath: '/automate-the-diagnosis-of-knee-injuries-with-deep-learning-part-3-interpret-models-predictions.html', toPath: '/blog/acl-tear-detection-part-3', isPermanent: true })
  createRedirect({ fromPath: '/introduction-to-neural-networks-and-deep-learning-from-scratch.html', toPath: '/blog/neural-networks-from-scratch', isPermanent: true })
  createRedirect({ fromPath: '/introduction-to-automl-with-mlbox.html', toPath: '/blog/introduction-to-mlbox', isPermanent: true })
  createRedirect({ fromPath: '/end-to-end-ml.html', toPath: '/blog/end-to-end-machine-learning', isPermanent: true })

  const caseStudyTemplate = path.resolve('src/templates/case-study.js');
  const blogPostTemplate = path.resolve('src/templates/blog-post.js');
  const tagTemplate = path.resolve("src/templates/tags.js");

  return graphql(`
    {
      allMarkdownRemark {
        edges {
          node {
            frontmatter {
              tags
            }
            fields {
              slug
              posttype
            }
          }
        }
      }
    }
  `).then(res => {
    if (res.errors) return Promise.reject(res.errors);

    const edges = res.data.allMarkdownRemark.edges;
    edges.forEach(({ node }) => {
      // if the posttype is case-studies then createPage with caseStudyTemplate
      // we get fileds.posttype because we created this node with onCreateNode
      if (node.fields.posttype === 'case-studies') {
        createPage({
          path: node.fields.slug,
          component: caseStudyTemplate,
          context: {
            slug: node.fields.slug
          }
        })
      } else {
        const tagSet = new Set();
        // for each tags on the frontmatter add them to the set
        node.frontmatter.tags.forEach(tag => tagSet.add(tag));
        const tagList = Array.from(tagSet);
        // for each tags create a page with the specific `tag slug` (/blog/tags/:name)
        // pass the tag through the PageContext
        tagList.forEach(tag => {
          createPage({
            path: `/blog/tags/${slugify(tag)}/`,
            component: tagTemplate,
            context: {
              tag
            }
          });
        });

        // create each individual blog post with `blogPostTemplate`
        createPage({
          path: node.fields.slug,
          component: blogPostTemplate,
          context: {
            slug: node.fields.slug
          }
        })
      }
    })

  })
}


exports.sourceNodes = ({ actions, createNodeId, createContentDigest, store, cache }) => {
  const { createNode } = actions;
  const CC_PROJECTS_URI = 'https://anuraghazra.github.io/CanvasFun/data.json';


  const createCreativeCodingNode = (project, i) => {
    const node = {
      id: createNodeId(`${i}`),
      parent: null,
      children: [],
      internal: {
        type: `CreativeCoding`,
        content: JSON.stringify(project),
        contentDigest: createContentDigest(project)
      },
      ...project
    }

    // create `allCreativeCoding` Node
    createNode(node);
  }

  // GET IMAGE THUMBNAILS
  const createRemoteImage = async (project, i) => {
    try {
      // it will download the remote files
      await createRemoteFileNode({
        id: `${i}`,
        url: project.img, // the image url
        store,
        cache,
        createNode,
        createNodeId
      });
    } catch (error) {
      throw new Error('error creating remote img node - ' + error)
    }
  }

  // promise based sourcing
  return axios.get(CC_PROJECTS_URI)
    .then(res => {
      res.data.forEach((project, i) => {
        createCreativeCodingNode(project, i);
        createRemoteImage(project, i);
      })
    }).catch(err => {
      // just create a dummy node to pass the build if faild to fetch data
      createCreativeCodingNode({
        id: '0',
        demo: '',
        img: '',
        title: 'Error while loading Data',
        src: '',
      }, 0);
      throw new Error(err);
    })
}