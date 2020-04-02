import React from "react"
import { graphql } from "gatsby";


import SEO from "src/components/seo";
import Layout from "src/components/Layout/Layout"
import BlogCard from "src/components/Blog/BlogCard";
import BlogLayout from "src/components/Blog/BlogLayout";

const TagsPage = ({ data, pageContext }) => {
  const { tag } = pageContext;
  const { edges, totalCount } = data.allMarkdownRemark;

  const tagHeader = `${totalCount} post${totalCount === 1 ? "" : "s"} tagged with "${tag}"`

  return (
    <Layout>
      <SEO title={tagHeader + ' | Zuhair Abbas'} />

      <BlogLayout>
        <h1>{tagHeader}</h1>
        <br />
        <br />
        {
          edges.map(({ node }) => {
            const { slug } = node.fields;
            const { title, date, tags, excerpt } = node.frontmatter;
            return (
              <BlogCard
                tags={tags}
                key={node.id}
                slug={slug}
                title={title}
                date={date}
                readtime={node.timeToRead}
                excerpt={excerpt}
              />
            )
          })
        }
      </BlogLayout>
    </Layout>
  )
}
export default TagsPage;


export const pageQuery = graphql`
  query($tag: String) {
    allMarkdownRemark(
      sort: { fields: [frontmatter___date], order: DESC }
      filter: { frontmatter: { tags: { in: [$tag] } } }
    ) {
      totalCount
      edges {
        node {
          id
          timeToRead
          fields {
            slug
          }
          frontmatter {
            tags
            title
            date(formatString: "MMMM DD, YYYY", locale: "en")
            excerpt
          }
        }
      }
    }
  }
`
