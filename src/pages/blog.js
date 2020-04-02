import React from "react";
import { useStaticQuery, graphql } from 'gatsby';

import Layout from "src/components/Layout/Layout"
import SEO from "src/components/seo";

import BlogCard from 'src/components/Blog/BlogCard';
import BlogLayout from 'src/components/Blog/BlogLayout';

const BlogPage = () => {
   const blogposts = useStaticQuery(
    graphql`
      query blogPageQuery {
        allMarkdownRemark(
          filter: {fields: {posttype: {eq: "blog"}}}
          sort: {fields: frontmatter___date, order: DESC}
        ){
          edges {
            node {
              id
              timeToRead
              frontmatter {
                title
                date(formatString: "MMMM DD, YYYY", locale: "en")
                tags
                excerpt
              }
              fields {
                slug
              }
            }
          }
        }
      }
    `
  )
  return (
    <Layout>
      <SEO title="Zuhair Abbas - Data Science Portfolio" />

      <BlogLayout>
        {
          blogposts.allMarkdownRemark.edges.map(({ node }) => (
            <BlogCard
              key={node.id}
              slug={node.fields.slug}
              title={node.frontmatter.title}
              date={node.frontmatter.date}
              tags={node.frontmatter.tags}
              readtime={node.timeToRead}
              excerpt={node.frontmatter.excerpt}
            />
          ))
        }
      </BlogLayout>
    </Layout>
  )
}

export default BlogPage
