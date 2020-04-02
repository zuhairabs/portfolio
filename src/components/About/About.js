import React from "react";

import SkewBg from 'src/components/common/SkewBg';
import PageHeader from 'src/components/common/PageHeader';
import Flex from "src/components/common/Flex";

import Quote from './Quote';
import Avatar from './Avatar';

import { AboutWrapper, AboutInfo } from './About.style';


const About = () => {
  return (
    <AboutWrapper id="about">
      <PageHeader>About Me</PageHeader>
      <SkewBg />
      <AboutInfo>
        <div>
          <Avatar src="me.png" />
        </div>
        <p>
          Hello there 👋 I'm Zuhair. 
          <br />
          I hope you stumbled upon this website on purpose! Otherwise, let me introduce you to my world 🌍 
          <br />
          <br />
          I'm a data scientist living in India 🇮🇳. I've been working on many projects.
          <br />
          Part of my work include crafting, building and deploying AI applications to answer business issues.
          <br/>
          I also blog about technical topics such as deep learning. You can check my open source <a className="about__rkmscc-link" href="#projects"> projects</a>,
          <a className="about__rkmscc-link" href="blog"> my blog</a> and my <a className="about__rkmscc-link" href="https://github.com/zuhairabs">github</a> for more details.
          <br />
          <br />
          Whether you're having an idea or a business inquiry, do not hesitate to drop a message below ⬇️


        </p>
      </AboutInfo>
    </AboutWrapper>
  )
}

export default About;
