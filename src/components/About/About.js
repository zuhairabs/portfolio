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
          <Avatar src="ahmed.jpg" />
        </div>
        <p>
          Hello there! I'm Ahmed. I hope you stumbled upon this website on purpose! Otherwise, let me introduce you to my world! 
          <br />
          I'm a data scientist living in France 🇫🇷. I've been working accross many industries such financial services, media and public sector.
          <br />
          Part of my work include crafting, building and deploying AI applications to answer business issues.
          
          I also blog about technical topics such as deep learning. You can check <a className="about__rkmscc-link" href="blog"> my blog</a> or 
          my <a className="about__rkmscc-link" href="https://github.com/ahmedbesbes">github</a> for more details.
          <br />
          You're having an idea or a business inquiry, do not hesitate to drop a message below ⬇️


        </p>
      </AboutInfo>
    </AboutWrapper>
  )
}

export default About;