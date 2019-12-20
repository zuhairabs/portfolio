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
          <Avatar src="ahmed_avatar.jpg" />
        </div>
        <p>
          Hello there! I'm Ahmed. I hope you stumbled upon this blog on purpose! Otherwise, let me introduce you to my world ! 
          <br />
          <br />
          I'm a data scientist who builds and deploys AI products.
          <br />
          <br />
          I live in France 🇫🇷, I've been working accross many industries (financial services, media, public sector) and I'm also a blogger and a Youtuber enthusiast.
          <br />
          <br />
          I do open source work that you can check out on <a className="about__rkmscc-link" href="https://github.com/ahmedbesbes">github</a> .
          <br />
          <br />
          Do not hesitate to contact me for any request.


        </p>
      </AboutInfo>
    </AboutWrapper>
  )
}

export default About;