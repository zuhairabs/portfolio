import React from "react";
import styled from 'styled-components';

import PageHeader from 'src/components/common/PageHeader';
import { Hidden } from "react-grid-system";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome"
import { faPython, faDocker, faAws } from '@fortawesome/free-brands-svg-icons'


import Flex from 'src/components/common/Flex'

import { Parallax, ParallaxProvider } from 'react-scroll-parallax';

const SkillsWrapper = styled.section`
  ${props => props.theme.spacing.sectionBottom};
  position: relative;
  .skill__icons {
    padding: 30px 0;
    @media ${props => props.theme.media.tablet} {
      padding: 10px 0;
    }

    svg {
      color: ${props => props.theme.primaryColor};
    }
  }
  .skills__word-clouds {
    @media ${props => props.theme.media.tablet} {
      display: none;
    }

    p {
      position: absolute;
      color: ${props => props.theme.accentColor};
      z-index: -1;
      left: 0;
      right: 0;
      font-weight: 900;
    }
    z-index: -1;
  }
`

const WordClouds = () => {
  return (
    <Parallax
      y={["-350px", "-150px"]}
      slowerScrollRate={true}
      aria-hidden="true"
      className="skills__word-clouds">
      <p style={{ top: 50, left: '100%', fontSize: 23 }}>PyTorch</p>
      <p style={{ top: 0, left: 0, fontSize: 25 }}>Tensorflow</p>
      <p style={{ top: 200, left: -60, fontSize: 14 }}>Dash</p>
      <p style={{ top: '30%', left: '25%', fontSize: 18 }}>Python</p>
      <p style={{ top: 40, left: '75%', fontSize: 12 }}>Docker</p>
      <p style={{ top: 380, left: '85%', fontSize: 18 }}>MongoDB</p>
      <p style={{ top: 320, left: '65%', fontSize: 18 }}>ElasticSearch</p>
      <p style={{ top: 350, left: 150, fontSize: 20 }}>Kibana</p>
      <p style={{ top: 120, left: '22%', fontSize: 12 }}>Amazon Web Services</p>
    </Parallax>
  )
}

const Skills = () => (
  <SkillsWrapper>
    <ParallaxProvider>

      <PageHeader>My Skillsets</PageHeader>

      {/* <SkillPie percent={90} name={"JavaScript"} /> */}
      <Flex className="skill__icons" justify="space-around" align="center">
        <FontAwesomeIcon icon={faPython} size="5x" />
        <FontAwesomeIcon icon={faAws} size="5x" />
        <FontAwesomeIcon icon={["fab", "html5"]} size="5x" />
        <FontAwesomeIcon icon={["fab", "js"]} size="5x" />
        <FontAwesomeIcon icon={faDocker} size="5x" />
      </Flex>

      <Hidden md xs sm>
        <WordClouds />
      </Hidden>
    </ParallaxProvider>

  </SkillsWrapper>
)

export default Skills;