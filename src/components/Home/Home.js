import React from "react";
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { faYoutube, faQuora, faLinkedinIn, faStackOverflow, faMedium} from '@fortawesome/free-brands-svg-icons'
import { faRobot, faRocket } from '@fortawesome/free-solid-svg-icons'


import svgRect from 'src/static/home_rect.svg'

import { HeroCard } from './HeroCard';
import { HomeWrapper, Intro } from './Home.style';

import IconLink from 'src/components/common/IconLink';
import PageHeader from 'src/components/common/PageHeader';
import Flex from "src/components/common/Flex";
import Button from "src/components/common/Button";

import { Card, CardIcon, CardText, CardTitle } from "src/components/common/Card";
import cv from "src/static/pdfs/cv.pdf"

const ThingsILove = () => (
  <Flex justify="space-between" align="center">
    <Card>
      <CardIcon><FontAwesomeIcon icon={faRobot} /></CardIcon>
      <CardTitle>Machine Learning</CardTitle>
      <CardText>
        I train robust models for various tasks in NLP, computer vision and more
      </CardText>
    </Card>

    <Card>
      <CardIcon><FontAwesomeIcon icon="code" /></CardIcon>
      <CardTitle>Software Engineering</CardTitle>
      <CardText>
        I build apps to encapsulate ML models and provide a better user experience
      </CardText>
    </Card>

    <Card>
      <CardIcon><FontAwesomeIcon icon={faRocket}/></CardIcon>
      <CardTitle>Deployment</CardTitle>
      <CardText>
        I go beyond scripts and notebooks and deploy apps to production
      </CardText>
    </Card>
  </Flex>
);

const Home = () => {
  return (
    <HomeWrapper id="home">
      <img className="svg-rect" src={svgRect} alt=""></img>

      <Intro>
        {/* <Parallax y={[50, -50]} className="home__text"> */}
        <div className="home__text">
          <p>Hello, i’m</p>
          <h1>Zuhair Abbas</h1>
          <p className="adjust">AI Engineer // Data Scientist // Reader</p>

          <div className="home__CTA">
            <Button className="cta" as="a" href={cv}>Download Resume</Button>

            <div className="home__social">
              <IconLink label="github" icon={["fab", "github"]} href="//github.com/zuhairabs" />
              <IconLink label="linkedin" icon={faLinkedinIn} href="//linkedin.com/in/zuhairabs/" />
              <IconLink label="twitter" icon={["fab", "twitter"]} href="//twitter.com/zuhairabs" />
              <IconLink label="medium" icon={faMedium} href="//medium.com/@zuhairabs" />
              <IconLink label="youtube" icon={faYoutube} href="//youtube.com/channel/UCE2zC9B8zJZCywWfxcxrvrw?view_as=subscriber" />
            </div>
          </div>
        </div>
        {/* </Parallax> */}
        <HeroCard />
      </Intro>

      {/* Things I LOVE */}
      <PageHeader style={{ marginBottom: 30 }}>Things i do <i className="fas fa-heart" /></PageHeader>
      <ThingsILove />

    </HomeWrapper>
  )
}

export default Home;
