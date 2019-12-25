import React from "react";
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { faYoutube, faQuora, faLinkedinIn, faStackOverflow} from '@fortawesome/free-brands-svg-icons'
import { faRobot, faRocket } from '@fortawesome/free-solid-svg-icons'


import svgRect from 'src/static/home_rect.svg'

import { HeroCard } from './HeroCard';
import { HomeWrapper, Intro } from './Home.style';

import IconLink from 'src/components/common/IconLink';
import PageHeader from 'src/components/common/PageHeader';
import Flex from "src/components/common/Flex";
import Button from "src/components/common/Button";

import { Card, CardIcon, CardText, CardTitle } from "src/components/common/Card";

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
        I push apps to production. Docker is my one of my friends
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
          <h1>Ahmed BESBES</h1>
          <p className="adjust">AI Engineer // Blogger // Runner</p>

          <div className="home__CTA">
            <Button className="cta" as="a" href="#">Download Resume</Button>

            <div className="home__social">
              <IconLink label="github" icon={["fab", "github"]} href="//github.com/ahmedbesbes" />
              <IconLink label="linkedin" icon={faLinkedinIn} href="//linkedin.com/in/ahmed-besbes-99a91661/" />
              <IconLink label="twitter" icon={["fab", "twitter"]} href="//twitter.com/ahmed_besbes_" />
              <IconLink label="stack-overflow" icon={faStackOverflow} href="//stackoverflow.com/users/4583959/ahmed-besbess" />
              <IconLink label="youtube" icon={faYoutube} href="//youtube.com/channel/UCP1M7FpkpNljH4r6ORiRg6g?view_as=subscriber" />
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