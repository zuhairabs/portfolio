import React, { useRef, useEffect, useState } from 'react';
import { withTheme } from 'styled-components';

import {
  HeroCardWrapper,
  CodeCardWrapper,
  ColorPaletteWrapper,
  ColorBoxWrapper,
} from './HeroCard.style'


function repeatString(str, count) {
  let maxCount = str.length * count;
  count = Math.floor(Math.log(count) / Math.log(2));
  while (count) {
    str += str;
    count--;
  }
  str += str.substring(0, maxCount - str.length);
  return str;
}

function copyToClipboard(str) {
  var el = document.createElement('textarea');
  el.value = str;
  el.setAttribute('readonly', '');
  el.style = { position: 'absolute', left: '-9999px' };
  document.body.appendChild(el);
  el.select();
  document.execCommand('copy');
  document.body.removeChild(el);
}

export const CodeCard = () => {
  const [text, setText] = useState(`new Date().getFullYear() - 2001;`);
  const age = new Date().getFullYear() - 2001;
  // trimed down polyfill of String.repeat
  const changeText = () => {
    let space = repeatString(' ', 54);
    setText(age + ';' + space)
  }
  return (
    <CodeCardWrapper>
      <pre>1   class <b>Person</b> {'{'}</pre>
      <pre>2         constructor() {'{'}</pre>
      <pre>3             this.name = "<b>Zuhair Abbas</b>";</pre>
      <pre>4             this.skills = ["<b>Machine Learning</b>", "<b>Data Science</b>", 
                                        "<b>Developer</b>"];</pre>
      <pre>6         {"}"}</pre>
      <pre>7   {"}"}</pre>
    </CodeCardWrapper>
  )
}


const ColorBox = ({ color }) => {
  const tooltipRef = useRef();
  useEffect(() => {
    return tooltipRef.current.addEventListener('animationend', () => {
      tooltipRef.current.classList.remove('tooltip-animate')
    });
  })
  const copy = () => {
    copyToClipboard(color);
    tooltipRef.current.classList.add('tooltip-animate');
  };

  return <ColorBoxWrapper ref={tooltipRef} onClick={copy} style={{ background: color }} />
}

export const ColorPalette = withTheme(({ theme }) => {
  return (
    <ColorPaletteWrapper>
      <ColorBox color={theme.primaryColor} />
      <ColorBox color={'#6A98F0'} />
      <ColorBox color={theme.gradient} />
      <ColorBox color={theme.primaryBlack} />
      <ColorBox color={theme.accentColor} />
    </ColorPaletteWrapper>
  )
})

export const HeroCard = () => {
  return (
    <HeroCardWrapper>
      <CodeCard />
      <ColorPalette />
    </HeroCardWrapper>
  )
}
