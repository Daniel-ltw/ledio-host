---
title: Ledio Host
emoji: 🏢
colorFrom: gray
colorTo: yellow
sdk: docker
pinned: false
---

## Ledio Host

Have been listening to Kugou music for a while and have used their AI radio host feature, which would introduce the music that will be playing from your playlist just when the song is about to start.

Have also been seeing how nice [ChatTTS](https://chattts.me/) works and was keen to give it a try.

Project will try to replicate the AI radio host feature with [ChatTTS](https://chattts.me/).

### Issues with ChatTTS

Started off by following much of what we did with the podcast script and recording challenge. But when getting ChatTTS to run on my machine, I seem to be encountering an issue with my machines limitation. My M1 chipset machine is not able to run the complex pytorch related code.

So from the above, I have gone back into using xTTS V2.

### Issues with running huge model locally

I have tried to run more complex/large model on my M1 chipset machine, but my machine seems to be hitting its limit and has issues.

With the above problem, will need to work with the smallest models, but will also need to ensure that we prevent the model from hallucinating.

###
