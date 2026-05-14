# fatu.github.io

Personal blog — notes on numerics, quantization, and ML systems.

Built with [Astro](https://astro.build) and the [AstroPaper](https://github.com/satnaing/astro-paper) theme. Deployed via GitHub Pages.

## Local dev

```bash
npm install
npm run dev
```

Open http://localhost:4321

## Publish a post

Add a markdown file under `src/data/blog/`. Frontmatter required: `title`, `pubDatetime`, `description`. Set `draft: true` to keep it unlisted.

## Deploy

Push to `main`. GitHub Actions builds and deploys to https://fatu.github.io
