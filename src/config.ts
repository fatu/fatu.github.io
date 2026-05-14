export const SITE = {
  website: "https://fatu.github.io/",
  author: "Fangbo Tu",
  profile: "https://github.com/fatu",
  desc: "Notes on numerics, quantization, and ML systems.",
  title: "Fangbo Tu",
  ogImage: "astropaper-og.jpg",
  lightAndDarkMode: true,
  postPerIndex: 4,
  postPerPage: 4,
  scheduledPostMargin: 15 * 60 * 1000, // 15 minutes
  showArchives: true,
  showBackButton: true,
  editPost: {
    enabled: false,
    text: "Edit page",
    url: "https://github.com/fatu/fatu.github.io/edit/main/",
  },
  dynamicOgImage: true,
  dir: "ltr",
  lang: "en",
  timezone: "America/Los_Angeles",
} as const;
