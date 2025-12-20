import {themes as prismThemes} from 'prism-react-renderer';
import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

const config: Config = {
  title: 'Reinforce Tactics',
  tagline: 'Evaluate LLMs and train RL agents on strategic reasoning. Benchmark GPT-4, Claude, and Gemini.',
  favicon: 'img/favicon.ico',

  // SEO metadata
  headTags: [
    {
      tagName: 'meta',
      attributes: {
        name: 'keywords',
        content: 'LLM evaluation, LLM benchmark, GPT-4 evaluation, Claude benchmark, Gemini testing, reinforcement learning, AI benchmark, strategic reasoning, tactical AI, game AI, Gymnasium environment, turn-based strategy, AI research, machine learning, model evaluation',
      },
    },
    {
      tagName: 'meta',
      attributes: {
        name: 'author',
        content: 'Reinforce Tactics',
      },
    },
    {
      tagName: 'meta',
      attributes: {
        property: 'og:type',
        content: 'website',
      },
    },
    {
      tagName: 'link',
      attributes: {
        rel: 'canonical',
        href: 'https://reinforcetactics.com',
      },
    },
  ],

  // Future flags, see https://docusaurus.io/docs/api/docusaurus-config#future
  future: {
    v4: true, // Improve compatibility with the upcoming Docusaurus v4
  },

  // Set the production url of your site here
  url: 'https://reinforcetactics.com',
  // Set the /<baseUrl>/ pathname under which your site is served
  // For custom domains, this should be '/'
  baseUrl: '/',

  // GitHub pages deployment config.
  // If you aren't using GitHub pages, you don't need these.
  organizationName: 'kuds', // Usually your GitHub org/user name.
  projectName: 'reinforce-tactics', // Usually your repo name.

  onBrokenLinks: 'throw',

  // Even if you don't use internationalization, you can use this field to set
  // useful metadata like html lang. For example, if your site is Chinese, you
  // may want to replace "en" with "zh-Hans".
  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      {
        docs: {
          sidebarPath: './sidebars.ts',
          // Please change this to your repo.
          // Remove this to remove the "edit this page" links.
          editUrl:
            'https://github.com/kuds/reinforce-tactics/tree/main/docs-site/',
        },
        blog: false, // Disable blog for this project
        theme: {
          customCss: './src/css/custom.css',
        },
        // Configure Google Analytics plugin
        // IMPORTANT: Replace 'G-XXXXXXXXXX' with your actual Google Analytics 4 (GA4) Measurement ID
        // 
        // To get your tracking ID:
        // 1. Go to Google Analytics (https://analytics.google.com/)
        // 2. Navigate to Admin > Data Streams
        // 3. Select your web stream
        // 4. Copy the Measurement ID (format: G-XXXXXXXXXX)
        // 
        // Note: If you don't replace this placeholder, analytics tracking will not work,
        // but the site will still function normally.
        gtag: {
          trackingID: 'G-7ETBJXYY4L', // Replace with your actual GA4 Measurement ID
          anonymizeIP: true,
        },
      } satisfies Preset.Options,
    ],
  ],

  themeConfig: {
    // Social card and SEO metadata
    image: 'img/docusaurus-social-card.jpg',
    metadata: [
      {
        name: 'description',
        content: 'Evaluate GPT-4, Claude, Gemini and other LLMs on strategic reasoning. Open-source turn-based strategy environment for AI research, reinforcement learning, and model benchmarking.',
      },
      {
        property: 'og:description',
        content: 'Evaluate GPT-4, Claude, Gemini and other LLMs on strategic reasoning. Open-source turn-based strategy environment for AI research and model benchmarking.',
      },
      {
        name: 'twitter:description',
        content: 'Evaluate GPT-4, Claude, Gemini and other LLMs on strategic reasoning. Open-source RL environment for AI research.',
      },
    ],
    colorMode: {
      respectPrefersColorScheme: true,
    },
    navbar: {
      title: 'Reinforce Tactics',
      logo: {
        alt: 'Reinforce Tactics Logo',
        src: 'img/logo.svg',
      },
      items: [
        {
          type: 'docSidebar',
          sidebarId: 'tutorialSidebar',
          position: 'left',
          label: 'Documentation',
        },
        {
          href: 'https://github.com/kuds/reinforce-tactics',
          label: 'GitHub',
          position: 'right',
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Documentation',
          items: [
            {
              label: 'Getting Started',
              to: '/docs/',
            },
            {
              label: 'Implementation Status',
              to: '/docs/implementation-status',
            },
          ],
        },
        {
          title: 'Resources',
          items: [
            {
              label: 'GitHub Repository',
              href: 'https://github.com/kuds/reinforce-tactics',
            },
            {
              label: 'Issues',
              href: 'https://github.com/kuds/reinforce-tactics/issues',
            },
            {
              label: 'Pull Requests',
              href: 'https://github.com/kuds/reinforce-tactics/pulls',
            },
          ],
        },
        {
          title: 'More',
          items: [
            {
              label: 'Finding Theta',
              href: 'https://findingtheta.com',
            },
            {
              label: 'Main README',
              href: 'https://github.com/kuds/reinforce-tactics#readme',
            },
            {
              label: 'License (MIT)',
              href: 'https://github.com/kuds/reinforce-tactics/blob/main/LICENSE',
            },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} Reinforce Tactics. Built with Docusaurus.`,
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
    },
  } satisfies Preset.ThemeConfig,
};

export default config;
