import {themes as prismThemes} from 'prism-react-renderer';
import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

const config: Config = {
  title: 'Reinforce Tactics',
  tagline: 'Turned based strategy game with the goal of developing reinforcement learning algorithms',
  favicon: 'img/favicon.ico',

  // Future flags, see https://docusaurus.io/docs/api/docusaurus-config#future
  future: {
    v4: true, // Improve compatibility with the upcoming Docusaurus v4
  },

  // Set the production url of your site here
  url: 'https://kuds.github.io',
  // Set the /<baseUrl>/ pathname under which your site is served
  // For GitHub pages deployment, it is often '/<projectName>/'
  baseUrl: '/reinforce-tactics/',

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
        // IMPORTANT: Replace 'G-XXXXXXXXXX' with your actual Google Analytics tracking ID
        // You can find this in your Google Analytics account under Admin > Data Streams
        gtag: {
          trackingID: 'G-XXXXXXXXXX', // Replace with your actual tracking ID
          anonymizeIP: true,
        },
      } satisfies Preset.Options,
    ],
  ],

  themeConfig: {
    // Replace with your project's social card
    image: 'img/docusaurus-social-card.jpg',
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
            {
              label: 'Project Timeline',
              to: '/docs/timeline',
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
