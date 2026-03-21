import { type ReactNode, useState } from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';
import Head from '@docusaurus/Head';

import styles from './index.module.css';

// ===== Featured Units (trimmed to 4 for landing page) =====
const featuredUnits = [
  {
    name: 'Warrior',
    role: 'Frontline Fighter',
    gif: '/img/units/warrior_idle.gif',
    color: '#8b2942',
  },
  {
    name: 'Mage',
    role: 'Arcane Striker',
    gif: '/img/units/mage_idle.gif',
    color: '#4a148c',
  },
  {
    name: 'Knight',
    role: 'Heavy Cavalry',
    gif: '/img/units/knight_idle.gif',
    color: '#5c5c5c',
  },
  {
    name: 'Archer',
    role: 'Ranged Specialist',
    gif: '/img/units/archer_idle.gif',
    color: '#1e3a5f',
  },
  {
    name: 'Rogue',
    role: 'Stealth Assassin',
    gif: '/img/units/rogue_idle.gif',
    color: '#2d2d2d',
  },
  {
    name: 'Cleric',
    role: 'Support Healer',
    gif: '/img/units/cleric_idle.gif',
    color: '#2d5a3d',
  },
];

// ===== Map previews =====
const mapPreviews = [
  { src: '/img/maps/crossroads.png', name: 'Crossroads' },
  { src: '/img/maps/island_fortress.png', name: 'Island Fortress' },
  { src: '/img/maps/tower_rush.png', name: 'Tower Rush' },
  { src: '/img/maps/center_mountains.png', name: 'Center Mountains' },
];

// ===== SEO Structured Data =====
const structuredData = {
  '@context': 'https://schema.org',
  '@type': 'SoftwareApplication',
  name: 'Reinforce Tactics',
  applicationCategory: 'GameApplication',
  operatingSystem: 'Cross-platform',
  description: 'A turn-based strategy game environment for reinforcement learning research and LLM evaluation. Benchmark AI models on tactical reasoning and strategic planning.',
  offers: {
    '@type': 'Offer',
    price: '0',
    priceCurrency: 'USD',
  },
  author: {
    '@type': 'Organization',
    name: 'Reinforce Tactics',
    url: 'https://reinforcetactics.com',
  },
  keywords: 'reinforcement learning, LLM evaluation, AI benchmark, turn-based strategy, tactical AI, GPT evaluation, Claude benchmark, Gemini testing, game AI, Gymnasium environment',
};

// ===== Hero Section =====
function HeroSection(): ReactNode {
  return (
    <header className={styles.heroBanner}>
      <div className={styles.heroGrid}>
        <div className={styles.heroText}>
          <span className={styles.heroBadge}>Open Source</span>
          <Heading as="h1" className={styles.heroTitle}>
            Tactical AI<br />
            <span className={styles.heroTitleAccent}>Benchmark</span>
          </Heading>
          <p className={styles.heroSubtitle}>
            Evaluate LLMs and train RL agents on turn-based tactical combat.
            Compare models head-to-head in competitive tournaments.
          </p>
          <div className={styles.heroButtons}>
            <Link
              className={styles.btnPrimary}
              to="/docs/">
              Get Started
            </Link>
            <Link
              className={styles.btnSecondary}
              to="https://github.com/kuds/reinforce-tactics">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor"><path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/></svg>
              GitHub
            </Link>
          </div>
        </div>
        <div className={styles.heroTerminal}>
          <div className={styles.terminalBar}>
            <span className={styles.terminalDot} style={{background: '#ff5f56'}} />
            <span className={styles.terminalDot} style={{background: '#ffbd2e'}} />
            <span className={styles.terminalDot} style={{background: '#27c93f'}} />
            <span className={styles.terminalTitle}>terminal</span>
          </div>
          <div className={styles.terminalBody}>
            <code>
              <span className={styles.terminalPrompt}>$</span>{' '}
              <span className={styles.terminalCmd}>pip install</span> reinforcetactics
              {'\n\n'}
              <span className={styles.terminalPrompt}>$</span>{' '}
              <span className={styles.terminalCmd}>python</span> -m reinforcetactics tournament \{'\n'}
              {'  '}--agents gpt-4o claude-sonnet gemini-pro \{'\n'}
              {'  '}--rounds 50 \{'\n'}
              {'  '}--map crossroads
              {'\n\n'}
              <span className={styles.terminalOutput}>
                {'Tournament started: 3 agents, 50 rounds\n'}
                {'Round 50/50 complete\n'}
                {'─────────────────────────────\n'}
                {'  #1  claude-sonnet   ELO 1847\n'}
                {'  #2  gpt-4o          ELO 1723\n'}
                {'  #3  gemini-pro      ELO 1630\n'}
              </span>
            </code>
          </div>
        </div>
      </div>
    </header>
  );
}

// ===== Value Props Section =====
function ValuePropsSection(): ReactNode {
  return (
    <section className={styles.valueProps}>
      <div className="container">
        <div className={styles.valueGrid}>
          <div className={styles.valueCard}>
            <div className={styles.valueIcon}>
              <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>
            </div>
            <h3>LLM Evaluation</h3>
            <p>Benchmark GPT, Claude, Gemini, and custom models on strategic reasoning, spatial awareness, and multi-step planning.</p>
          </div>
          <div className={styles.valueCard}>
            <div className={styles.valueIcon}>
              <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="10"/><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>
            </div>
            <h3>RL Training</h3>
            <p>Full Gymnasium environment with multi-discrete action space, configurable reward shaping, and headless mode for fast training.</p>
          </div>
          <div className={styles.valueCard}>
            <div className={styles.valueIcon}>
              <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M6 9H4.5a2.5 2.5 0 0 1 0-5C7 4 9 7 12 7s5-3 7.5-3a2.5 2.5 0 0 1 0 5H18"/><path d="M18 15h1.5a2.5 2.5 0 0 0 0-5H18"/><path d="M6 15H4.5a2.5 2.5 0 0 1 0-5H6"/><line x1="6" y1="9" x2="6" y2="15"/><line x1="18" y1="9" x2="18" y2="15"/><path d="M6 15a6 6 0 0 0 12 0"/></svg>
            </div>
            <h3>Tournaments</h3>
            <p>Automated round-robin tournaments with ELO ratings, replay recording, and detailed performance analytics.</p>
          </div>
        </div>
      </div>
    </section>
  );
}

// ===== Map Gallery Section =====
function MapGallerySection(): ReactNode {
  return (
    <section className={styles.mapSection}>
      <div className="container">
        <Heading as="h2" className={styles.sectionTitle}>
          Diverse Tactical Maps
        </Heading>
        <p className={styles.sectionSubtitle}>
          14 hand-crafted maps across 1v1, 1v1v1, and 2v2 formats with varied terrain, chokepoints, and strategic objectives.
        </p>
        <div className={styles.mapGrid}>
          {mapPreviews.map((map) => (
            <div key={map.name} className={styles.mapCard}>
              <img src={map.src} alt={map.name} loading="lazy" />
              <span className={styles.mapLabel}>{map.name}</span>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

// ===== Units Strip Section =====
function UnitsStripSection(): ReactNode {
  return (
    <section className={styles.unitsSection}>
      <div className="container">
        <Heading as="h2" className={styles.sectionTitle}>
          8 Unique Unit Types
        </Heading>
        <p className={styles.sectionSubtitle}>
          Each unit has distinct stats, abilities, and roles — creating a rich decision space
          for AI agents to master.
        </p>
        <div className={styles.unitsStrip}>
          {featuredUnits.map((unit) => (
            <div key={unit.name} className={styles.unitChip}>
              <div
                className={styles.unitChipIcon}
                style={{ borderColor: unit.color }}
              >
                <img
                  src={unit.gif}
                  alt={unit.name}
                  className={styles.unitGif}
                  loading="lazy"
                />
              </div>
              <span className={styles.unitChipName}>{unit.name}</span>
              <span className={styles.unitChipRole}>{unit.role}</span>
            </div>
          ))}
        </div>
        <div className={styles.unitsCta}>
          <Link to="/docs/game-mechanics" className={styles.linkArrow}>
            View all units and mechanics &rarr;
          </Link>
        </div>
      </div>
    </section>
  );
}

// ===== How It Works Section =====
function HowItWorksSection(): ReactNode {
  return (
    <section className={styles.howSection}>
      <div className="container">
        <Heading as="h2" className={styles.sectionTitle}>
          How It Works
        </Heading>
        <div className={styles.stepsGrid}>
          <div className={styles.step}>
            <div className={styles.stepNumber}>1</div>
            <h3>Install</h3>
            <p>Install via pip with optional GPU, GUI, and LLM extras. Works on Python 3.10+.</p>
            <code className={styles.stepCode}>pip install reinforcetactics[llm]</code>
          </div>
          <div className={styles.step}>
            <div className={styles.stepNumber}>2</div>
            <h3>Configure</h3>
            <p>Pick your agents — LLM bots, RL models, rule-based bots, or your own custom agent.</p>
            <code className={styles.stepCode}>--agents gpt-4o claude-sonnet</code>
          </div>
          <div className={styles.step}>
            <div className={styles.stepNumber}>3</div>
            <h3>Compete</h3>
            <p>Run tournaments, compare ELO ratings, analyze replays, and iterate on your models.</p>
            <code className={styles.stepCode}>python -m reinforcetactics tournament</code>
          </div>
        </div>
      </div>
    </section>
  );
}

// ===== Features Grid (compact) =====
function FeaturesSection(): ReactNode {
  return (
    <section className={styles.featuresSection}>
      <div className="container">
        <Heading as="h2" className={styles.sectionTitle}>
          Built for AI Research
        </Heading>
        <div className={styles.featuresGrid}>
          <div className={styles.featureItem}>
            <h4>Gymnasium Compatible</h4>
            <p>Standard RL interface with observation and action spaces, reward shaping, and episode management.</p>
          </div>
          <div className={styles.featureItem}>
            <h4>Multi-Agent Support</h4>
            <p>PettingZoo integration for multi-agent RL. Train cooperative and competitive policies.</p>
          </div>
          <div className={styles.featureItem}>
            <h4>Replay & Analysis</h4>
            <p>Record battles, export to video, and analyze decision patterns for model interpretability.</p>
          </div>
          <div className={styles.featureItem}>
            <h4>Extensible Architecture</h4>
            <p>Add custom units, maps, reward functions, and AI agents with a clean Python API.</p>
          </div>
          <div className={styles.featureItem}>
            <h4>Multiple AI Backends</h4>
            <p>OpenAI, Anthropic, and Google Gemini SDKs built-in. Plug in any LLM via API.</p>
          </div>
          <div className={styles.featureItem}>
            <h4>Docker Tournaments</h4>
            <p>Containerized tournament runner for reproducible benchmarks at scale.</p>
          </div>
        </div>
      </div>
    </section>
  );
}

// ===== CTA Section =====
function CTASection(): ReactNode {
  return (
    <section className={styles.ctaSection}>
      <div className={styles.ctaContent}>
        <Heading as="h2" className={styles.ctaTitle}>
          Ready to benchmark your AI?
        </Heading>
        <p className={styles.ctaText}>
          Open source and ready for research. Clone the repo and run your first tournament in minutes.
        </p>
        <div className={styles.ctaButtons}>
          <Link className={styles.btnPrimary} to="/docs/">
            Read the Docs
          </Link>
          <Link className={styles.btnSecondary} to="https://github.com/kuds/reinforce-tactics">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor"><path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/></svg>
            Star on GitHub
          </Link>
        </div>
      </div>
    </section>
  );
}

// ===== Main Page Component =====
export default function Home(): ReactNode {
  const { siteConfig } = useDocusaurusContext();
  return (
    <Layout
      title="LLM Evaluation & Reinforcement Learning Benchmark"
      description="Evaluate GPT, Claude, Gemini and other LLMs on strategic reasoning. Open-source turn-based strategy environment for AI research, reinforcement learning, and model benchmarking.">
      <Head>
        <meta name="keywords" content="LLM evaluation, LLM benchmark, GPT evaluation, Claude benchmark, Gemini testing, reinforcement learning, AI benchmark, strategic reasoning, tactical AI, game AI, Gymnasium environment, turn-based strategy, AI research, machine learning" />
        <meta name="author" content="Reinforce Tactics" />
        <meta property="og:type" content="website" />
        <meta property="og:site_name" content="Reinforce Tactics" />
        <meta name="twitter:card" content="summary_large_image" />
        <link rel="canonical" href="https://reinforcetactics.com" />
        <script type="application/ld+json">
          {JSON.stringify(structuredData)}
        </script>
      </Head>
      <HeroSection />
      <main>
        <ValuePropsSection />
        <MapGallerySection />
        <UnitsStripSection />
        <HowItWorksSection />
        <FeaturesSection />
        <CTASection />
      </main>
    </Layout>
  );
}
