import type { ReactNode } from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';
import Head from '@docusaurus/Head';

import styles from './index.module.css';

// ===== Unit Data =====
const units = [
  {
    name: 'Warrior',
    code: 'W',
    role: 'Frontline Fighter',
    description: 'Stalwart defenders who excel in close combat. High durability makes them perfect for holding the line.',
    stats: { hp: 15, attack: 10, defense: 6, movement: 3 },
    cardClass: styles.unitCardWarrior,
    iconClass: styles.unitIconWarrior,
  },
  {
    name: 'Mage',
    code: 'M',
    role: 'Arcane Striker',
    description: 'Masters of mystical arts who can strike from afar and paralyze enemies for 3 turns.',
    stats: { hp: 10, attack: 12, defense: 4, movement: 2 },
    cardClass: styles.unitCardMage,
    iconClass: styles.unitIconMage,
  },
  {
    name: 'Cleric',
    code: 'C',
    role: 'Support Healer',
    description: 'Devoted healers who restore allies and cure status effects. Essential for sustained campaigns.',
    stats: { hp: 8, attack: 2, defense: 4, movement: 2 },
    cardClass: styles.unitCardCleric,
    iconClass: styles.unitIconCleric,
  },
  {
    name: 'Archer',
    code: 'A',
    role: 'Ranged Specialist',
    description: 'Precise marksmen with extended range from high ground. Enemies cannot counter-attack.',
    stats: { hp: 15, attack: 5, defense: 1, movement: 3 },
    cardClass: styles.unitCardArcher,
    iconClass: styles.unitIconArcher,
  },
];

// ===== Feature Data =====
const features = [
  {
    icon: '🎮',
    title: 'Turn-Based Tactical Combat',
    description: 'Strategic grid-based battles with attacks, counter-attacks, paralysis, and healing mechanics inspired by Fire Emblem and Advance Wars.',
  },
  {
    icon: '🤖',
    title: 'Gymnasium RL Environment',
    description: 'Full Gymnasium compatibility with multi-discrete action space, configurable reward shaping, and headless mode for high-speed training.',
  },
  {
    icon: '🧠',
    title: 'LLM Evaluation Framework',
    description: 'Benchmark GPT-4, Claude, Gemini, and other large language models on strategic reasoning, planning, and multi-step decision making.',
  },
  {
    icon: '🏆',
    title: 'Tournament System',
    description: 'Run automated tournaments between AI agents, track ELO ratings, and generate detailed performance analytics and leaderboards.',
  },
  {
    icon: '📊',
    title: 'Replay & Analysis Tools',
    description: 'Record battles, export replays to video, and analyze decision patterns. Essential for AI research and model interpretability.',
  },
  {
    icon: '🔧',
    title: 'Modular Architecture',
    description: 'Clean, extensible Python codebase for adding new units, mechanics, reward functions, and custom AI agents.',
  },
];

// ===== LLM Models Data =====
const llmModels = [
  {
    name: 'OpenAI GPT-5',
    icon: '🟢',
    description: 'Evaluate GPT-5 and GPT-5 Mini on complex tactical scenarios requiring multi-step planning.',
  },
  {
    name: 'Anthropic Claude',
    icon: '🟣',
    description: 'Benchmark Claude 4.5 Sonnet, Claude 4.5 Opus, and Claude Haiku 4.5 on strategic reasoning tasks.',
  },
  {
    name: 'Google Gemini',
    icon: '🔵',
    description: 'Test Gemini Pro and Gemini Ultra on spatial reasoning and resource management.',
  },
  {
    name: 'Custom Models',
    icon: '⚪',
    description: 'Integrate any LLM via API or local inference for comparative evaluation.',
  },
];

// ===== Documentation Links =====
const docLinks = [
  {
    icon: '📖',
    title: 'Getting Started',
    description: 'Installation, setup, and quick start guide',
    href: '/docs/',
  },
  {
    icon: '⚔️',
    title: 'Game Mechanics',
    description: 'Units, combat system, and structures',
    href: '/docs/game-mechanics',
  },
  {
    icon: '🏆',
    title: 'Tournament System',
    description: 'Run AI tournaments and track ELO ratings',
    href: '/docs/tournament-system',
  },
  {
    icon: '📋',
    title: 'Implementation Status',
    description: 'Features and development roadmap',
    href: '/docs/implementation-status',
  },
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
  keywords: 'reinforcement learning, LLM evaluation, AI benchmark, turn-based strategy, tactical AI, GPT-5 evaluation, Claude benchmark, Gemini testing, game AI, Gymnasium environment',
};

// ===== Hero Section =====
function HeroSection(): ReactNode {
  return (
    <header className={styles.heroBanner}>
      <div className={styles.heroContent}>
        <span className={styles.heroBadge}>LLM Evaluation + Reinforcement Learning</span>
        <Heading as="h1" className={styles.heroTitle}>
          Reinforce<span className={styles.heroTitleAccent}>Tactics</span>
        </Heading>
        <p className={styles.heroSubtitle}>
          The open-source tactical strategy environment for evaluating large language models
          and training reinforcement learning agents. Benchmark GPT-5, Claude, Gemini, and custom
          AI on strategic reasoning, multi-step planning, and resource management.
        </p>
        <div className={styles.buttons}>
          <Link
            className="button button--lg button--tactics-primary"
            to="/docs/">
            Get Started
          </Link>
          <Link
            className="button button--lg button--tactics-secondary"
            to="https://github.com/kuds/reinforce-tactics">
            View on GitHub
          </Link>
        </div>
      </div>
    </header>
  );
}

// ===== LLM Evaluation Section =====
function LLMSection(): ReactNode {
  return (
    <section className={styles.llmSection}>
      <div className="container">
        <Heading as="h2" className={styles.sectionTitle}>
          Evaluate Large Language Models
        </Heading>
        <p className={styles.sectionSubtitle}>
          Reinforce Tactics provides a rigorous benchmark for testing LLM capabilities in
          strategic reasoning, spatial awareness, and long-horizon planning. Compare models
          head-to-head in competitive tournaments.
        </p>
        <div className={styles.llmGrid}>
          {llmModels.map((model) => (
            <div key={model.name} className={styles.llmCard}>
              <span className={styles.llmIcon}>{model.icon}</span>
              <h3>{model.name}</h3>
              <p>{model.description}</p>
            </div>
          ))}
        </div>
        <div className={styles.llmCta}>
          <p className={styles.llmCtaText}>
            Run automated tournaments, generate ELO ratings, and analyze decision-making patterns
            across different model architectures and prompting strategies.
          </p>
          <Link
            className="button button--md button--tactics-primary"
            to="/docs/tournament-system">
            Learn About Tournaments
          </Link>
        </div>
      </div>
    </section>
  );
}

// ===== Unit Card Component =====
function UnitCard({ unit }: { unit: typeof units[0] }): ReactNode {
  return (
    <div className={clsx(styles.unitCard, unit.cardClass)}>
      <div className={styles.unitHeader}>
        <div className={clsx(styles.unitIcon, unit.iconClass)}>
          {unit.code}
        </div>
        <div className={styles.unitInfo}>
          <h3>{unit.name}</h3>
          <p className={styles.unitRole}>{unit.role}</p>
        </div>
      </div>
      <p className={styles.unitDescription}>{unit.description}</p>
      <div className={styles.unitStats}>
        <div className={styles.statItem}>
          <span className={styles.statLabel}>HP</span>
          <span className={styles.statValue}>{unit.stats.hp}</span>
        </div>
        <div className={styles.statItem}>
          <span className={styles.statLabel}>Attack</span>
          <span className={styles.statValue}>{unit.stats.attack}</span>
        </div>
        <div className={styles.statItem}>
          <span className={styles.statLabel}>Defense</span>
          <span className={styles.statValue}>{unit.stats.defense}</span>
        </div>
        <div className={styles.statItem}>
          <span className={styles.statLabel}>Movement</span>
          <span className={styles.statValue}>{unit.stats.movement}</span>
        </div>
      </div>
    </div>
  );
}

// ===== Units Showcase Section =====
function UnitsSection(): ReactNode {
  return (
    <section className={styles.unitsSection}>
      <div className="container">
        <Heading as="h2" className={styles.sectionTitle}>
          Rich Tactical Environment
        </Heading>
        <p className={styles.sectionSubtitle}>
          Four distinct unit types create a complex decision space that challenges AI agents
          to reason about positioning, resource allocation, and opponent modeling.
        </p>
        <div className={styles.unitsGrid}>
          {units.map((unit) => (
            <UnitCard key={unit.name} unit={unit} />
          ))}
        </div>
      </div>
    </section>
  );
}

// ===== Features Section =====
function FeaturesSection(): ReactNode {
  return (
    <section className={styles.featuresSection}>
      <div className="container">
        <Heading as="h2" className={styles.sectionTitle}>
          Built for AI Research
        </Heading>
        <p className={styles.sectionSubtitle}>
          A complete tactical environment designed for reinforcement learning experimentation,
          LLM benchmarking, and AI development.
        </p>
        <div className={styles.featuresGrid}>
          {features.map((feature) => (
            <div key={feature.title} className={styles.featureCard}>
              <div className={styles.featureIcon}>{feature.icon}</div>
              <h3>{feature.title}</h3>
              <p>{feature.description}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

// ===== Documentation Links Section =====
function DocsSection(): ReactNode {
  return (
    <section className={styles.docsSection}>
      <div className="container">
        <Heading as="h2" className={styles.sectionTitle}>
          Explore the Documentation
        </Heading>
        <p className={styles.sectionSubtitle}>
          Everything you need to start evaluating LLMs and training RL agents.
        </p>
        <div className={styles.docsGrid}>
          {docLinks.map((doc) => (
            <Link key={doc.title} to={doc.href} className={styles.docLink}>
              <span className={styles.docLinkIcon}>{doc.icon}</span>
              <div className={styles.docLinkContent}>
                <h4>{doc.title}</h4>
                <p>{doc.description}</p>
              </div>
            </Link>
          ))}
        </div>
      </div>
    </section>
  );
}

// ===== Call to Action Section =====
function CTASection(): ReactNode {
  return (
    <section className={styles.ctaSection}>
      <div className={styles.ctaContent}>
        <Heading as="h2" className={styles.ctaTitle}>
          Start Evaluating Your AI Models
        </Heading>
        <p className={styles.ctaText}>
          Clone the repository, run your first LLM tournament, and discover how different
          models perform on strategic reasoning tasks. Open source and ready for research.
        </p>
        <div className={styles.ctaButtons}>
          <Link
            className="button button--lg button--tactics-primary"
            to="/docs/">
            Read the Docs
          </Link>
          <Link
            className="button button--lg button--tactics-secondary"
            to="https://github.com/kuds/reinforce-tactics">
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
      description="Evaluate GPT-4, Claude, Gemini and other LLMs on strategic reasoning. Open-source turn-based strategy environment for AI research, reinforcement learning, and model benchmarking.">
      <Head>
        {/* Additional SEO meta tags */}
        <meta name="keywords" content="LLM evaluation, LLM benchmark, GPT-5 evaluation, Claude benchmark, Gemini testing, reinforcement learning, AI benchmark, strategic reasoning, tactical AI, game AI, Gymnasium environment, turn-based strategy, AI research, machine learning" />
        <meta name="author" content="Reinforce Tactics" />
        <meta property="og:type" content="website" />
        <meta property="og:site_name" content="Reinforce Tactics" />
        <meta name="twitter:card" content="summary_large_image" />
        <link rel="canonical" href="https://reinforcetactics.com" />
        {/* Structured Data for SEO */}
        <script type="application/ld+json">
          {JSON.stringify(structuredData)}
        </script>
      </Head>
      <HeroSection />
      <main>
        <LLMSection />
        <UnitsSection />
        <FeaturesSection />
        <DocsSection />
        <CTASection />
      </main>
    </Layout>
  );
}
