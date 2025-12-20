import type { ReactNode } from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';

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
    title: 'Turn-Based Combat',
    description: 'Strategic grid-based battles with attacks, counter-attacks, paralysis, and healing mechanics inspired by classic tactical RPGs.',
  },
  {
    icon: '🤖',
    title: 'RL-Ready Environment',
    description: 'Full Gymnasium compatibility with multi-discrete action space, configurable rewards, and headless mode for fast training.',
  },
  {
    icon: '🧠',
    title: 'Multiple AI Opponents',
    description: 'Battle against rule-based bots or LLM-powered opponents using GPT, Claude, or Gemini for dynamic challenges.',
  },
  {
    icon: '🏰',
    title: 'Economic Warfare',
    description: 'Capture structures for income, build armies, and manage resources. Control the map to starve your enemy.',
  },
  {
    icon: '📊',
    title: 'Replay & Analysis',
    description: 'Record battles, export replays to video, and analyze strategies. Perfect for research and learning.',
  },
  {
    icon: '🔧',
    title: 'Modular Architecture',
    description: 'Clean, extensible codebase makes it easy to add new units, mechanics, and reward functions for experiments.',
  },
];

// ===== Documentation Links =====
const docLinks = [
  {
    icon: '📖',
    title: 'Getting Started',
    description: 'Installation and quick start guide',
    href: '/docs/',
  },
  {
    icon: '⚔️',
    title: 'Game Mechanics',
    description: 'Units, combat, and structures',
    href: '/docs/game-mechanics',
  },
  {
    icon: '🏆',
    title: 'Tournament System',
    description: 'Run AI tournaments and analyze results',
    href: '/docs/tournament-system',
  },
  {
    icon: '📋',
    title: 'Implementation Status',
    description: 'Features and development roadmap',
    href: '/docs/implementation-status',
  },
];

// ===== Hero Section =====
function HeroSection(): ReactNode {
  return (
    <header className={styles.heroBanner}>
      <div className={styles.heroContent}>
        <span className={styles.heroBadge}>Tactical Strategy + Reinforcement Learning</span>
        <Heading as="h1" className={styles.heroTitle}>
          Reinforce<span className={styles.heroTitleAccent}>Tactics</span>
        </Heading>
        <p className={styles.heroSubtitle}>
          A modular turn-based strategy game designed for developing and testing
          reinforcement learning algorithms. Command your forces, outsmart AI opponents,
          and train the next generation of tactical agents.
        </p>
        <div className={styles.buttons}>
          <Link
            className="button button--lg button--tactics-primary"
            to="/docs/">
            Begin Campaign
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
          Command Your Forces
        </Heading>
        <p className={styles.sectionSubtitle}>
          Four distinct unit types, each with unique abilities and tactical roles.
          Build the perfect army composition for victory.
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
          Built for Research
        </Heading>
        <p className={styles.sectionSubtitle}>
          A complete tactical environment designed for reinforcement learning experimentation
          and AI development.
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
          Everything you need to get started, from installation to advanced RL training.
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
          Ready to Deploy Your Agents?
        </Heading>
        <p className={styles.ctaText}>
          Clone the repository, train your first RL agent, and join the battle.
          Whether you're researching new algorithms or just love tactical games,
          Reinforce Tactics has something for you.
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
      title={`${siteConfig.title} - Tactical RL Environment`}
      description="A modular turn-based strategy game built for reinforcement learning research and experimentation. Command your forces, train AI agents, and conquer the battlefield.">
      <HeroSection />
      <main>
        <UnitsSection />
        <FeaturesSection />
        <DocsSection />
        <CTASection />
      </main>
    </Layout>
  );
}
