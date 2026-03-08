// Main barrel export for all components
export * from './core';
export * from './ui';
export * from './game-flow';
export * from './debug';
export * from './modals';

// Re-export the main component for convenience
export { default as MultiAgentGame } from './core/MultiAgentGame';