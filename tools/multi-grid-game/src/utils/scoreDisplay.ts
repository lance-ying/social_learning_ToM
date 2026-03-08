// src/utils/scoreDisplay.ts

/**
 * Utility functions for displaying scores to users.
 *
 * The game tracks two types of scores:
 * 1. Actual scores (stepsRemaining) - can be negative, used for data analysis
 * 2. Displayed scores - clamped to -500 minimum, shown to users
 *
 * This separation allows us to:
 * - Keep accurate backend data for research purposes
 * - Limit displayed scores to a minimum of -500
 */

/**
 * Get the display-friendly version of a score.
 * Returns -500 if the score is below -500, otherwise returns the actual score.
 *
 * @param actualScore - The true score (can be negative)
 * @returns The score to display to the user (minimum -500)
 */
export function getDisplayScore(actualScore: number): number {
  return Math.max(-500, actualScore);
}

/**
 * Calculate displayed points earned, excluding tutorial levels.
 * Tutorial levels (levelIndex 0 and 1) are excluded from the displayed total.
 *
 * @param actualScore - The true score earned
 * @param levelIndex - The index of the level (0-based)
 * @returns The displayed score to show user (0 for tutorials, max(-500, score) for others)
 */
export function getDisplayedPointsForLevel(actualScore: number, levelIndex: number): number {
  // Tutorial levels (0, 1) don't contribute to displayed score
  if (levelIndex < 2) {
    return 0;
  }
  
  // For main experiment trials, return the clamped score
  return getDisplayScore(actualScore);
}

