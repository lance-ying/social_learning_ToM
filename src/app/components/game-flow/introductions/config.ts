// Central configuration for introduction variants
// Change this value to switch between different experiment types:
// 'single' - Single treasure chest with one other player (exp2)
// 'three' - Three treasure chests variant
// 'multi' - Single treasure with TWO other players (exp3_true with 3 agents total)
export type ExperimentType = "single" | "three" | "multi" | "exp4";
export const EXPERIMENT_TYPE: ExperimentType = "exp4";

// Legacy support - maps to 'single' or 'three' (for backward compatibility)
// Cast to string to avoid type mismatch warnings
export const USE_SINGLE_TREASURE_VARIANT =
  (EXPERIMENT_TYPE as string) === "single";
