"use client";

import React, { useEffect, useMemo, useState } from "react";
import { useSearchParams } from "next/navigation";
import EnhancedPathVisualizer from "@/app/components/debug/EnhancedPathVisualizer";
import { loadLevel } from "@/data/levels";

const VALID_PATH_TYPES = new Set([
  "experienced1",
  "experienced2",
  "experienced3",
  "experienced4",
]);

export default function DebugRenderPage() {
  const params = useSearchParams();
  const exp = params.get("exp") || "";
  const jsonLevelId = params.get("jsonLevelId") || "";
  const baseLevelId = params.get("baseLevelId") || "";
  const pathType = params.get("pathType") || "";

  const [gameState, setGameState] = useState<any | null>(null);
  const [error, setError] = useState<string>("");
  const [isReady, setIsReady] = useState(false);

  const resolvedPathType = useMemo(() => {
    if (!VALID_PATH_TYPES.has(pathType)) {
      return "experienced1";
    }
    return pathType;
  }, [pathType]);

  useEffect(() => {
    setIsReady(false);
    setGameState(null);
    setError("");

    try {
      if (!exp || !jsonLevelId || !baseLevelId) {
        throw new Error(
          "Missing required query params: exp, jsonLevelId, baseLevelId",
        );
      }

      const loadedState = loadLevel(baseLevelId);
      const updatedAgents = loadedState.agents.map((agent: any) => {
        const movement = agent.movements?.[resolvedPathType];
        if (!movement) {
          return agent;
        }

        return {
          ...agent,
          selectedPath: resolvedPathType,
          currentPath: movement.path,
          type: movement.type,
        };
      });

      setGameState({
        ...loadedState,
        agents: updatedAgents,
      });
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      setError(message);
    }
  }, [baseLevelId, exp, jsonLevelId, resolvedPathType]);

  useEffect(() => {
    if (!gameState || error) {
      return;
    }

    // Delay readiness slightly so React updates and SVG/icon paints settle.
    const timeout = window.setTimeout(() => {
      setIsReady(true);
    }, 250);

    return () => {
      window.clearTimeout(timeout);
    };
  }, [error, gameState]);

  return (
    <main className="min-h-screen bg-white text-black p-4">
      <div className="max-w-6xl mx-auto">
        <div className="text-sm mb-2" data-testid="render-meta">
          exp={exp} jsonLevelId={jsonLevelId} baseLevelId={baseLevelId} pathType=
          {resolvedPathType}
        </div>

        <div
          data-testid="render-ready"
          data-ready={isReady ? "true" : "false"}
          className="hidden"
        />

        {error ? (
          <pre
            data-testid="render-error"
            className="text-red-700 bg-red-50 border border-red-200 p-3 rounded"
          >
            {error}
          </pre>
        ) : null}

        {gameState ? (
          <EnhancedPathVisualizer
            gameState={gameState}
            initialSelectedPath={resolvedPathType}
            initialMovementIndex={1000}
          />
        ) : null}
      </div>
    </main>
  );
}
