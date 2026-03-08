#!/usr/bin/env node

const fs = require("fs");
const path = require("path");
const { spawn } = require("child_process");
const puppeteer = require("puppeteer");

const PROJECT_ROOT = path.join(__dirname, "..");
const OUTPUT_ROOT_DEFAULT = path.join(__dirname, "generated_images");
const VALID_EXPERIMENTS = new Set(["exp1", "exp2", "exp3", "exp4"]);
const SCENARIO_TO_PATH = {
  "1": "experienced1",
  "2": "experienced2",
  "3": "experienced3",
};

function parseArgs(argv) {
  const options = {
    experiments: ["exp1", "exp2", "exp3", "exp4"],
    outputRoot: OUTPUT_ROOT_DEFAULT,
    limit: null,
    headful: false,
    port: 3107,
    baseUrl: null,
  };

  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];

    if (arg === "--experiments") {
      const value = argv[i + 1];
      if (!value) {
        throw new Error("Missing value for --experiments");
      }
      i += 1;
      const experiments = value
        .split(",")
        .map((item) => item.trim())
        .filter(Boolean);
      if (experiments.length === 0) {
        throw new Error("--experiments must include at least one experiment");
      }
      const invalid = experiments.filter((exp) => !VALID_EXPERIMENTS.has(exp));
      if (invalid.length > 0) {
        throw new Error(`Invalid experiments: ${invalid.join(", ")}`);
      }
      options.experiments = experiments;
      continue;
    }

    if (arg === "--output") {
      const value = argv[i + 1];
      if (!value) {
        throw new Error("Missing value for --output");
      }
      i += 1;
      options.outputRoot = path.resolve(PROJECT_ROOT, value);
      continue;
    }

    if (arg === "--limit") {
      const value = argv[i + 1];
      if (!value) {
        throw new Error("Missing value for --limit");
      }
      i += 1;
      const limit = Number(value);
      if (!Number.isInteger(limit) || limit <= 0) {
        throw new Error("--limit must be a positive integer");
      }
      options.limit = limit;
      continue;
    }

    if (arg === "--headful") {
      options.headful = true;
      continue;
    }

    if (arg === "--port") {
      const value = argv[i + 1];
      if (!value) {
        throw new Error("Missing value for --port");
      }
      i += 1;
      const port = Number(value);
      if (!Number.isInteger(port) || port <= 0 || port > 65535) {
        throw new Error("--port must be a valid TCP port");
      }
      options.port = port;
      continue;
    }

    if (arg === "--base-url") {
      const value = argv[i + 1];
      if (!value) {
        throw new Error("Missing value for --base-url");
      }
      i += 1;
      options.baseUrl = value.replace(/\/+$/, "");
      continue;
    }

    if (arg === "--help" || arg === "-h") {
      printHelp();
      process.exit(0);
    }

    throw new Error(`Unknown argument: ${arg}`);
  }

  if (!options.baseUrl) {
    options.baseUrl = `http://127.0.0.1:${options.port}`;
  }

  return options;
}

function printHelp() {
  console.log(`
Usage: node scripts/generate_json_level_images.js [options]

Options:
  --experiments exp1,exp2,exp3,exp4   Comma-separated experiments to process
  --output scripts/generated_images    Output directory root
  --limit 10                           Limit total number of screenshots
  --headful                            Run browser in headed mode
  --port 3107                          Port for auto-started Next.js server
  --base-url http://127.0.0.1:3000    Use an existing server instead of auto-starting
  --help                               Show this help
`);
}

function getJsonPathForExperiment(exp) {
  return path.join(__dirname, `${exp}_level_ids.json`);
}

function readJsonLevelIds(exp) {
  const jsonPath = getJsonPathForExperiment(exp);
  if (!fs.existsSync(jsonPath)) {
    throw new Error(
      `Missing ${path.basename(jsonPath)} in scripts/. Copy the JSON files first.`,
    );
  }

  const content = fs.readFileSync(jsonPath, "utf-8");
  const parsed = JSON.parse(content);
  const levelIds = Object.keys(parsed.level_ids || {});
  return levelIds.sort();
}

function mapJsonId(exp, jsonLevelId) {
  if (exp === "exp1") {
    const match = jsonLevelId.match(/^(mod_s\d+)_(\d+)$/);
    if (!match) {
      throw new Error(`Unsupported exp1 ID format: ${jsonLevelId}`);
    }
    const [, base, scenario] = match;
    const pathType = SCENARIO_TO_PATH[scenario];
    if (!pathType) {
      throw new Error(`Unsupported exp1 scenario suffix in ID: ${jsonLevelId}`);
    }
    return { baseLevelId: base, pathType };
  }

  if (exp === "exp2") {
    const match = jsonLevelId.match(/^(s\d+)_(\d+)$/);
    if (!match) {
      throw new Error(`Unsupported exp2 ID format: ${jsonLevelId}`);
    }
    const [, base, scenario] = match;
    const pathType = SCENARIO_TO_PATH[scenario];
    if (!pathType) {
      throw new Error(`Unsupported exp2 scenario suffix in ID: ${jsonLevelId}`);
    }
    return { baseLevelId: base, pathType };
  }

  if (exp === "exp3" || exp === "exp4") {
    const match = jsonLevelId.match(/^(sm[\w\d]+)_scenario(\d+)$/);
    if (!match) {
      throw new Error(`Unsupported ${exp} ID format: ${jsonLevelId}`);
    }
    const [, baseSm, scenario] = match;
    const pathType = SCENARIO_TO_PATH[scenario];
    if (!pathType) {
      throw new Error(`Unsupported ${exp} scenario suffix in ID: ${jsonLevelId}`);
    }
    const baseLevelId = exp === "exp3" ? `${baseSm}_true` : `${baseSm}_exp4`;
    return { baseLevelId, pathType };
  }

  throw new Error(`Unsupported experiment: ${exp}`);
}

function shouldOmitLevelId(jsonLevelId) {
  const match = jsonLevelId.match(/s(?:m)?(\d{3})/);
  return Boolean(match && (match[1] === "111" || match[1] === "112"));
}

function buildJobs(experiments) {
  const jobs = [];
  const mappingErrors = [];
  let omittedCount = 0;

  for (const exp of experiments) {
    const levelIds = readJsonLevelIds(exp);

    for (const jsonLevelId of levelIds) {
      if (shouldOmitLevelId(jsonLevelId)) {
        omittedCount += 1;
        continue;
      }
      try {
        const mapped = mapJsonId(exp, jsonLevelId);
        jobs.push({
          exp,
          jsonLevelId,
          ...mapped,
        });
      } catch (error) {
        mappingErrors.push(
          `${exp}/${jsonLevelId}: ${error instanceof Error ? error.message : String(error)}`,
        );
      }
    }
  }

  return { jobs, mappingErrors, omittedCount };
}

async function sleep(ms) {
  return new Promise((resolve) => {
    setTimeout(resolve, ms);
  });
}

async function isServerReady(baseUrl) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 1500);
  try {
    const response = await fetch(`${baseUrl}/debug/render`, {
      signal: controller.signal,
    });
    return response.ok;
  } catch (error) {
    return false;
  } finally {
    clearTimeout(timeout);
  }
}

async function waitForServer(baseUrl, timeoutMs = 90000) {
  const startedAt = Date.now();
  while (Date.now() - startedAt < timeoutMs) {
    if (await isServerReady(baseUrl)) {
      return;
    }
    await sleep(1000);
  }
  throw new Error(`Timed out waiting for server at ${baseUrl}`);
}

function startNextDevServer(port) {
  return spawn(
    "npm",
    ["run", "dev", "--", "--hostname", "127.0.0.1", "--port", String(port)],
    {
      cwd: PROJECT_ROOT,
      stdio: "inherit",
      env: process.env,
    },
  );
}

async function captureJob(page, baseUrl, job, outputPath) {
  const url = new URL(`${baseUrl}/debug/render`);
  url.searchParams.set("exp", job.exp);
  url.searchParams.set("jsonLevelId", job.jsonLevelId);
  url.searchParams.set("baseLevelId", job.baseLevelId);
  url.searchParams.set("pathType", job.pathType);

  await page.goto(url.toString(), {
    waitUntil: "networkidle2",
    timeout: 120000,
  });

  const errorNode = await page.$('[data-testid="render-error"]');
  if (errorNode) {
    const text = await page.evaluate(
      (node) => node.textContent || "Unknown render error",
      errorNode,
    );
    throw new Error(text.trim());
  }

  await page.waitForSelector('[data-testid="render-ready"][data-ready="true"]', {
    timeout: 120000,
  });
  const vizElement = await page.$('[data-testid="enhanced-path-visualization"]');
  if (!vizElement) {
    throw new Error("Could not locate enhanced visualization element");
  }

  await vizElement.screenshot({ path: outputPath });
}

async function main() {
  const options = parseArgs(process.argv.slice(2));

  console.log("=".repeat(70));
  console.log("JSON Level Image Generator");
  console.log("=".repeat(70));
  console.log(`Experiments: ${options.experiments.join(", ")}`);
  console.log(`Output root: ${options.outputRoot}`);

  const { jobs, mappingErrors, omittedCount } = buildJobs(options.experiments);
  for (const err of mappingErrors) {
    console.error(`Mapping error: ${err}`);
  }

  const filteredJobs = options.limit ? jobs.slice(0, options.limit) : jobs;
  console.log(`Jobs queued: ${filteredJobs.length}`);
  if (mappingErrors.length > 0) {
    console.log(`Jobs skipped due to mapping errors: ${mappingErrors.length}`);
  }
  if (omittedCount > 0) {
    console.log(`Jobs omitted by 111/112 rule: ${omittedCount}`);
  }

  fs.mkdirSync(options.outputRoot, { recursive: true });
  for (const exp of options.experiments) {
    fs.mkdirSync(path.join(options.outputRoot, exp), { recursive: true });
  }

  let devServerProcess = null;
  let serverWasAlreadyRunning = await isServerReady(options.baseUrl);
  if (!serverWasAlreadyRunning) {
    if (options.baseUrl !== `http://127.0.0.1:${options.port}`) {
      throw new Error(
        `No server found at ${options.baseUrl}. Start it manually or omit --base-url for auto-start.`,
      );
    }

    console.log(`Starting Next.js dev server on port ${options.port}...`);
    devServerProcess = startNextDevServer(options.port);
    await waitForServer(options.baseUrl);
  } else {
    console.log(`Using existing server at ${options.baseUrl}`);
  }

  let successCount = 0;
  let failCount = 0;

  const browser = await puppeteer.launch({
    headless: options.headful ? false : "new",
    args: ["--no-sandbox", "--disable-setuid-sandbox"],
  });

  try {
    const page = await browser.newPage();
    await page.setViewport({ width: 1700, height: 1400, deviceScaleFactor: 2 });

    for (let i = 0; i < filteredJobs.length; i += 1) {
      const job = filteredJobs[i];
      const outputFile = path.join(
        options.outputRoot,
        job.exp,
        `stimuli_${job.jsonLevelId}.png`,
      );

      process.stdout.write(
        `[${i + 1}/${filteredJobs.length}] ${job.exp}/${job.jsonLevelId} -> ${path.basename(outputFile)} ... `,
      );

      try {
        await captureJob(page, options.baseUrl, job, outputFile);
        successCount += 1;
        process.stdout.write("OK\n");
      } catch (error) {
        failCount += 1;
        process.stdout.write("FAIL\n");
        console.error(
          `  ${job.exp}/${job.jsonLevelId}: ${error instanceof Error ? error.message : String(error)}`,
        );
      }
    }
  } finally {
    await browser.close();

    if (devServerProcess) {
      devServerProcess.kill("SIGTERM");
      await sleep(1500);
      if (!devServerProcess.killed) {
        devServerProcess.kill("SIGKILL");
      }
    }
  }

  console.log("\n" + "=".repeat(70));
  console.log("SUMMARY");
  console.log("=".repeat(70));
  console.log(`Success: ${successCount}`);
  console.log(`Failed: ${failCount}`);
  console.log(`Output root: ${options.outputRoot}`);
  console.log("=".repeat(70));

  if (failCount > 0 || mappingErrors.length > 0) {
    process.exitCode = 1;
  }
}

main().catch((error) => {
  console.error("Fatal error:", error);
  process.exit(1);
});
