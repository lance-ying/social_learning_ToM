#!/usr/bin/env node

/**
 * Generate path visualization PNGs for all levels in the 'new' directory
 * Uses Puppeteer to render a standalone HTML visualizer and capture screenshots
 */

const fs = require('fs');
const path = require('path');
const puppeteer = require('puppeteer');

const LEVELS_DIR = path.join(__dirname, '../src/data/levels/new');
const OUTPUT_DIR = path.join(__dirname, '../visualizations/new_levels');
const VISUALIZER_HTML = path.join(__dirname, 'visualizer_standalone.html');

// Ensure output directory exists
if (!fs.existsSync(OUTPUT_DIR)) {
    fs.mkdirSync(OUTPUT_DIR, { recursive: true });
}

/**
 * Parse a TypeScript level file and extract the level configuration
 */
function parseLevelFile(filePath) {
    const content = fs.readFileSync(filePath, 'utf-8');

    // Extract level ID
    const idMatch = content.match(/id:\s*['"]([^'"]+)['"]/);
    const id = idMatch ? idMatch[1] : path.basename(filePath, '.ts');

    // Extract ASCII map
    const mapMatch = content.match(/asciiMap:\s*`\s*\n([\s\S]*?)\n\s*`\.trim\(\)/);
    const asciiMap = mapMatch ? mapMatch[1] : '';

    // Extract agent paths section
    const pathsMatch = content.match(/agentPaths:\s*\{([\s\S]*?)\n\s*\},\s*stepsRemaining/);

    if (!pathsMatch) {
        console.warn(`Could not parse agentPaths from ${filePath}`);
        return null;
    }

    const agentPathsText = pathsMatch[1];

    // Parse experienced1 path
    const exp1PathMatch = agentPathsText.match(/experienced1:\s*\{\s*path:\s*\[([\s\S]*?)\],\s*goal:\s*(\d+)/);
    const exp1Path = exp1PathMatch ? parsePathArray(exp1PathMatch[1]) : [];
    const exp1Goal = exp1PathMatch ? parseInt(exp1PathMatch[2]) : 1;

    // Parse experienced2 path
    const exp2PathMatch = agentPathsText.match(/experienced2:\s*\{\s*path:\s*\[([\s\S]*?)\],\s*goal:\s*(\d+)/);
    const exp2Path = exp2PathMatch ? parsePathArray(exp2PathMatch[1]) : [];
    const exp2Goal = exp2PathMatch ? parseInt(exp2PathMatch[2]) : 2;

    return {
        id,
        asciiMap,
        agentPaths: {
            1: {
                movements: {
                    experienced1: {
                        path: exp1Path,
                        goal: exp1Goal
                    },
                    experienced2: {
                        path: exp2Path,
                        goal: exp2Goal
                    }
                }
            }
        }
    };
}

/**
 * Parse a path array from TypeScript string format
 */
function parsePathArray(pathStr) {
    const matches = pathStr.match(/"([^"]+)"/g);
    return matches ? matches.map(m => m.replace(/"/g, '')) : [];
}

/**
 * Generate a screenshot for a specific level and path type
 */
async function generateScreenshot(browser, levelData, pathType, outputPath) {
    const page = await browser.newPage();

    try {
        // Set viewport to capture the entire visualization
        await page.setViewport({ width: 1200, height: 1000 });

        // Load the visualizer HTML
        const htmlUrl = `file://${VISUALIZER_HTML}`;
        const encodedLevelData = encodeURIComponent(JSON.stringify(levelData));
        const url = `${htmlUrl}?levelData=${encodedLevelData}&pathType=${pathType}`;

        await page.goto(url, { waitUntil: 'networkidle0' });

        // Wait for visualization to be ready
        await page.waitForFunction(() => window.visualizationReady === true, { timeout: 5000 });

        // Give it a moment to fully render
        await new Promise(resolve => setTimeout(resolve, 500));

        // Take screenshot of the visualization div
        const element = await page.$('#visualization');
        if (element) {
            await element.screenshot({ path: outputPath });
            console.log(`✓ Generated: ${path.basename(outputPath)}`);
        } else {
            console.error(`✗ Could not find visualization element for ${outputPath}`);
        }
    } catch (error) {
        console.error(`✗ Error generating ${outputPath}: ${error.message}`);
    } finally {
        await page.close();
    }
}

/**
 * Main function to process all levels
 */
async function main() {
    console.log('='.repeat(70));
    console.log('Path Visualization Generator');
    console.log('='.repeat(70));
    console.log();

    // Get all .ts files in the new directory
    const files = fs.readdirSync(LEVELS_DIR)
        .filter(f => f.endsWith('.ts') && !f.startsWith('_'))
        .sort();

    if (files.length === 0) {
        console.error('No level files found in', LEVELS_DIR);
        return;
    }

    console.log(`Found ${files.length} level files to process...\n`);

    // Launch browser
    console.log('Launching browser...');
    const browser = await puppeteer.launch({
        headless: 'new',
        args: ['--no-sandbox', '--disable-setuid-sandbox']
    });

    let successCount = 0;
    let failCount = 0;

    for (const file of files) {
        const filePath = path.join(LEVELS_DIR, file);
        const levelId = path.basename(file, '.ts');

        console.log(`\nProcessing: ${levelId}`);
        console.log('-'.repeat(50));

        // Parse level file
        const levelData = parseLevelFile(filePath);

        if (!levelData) {
            console.error(`✗ Failed to parse ${file}`);
            failCount += 2;
            continue;
        }

        const exp1Path = levelData.agentPaths[1].movements.experienced1.path;
        const exp2Path = levelData.agentPaths[1].movements.experienced2.path;

        console.log(`  experienced1: ${exp1Path.length} steps → goal ${levelData.agentPaths[1].movements.experienced1.goal}`);
        console.log(`  experienced2: ${exp2Path.length} steps → goal ${levelData.agentPaths[1].movements.experienced2.goal}`);

        // Generate screenshots for both path types
        const exp1Output = path.join(OUTPUT_DIR, `${levelId}_experienced1.png`);
        const exp2Output = path.join(OUTPUT_DIR, `${levelId}_experienced2.png`);

        await generateScreenshot(browser, levelData, 'experienced1', exp1Output);
        await generateScreenshot(browser, levelData, 'experienced2', exp2Output);

        successCount += 2;
    }

    await browser.close();

    // Summary
    console.log('\n' + '='.repeat(70));
    console.log('SUMMARY');
    console.log('='.repeat(70));
    console.log(`✓ Successfully generated: ${successCount} images`);
    console.log(`✗ Failed: ${failCount} images`);
    console.log(`\nOutput directory: ${OUTPUT_DIR}`);
    console.log('='.repeat(70));
}

// Run the script
main().catch(error => {
    console.error('Fatal error:', error);
    process.exit(1);
});
