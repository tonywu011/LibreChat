/**
 * Patches @librechat/agents to include reasoning_content for ALL assistant messages.
 *
 * The upstream bug: _convertMessagesToOpenAIParams() only adds reasoning_content
 * inside tool_calls branches. Plain assistant messages with 
 * additional_kwargs.reasoning_content silently drop it.
 *
 * Run: node patches/patch-reasoning-content.js
 * Or:  npx patch-package (if patch-package is installed)
 */

const fs = require('fs');
const path = require('path');

const files = [
  'node_modules/@librechat/agents/dist/cjs/llm/openai/utils/index.cjs',
  'node_modules/@librechat/agents/dist/esm/llm/openai/utils/index.mjs',
];

const INSERTION = `        if (
            options?.includeReasoningContent === true &&
            message.additional_kwargs.reasoning_content != null &&
            completionParam.reasoning_content == null
        ) {
            completionParam.reasoning_content =
                message.additional_kwargs.reasoning_content;
        }
`;

for (const relPath of files) {
  const filePath = path.resolve(__dirname, '..', relPath);
  if (!fs.existsSync(filePath)) {
    console.log(`File not found, skipping: ${relPath}`);
    continue;
  }

  let content = fs.readFileSync(filePath, 'utf8');

  // Check if already patched
  if (content.includes('completionParam.reasoning_content == null')) {
    console.log(`Already patched: ${relPath}`);
    continue;
  }

  // Find the insertion point: the closing brace of the else block,
  // right before the audio check
  const audioCheckPattern = 'if (message.additional_kwargs.audio &&';
  const audioIdx = content.indexOf(audioCheckPattern);
  if (audioIdx === -1) {
    console.log(`Could not find audio check pattern in ${relPath}, skipping.`);
    continue;
  }

  // Find the line before the audio check
  const beforeAudio = content.lastIndexOf('\n', audioIdx - 1);
  const lineBefore = content.lastIndexOf('\n', beforeAudio - 1);
  const indent = content.substring(lineBefore + 1, audioIdx);
  
  // The line before audio should be a closing brace
  const closingBraceLine = content.substring(lineBefore + 1, beforeAudio).trim();
  // Check if it ends with '}'
  if (!closingBraceLine.endsWith('}')) {
    console.log(`Unexpected structure before audio check in ${relPath}, skipping.`);
    continue;
  }

  // Insert the reasoning_content check after the closing brace
  const insertPos = beforeAudio + 1;
  const patched = content.slice(0, insertPos) + INSERTION + content.slice(insertPos);
  
  fs.writeFileSync(filePath, patched, 'utf8');
  console.log(`Patched: ${relPath}`);
}
