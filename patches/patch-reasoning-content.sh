#!/bin/sh
# Patch @librechat/agents to include reasoning_content for ALL assistant messages,
# not just those with tool_calls. The upstream bug is in
# _convertMessagesToOpenAIParams() in llm/openai/utils/index which only adds
# reasoning_content inside tool_calls branches.

AGENTS_CJS="node_modules/@librechat/agents/dist/cjs/llm/openai/utils/index.cjs"
AGENTS_ESM="node_modules/@librechat/agents/dist/esm/llm/openai/utils/index.mjs"

# Pattern to find: end of the else branch, before the audio check.
# We insert a generic reasoning_content check after line:
#   "if (message.tool_call_id != null) {"
# and before:
#   "if (message.additional_kwargs.audio &&"

# The insertion point is after the closing brace of the tool_call_id block
# but before the audio check.

INSERT_CODE='        }
        if (
          options?.includeReasoningContent === true &&
          message.additional_kwargs.reasoning_content != null &&
          completionParam.reasoning_content == null
        ) {
          completionParam.reasoning_content =
            message.additional_kwargs.reasoning_content;
        }'

patch_file() {
  FILE="$1"
  if [ ! -f "$FILE" ]; then
    echo "File not found: $FILE"
    return 1
  fi

  # Check if already patched
  if grep -q "completionParam.reasoning_content == null" "$FILE" 2>/dev/null; then
    echo "Already patched: $FILE"
    return 0
  fi

  # Find the pattern to insert after
  # The pattern is: closing brace of tool_call_id block, then audio check
  # We insert reasoning_content check between them
  awk '
  /if \(message\.tool_call_id != null\)/ { found_tool_id=1 }
  found_tool_id && /^[[:space:]]*\}[[:space:]]*$/ {
    tool_id_close = NR
    found_tool_id = 0
  }
  /if \(message\.additional_kwargs\.audio/ { audio_line = NR }
  { lines[NR] = $0 }
  END {
    insert_at = 0
    # Find the closing brace before audio check
    for (i = 1; i <= NR; i++) {
      if (i <= tool_id_close && lines[i] ~ /^[[:space:]]*\}[[:space:]]*$/) {
        # Check next non-blank line
        for (j = i+1; j <= NR; j++) {
          if (lines[j] ~ /[^[:space:]]/) {
            if (lines[j] ~ /if \(message\.additional_kwargs\.audio/) {
              insert_at = i
            }
            break
          }
        }
      }
    }
    if (insert_at == 0) {
      print "ERROR: Could not find insertion point" > "/dev/stderr"
      exit 1
    }
    for (i = 1; i <= insert_at; i++) print lines[i]
    print "        }"
    print "        if ("
    print "          options?.includeReasoningContent === true &&"
    print "          message.additional_kwargs.reasoning_content != null &&"
    print "          completionParam.reasoning_content == null"
    print "        ) {"
    print "          completionParam.reasoning_content ="
    print "            message.additional_kwargs.reasoning_content;"
    print "        }"
    for (i = insert_at + 1; i <= NR; i++) print lines[i]
  }' "$FILE" > "${FILE}.tmp" && mv "${FILE}.tmp" "$FILE"
  echo "Patched: $FILE"
}

patch_file "$AGENTS_CJS"
patch_file "$AGENTS_ESM"
